import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import os
from datetime import datetime, timedelta
import base64
import time
from typing import Dict, List, Any, Optional
from scipy import stats
from fpdf import FPDF
import tempfile
from io import BytesIO
import zipfile

# ✅ CORREÇÃO: Controle de estado para abas
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🏠 Dashboard"

def set_active_tab(tab_name):
    """Define a aba ativa na sessão"""
    st.session_state.active_tab = tab_name

# Adicionar o src ao path para importar nossos módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Quantum Risk Analytics - Multiagente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FUNÇÕES DE EXPORTAÇÃO E GRÁFICOS AVANÇADOS
# =============================================

def _fig_png(fig) -> bytes:
    """Converte figura Plotly para PNG via kaleido."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        # fallback para HTML se kaleido não estiver disponível
        return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


def _fig_mime(fig) -> tuple[bytes, str, str]:
    """Retorna (bytes, mime_type, extensão) — PNG se kaleido ok, HTML senão."""
    try:
        return fig.to_image(format="png", scale=2), "image/png", "png"
    except Exception:
        return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"), "text/html", "html"


def _dl_chart_btn(label: str, fig, basename: str, key: str, **kw):
    """Renderiza st.download_button para um gráfico Plotly (PNG ou HTML)."""
    _b, _m, _e = _fig_mime(fig)
    ext_label = label.replace("(HTML)", f"({_e.upper()})")
    st.download_button(ext_label, _b, f"{basename}.{_e}", _m, key=key, **kw)


def _build_export_excel(simulation_results: dict, initial_investment: float) -> BytesIO:
    """Gera Excel multi-aba com todos os dados das simulações."""
    output = BytesIO()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        curr_fmt = wb.add_format({"num_format": "R$ #,##0.00"})
        pct_fmt  = wb.add_format({"num_format": "0.00%"})

        # --- Resumo executivo ---
        summary = []
        for method, r in simulation_results.items():
            summary.append({
                "Método": method,
                "VaR 95% (R$)": abs(r.get("var", 0)),
                "CVaR 95% (R$)": abs(r.get("cvar", 0)),
                "Prob. Prejuízo": r.get("probability_loss", 0),
                "Retorno Esperado (R$)": r.get("expected_return", 0),
                "Simulações": r.get("simulations", 0),
                "Horizonte (dias)": r.get("time_horizon", 0),
            })
        df_summary = pd.DataFrame(summary)
        df_summary.to_excel(writer, sheet_name="Resumo", index=False)
        ws = writer.sheets["Resumo"]
        ws.write(0, 8, f"Gerado em: {ts}")
        ws.write(1, 8, f"Investimento Inicial: R$ {initial_investment:,.2f}")
        ws.set_column("B:C", 18, curr_fmt)
        ws.set_column("D:D", 16, pct_fmt)
        ws.set_column("E:E", 22, curr_fmt)

        # --- Estatísticas descritivas ---
        stats_rows = []
        for method, r in simulation_results.items():
            if "final_values" in r and len(r["final_values"]) > 0:
                fv = np.array(r["final_values"])
                losses = initial_investment - fv
                stats_rows.append({
                    "Método": method,
                    "Média Valor Final": float(np.mean(fv)),
                    "Mediana Valor Final": float(np.median(fv)),
                    "Desvio Padrão": float(np.std(fv)),
                    "Mínimo": float(np.min(fv)),
                    "Máximo": float(np.max(fv)),
                    "VaR 95%": abs(r.get("var", 0)),
                    "CVaR 95%": abs(r.get("cvar", 0)),
                    "Prob. Prejuízo": r.get("probability_loss", 0),
                    "Retorno Esperado": r.get("expected_return", 0),
                    "Skewness": float(pd.Series(fv).skew()),
                    "Kurtosis": float(pd.Series(fv).kurt()),
                    "Perda Média": float(np.mean(losses)),
                    "Perda Máxima": float(np.max(losses)),
                })
        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            df_stats.to_excel(writer, sheet_name="Estatísticas", index=False)

        # --- Dados brutos por método ---
        for method, r in simulation_results.items():
            if "final_values" in r and len(r["final_values"]) > 0:
                fv = np.array(r["final_values"])
                df_m = pd.DataFrame({
                    "Valor Final (R$)": fv,
                    "Perda (R$)": initial_investment - fv,
                    "Retorno (%)": (fv - initial_investment) / initial_investment,
                })
                sheet = (method[:27] + "_sim")[:31]
                df_m.to_excel(writer, sheet_name=sheet, index=False)

        # --- Stress Analysis ---
        stress_rows = []
        for method, r in simulation_results.items():
            if "var" in r and "probability_loss" in r:
                var_impact = abs(r["var"]) / initial_investment
                loss_prob = r["probability_loss"]
                if loss_prob > 0.3 or var_impact > 0.2:
                    nivel = "ALTO"
                elif loss_prob > 0.15 or var_impact > 0.1:
                    nivel = "MÉDIO"
                else:
                    nivel = "BAIXO"
                stress_rows.append({
                    "Método": method,
                    "Impacto VaR (%)": var_impact,
                    "Prob. Prejuízo": loss_prob,
                    "Nível de Stress": nivel,
                })
        if stress_rows:
            df_stress = pd.DataFrame(stress_rows)
            df_stress.to_excel(writer, sheet_name="Stress", index=False)

    output.seek(0)
    return output


def display_simulation_results(simulation_results: dict, initial_investment: float):
    """Exibe resultados das simulações com gráficos profissionais para mercado financeiro - CORRIGIDA"""
    
    if not simulation_results:
        st.warning("⚠️ Nenhum resultado de simulação disponível")
        return
    
    # ✅ CORREÇÃO: Garantir que simulation_results está na sessão
    if 'current_simulation_results' not in st.session_state:
        st.session_state.current_simulation_results = simulation_results
    else:
        # Atualizar se novos resultados chegarem
        st.session_state.current_simulation_results = simulation_results
    
    st.markdown("### 📊 Dashboard de Risco - Análise Profissional")
    
    # Usar resultados da sessão
    current_results = st.session_state.current_simulation_results
    
    # Métricas rápidas em cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        var_values = [abs(results['var']) for results in current_results.values() if 'var' in results]
        avg_var = np.mean(var_values) if var_values else 0
        st.metric(
            "Value at Risk (Médio)",
            f"R$ {avg_var:,.0f}",
            delta=f"{(avg_var/initial_investment):.1%}"
        )
    
    with col2:
        prob_loss_values = [results['probability_loss'] for results in current_results.values() if 'probability_loss' in results]
        avg_prob_loss = np.mean(prob_loss_values) if prob_loss_values else 0
        st.metric(
            "Prob. Prejuízo (Média)",
            f"{avg_prob_loss:.1%}",
            delta="ALTO RISCO" if avg_prob_loss > 0.2 else "MODERADO",
            delta_color="inverse"
        )
    
    with col3:
        expected_returns = [results['expected_return'] for results in current_results.values() if 'expected_return' in results]
        avg_expected_return = np.mean(expected_returns) if expected_returns else 0
        st.metric(
            "Retorno Esperado (Médio)",
            f"R$ {avg_expected_return:,.0f}",
            delta=f"{(avg_expected_return/initial_investment):.1%}"
        )
    
    with col4:
        total_simulations = sum(results.get('simulations', 0) for results in current_results.values())
        st.metric("Cenários Simulados", f"{total_simulations:,}")

    # Botão "Exportar tudo" sempre visível, acima das abas
    st.markdown("---")
    _exp_col, _ = st.columns([1, 3])
    with _exp_col:
        _excel_bytes = _build_export_excel(current_results, initial_investment)
        st.download_button(
            label="⬇️ Exportar tudo (Excel multi-aba)",
            data=_excel_bytes,
            file_name=f"quantum_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="export_tudo_top",
            help="Resumo, estatísticas, dados brutos por método e análise de stress em uma planilha.",
            use_container_width=True,
        )
    st.markdown("---")

    # Abas para diferentes visualizações
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribuição de Perdas", 
        "🔄 Trajetórias", 
        "📊 Comparação de Métodos",
        "🎯 Análise de Stress",
        "📤 Exportar Dados"
    ])
    
    with tab1:
        display_loss_distribution(current_results, initial_investment)
    
    with tab2:
        display_simulation_paths(current_results)
    
    with tab3:
        display_methods_comparison(current_results)
    
    with tab4:
        display_stress_analysis(current_results, initial_investment)
    
    with tab5:
        display_export_options(current_results, initial_investment)

def display_loss_distribution(simulation_results: dict, initial_investment: float):
    """Gráfico de distribuição de perdas - VaR/CVaR - CORRIGIDA"""
    st.markdown("#### 📉 Distribuição de Perdas - VaR & CVaR")
    
    # ✅ CORREÇÃO: Usar key única
    methods = list(simulation_results.keys())
    if not methods:
        st.info("Nenhum método disponível para análise")
        return
        
    selected_method = st.selectbox(
        "Selecione o método para análise detalhada:", 
        methods,
        key="loss_distribution_method_selector"  # ✅ KEY ÚNICA
    )
    
    if selected_method and selected_method in simulation_results:
        results = simulation_results[selected_method]
        
        if 'final_values' in results and results['final_values'] is not None and len(results['final_values']) > 0:
            final_values = results['final_values']
            losses = initial_investment - final_values
            
            # ✅ CORREÇÃO: Layout responsivo
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de distribuição com VaR/CVaR
                fig1 = go.Figure()
                
                fig1.add_trace(go.Histogram(
                    x=losses, 
                    nbinsx=50,
                    name='Distribuição de Perdas',
                    marker_color='lightcoral',
                    opacity=0.7
                ))
                
                # Linhas de VaR e CVaR
                var_95 = abs(results.get('var', 0))
                cvar_95 = abs(results.get('cvar', 0))
                
                fig1.add_vline(
                    x=var_95, line_dash="dash", line_color="red",
                    annotation_text=f"VaR 95%: R$ {var_95:,.0f}"
                )
                fig1.add_vline(
                    x=cvar_95, line_dash="dash", line_color="darkred",
                    annotation_text=f"CVaR 95%: R$ {cvar_95:,.0f}"
                )
                
                fig1.update_layout(
                    title="Distribuição de Perdas com VaR/CVaR",
                    xaxis_title="Perda (R$)",
                    yaxis_title="Frequência",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                _dl_chart_btn("⬇️ Distribuição (HTML)", fig1, "distribuicao_perdas", f"dl_fig1_{selected_method}", use_container_width=True)

            with col2:
                # CDF (Função de Distribuição Acumulada)
                sorted_losses = np.sort(losses)
                cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=sorted_losses, y=cdf,
                    mode='lines', name='CDF',
                    line=dict(color='blue', width=3),
                    fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'
                ))
                
                # Marcar VaR na CDF
                var_percentile = np.searchsorted(sorted_losses, var_95) / len(sorted_losses)
                fig2.add_trace(go.Scatter(
                    x=[var_95, var_95], y=[0, var_percentile],
                    mode='lines', line=dict(dash='dash', color='red'),
                    name=f'VaR 95% ({var_percentile:.1%})',
                    showlegend=True
                ))
                
                fig2.update_layout(
                    title="Função de Distribuição Acumulada (CDF)",
                    xaxis_title="Perda (R$)",
                    yaxis_title="Probabilidade Acumulada",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                _dl_chart_btn("⬇️ CDF (HTML)", fig2, "cdf_perdas", f"dl_fig2_{selected_method}", use_container_width=True)

            # ✅ CORREÇÃO: Estatísticas detalhadas
            st.markdown("##### 📊 Estatísticas Detalhadas")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Perda Média", f"R$ {np.mean(losses):,.0f}")
            with stats_col2:
                st.metric("Perda Máxima", f"R$ {np.max(losses):,.0f}")
            with stats_col3:
                st.metric("Desvio Padrão", f"R$ {np.std(losses):,.0f}")
            with stats_col4:
                st.metric("Skewness", f"{stats.skew(losses):.2f}")
                
        else:
            st.warning(f"⚠️ Dados finais não disponíveis para análise de {selected_method}")

def display_simulation_paths(simulation_results: dict):
    """Gráfico de trajetórias simuladas - CORRIGIDA"""
    st.markdown("#### 🔄 Trajetórias Simuladas")
    
    # ✅ CORREÇÃO: Usar key única para evitar conflitos
    methods = list(simulation_results.keys())
    if not methods:
        st.warning("Nenhum método de simulação disponível")
        return
        
    selected_method = st.selectbox(
        "Método para trajetórias:", 
        methods,
        key="simulation_paths_method_selector"  # ✅ KEY ÚNICA
    )
    
    if selected_method and selected_method in simulation_results:
        results = simulation_results[selected_method]
        
        # ✅ CORREÇÃO: Verificar se existem dados de trajetórias
        if 'simulation_paths' in results and results['simulation_paths'] is not None:
            paths = results['simulation_paths']
            
            # ✅ CORREÇÃO: Verificar a dimensionalidade dos dados
            if hasattr(paths, 'ndim') and paths.ndim == 2:
                
                # ✅ CORREÇÃO: Configurações com keys únicas
                col1, col2 = st.columns(2)
                with col1:
                    n_paths = st.slider(
                        "Número de trajetórias a mostrar:", 
                        10, 200, 50,
                        key=f"n_paths_{selected_method}"  # ✅ KEY ÚNICA
                    )
                with col2:
                    show_percentiles = st.checkbox(
                        "Mostrar percentis", 
                        value=True,
                        key=f"show_percentiles_{selected_method}"  # ✅ KEY ÚNICA
                    )
                
                fig = go.Figure()
                
                # Trajetórias individuais (amostra)
                n_paths_to_show = min(n_paths, paths.shape[1])
                for i in range(n_paths_to_show):
                    fig.add_trace(go.Scatter(
                        x=list(range(paths.shape[0])),
                        y=paths[:, i],
                        mode='lines',
                        line=dict(width=1, color='lightblue'),
                        opacity=0.3,  # ✅ CORREÇÃO: opacity vai aqui, não no line
                        showlegend=False,
                        name=f'Trajetória {i+1}'
                    ))
                
                # Percentis
                if show_percentiles:
                    percentiles = [5, 25, 50, 75, 95]
                    colors = ['red', 'orange', 'green', 'orange', 'red']
                    
                    for p, color in zip(percentiles, colors):
                        percentile_values = np.percentile(paths, p, axis=1)
                        fig.add_trace(go.Scatter(
                            x=list(range(paths.shape[0])),
                            y=percentile_values,
                            mode='lines',
                            line=dict(width=3, color=color),
                            name=f'Percentil {p}%'
                        ))
                
                # Média
                mean_path = np.mean(paths, axis=1)
                fig.add_trace(go.Scatter(
                    x=list(range(paths.shape[0])),
                    y=mean_path,
                    mode='lines',
                    line=dict(width=4, color='black', dash='dash'),
                    name='Média'
                ))
            
                fig.update_layout(
                    title=f"Trajetórias Simuladas - {selected_method}",
                    xaxis_title="Dias",
                    yaxis_title="Valor do Portfólio (R$)",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                _dl_chart_btn("⬇️ Trajetórias (HTML)", fig, f"trajetorias_{selected_method}", f"dl_paths_{selected_method}")
            else:
                st.warning(f"⚠️ Dados de trajetórias não disponíveis para {selected_method}")
        else:
            st.warning(f"⚠️ Nenhum dado de simulação disponível para {selected_method}")
            
def display_methods_comparison(simulation_results: dict):
    """Comparação entre diferentes métodos de simulação"""
    st.markdown("#### 📊 Comparação de Métodos de Simulação")
    
    if len(simulation_results) < 2:
        st.info("⚠️ Execute pelo menos 2 métodos diferentes para comparação")
        return
    
    # Dados para comparação
    methods_data = []
    for method, results in simulation_results.items():
        if 'var' in results and 'probability_loss' in results:
            methods_data.append({
                'Método': method,
                'VaR (R$)': abs(results['var']),
                'CVaR (R$)': abs(results.get('cvar', 0)),
                'Prob. Prejuízo': results['probability_loss'],
                'Retorno Esperado (R$)': results.get('expected_return', 0),
                'Simulações': results.get('simulations', 0)
            })
    
    if methods_data:
        methods_df = pd.DataFrame(methods_data)
        
        # Gráfico de comparação - Violin plot se tivermos dados de distribuição
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras para VaR
            fig_var = px.bar(
                methods_df,
                x='Método',
                y='VaR (R$)',
                title='Value at Risk por Método',
                color='Prob. Prejuízo',
                color_continuous_scale='RdYlGn_r'
            )
            fig_var.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_var, use_container_width=True)
            _dl_chart_btn("⬇️ VaR por Método (HTML)", fig_var, "var_por_metodo", "dl_fig_var_comp", use_container_width=True)

        with col2:
            # Scatter plot: VaR vs Prob. Prejuízo
            fig_scatter = px.scatter(
                methods_df,
                x='VaR (R$)',
                y='Prob. Prejuízo',
                size='Simulações',
                color='Método',
                title='Relação: VaR vs Probabilidade de Prejuízo',
                hover_data=['Retorno Esperado (R$)']
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            _dl_chart_btn("⬇️ VaR vs Prob (HTML)", fig_scatter, "var_vs_prob", "dl_fig_scatter_comp", use_container_width=True)

        # Tabela comparativa
        st.markdown("#### 📋 Métricas Comparativas")
        display_df = methods_df.copy()
        display_df['Prob. Prejuízo'] = display_df['Prob. Prejuízo'].apply(lambda x: f"{x:.2%}")
        display_df['VaR (R$)'] = display_df['VaR (R$)'].apply(lambda x: f"R$ {x:,.0f}")
        display_df['CVaR (R$)'] = display_df['CVaR (R$)'].apply(lambda x: f"R$ {x:,.0f}")
        display_df['Retorno Esperado (R$)'] = display_df['Retorno Esperado (R$)'].apply(lambda x: f"R$ {x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)

def display_stress_analysis(simulation_results: dict, initial_investment: float):
    """Análise de stress testing"""
    st.markdown("#### 🎯 Análise de Cenários de Stress")
    
    # Calcular métricas de stress para cada método
    stress_data = []
    for method, results in simulation_results.items():
        if 'probability_loss' in results and 'var' in results:
            var_impact = abs(results['var']) / initial_investment
            loss_prob = results['probability_loss']
            
            # Classificar o nível de stress
            if loss_prob > 0.3 or var_impact > 0.2:
                stress_level = "ALTO"
                color = "red"
            elif loss_prob > 0.15 or var_impact > 0.1:
                stress_level = "MÉDIO"
                color = "orange"
            else:
                stress_level = "BAIXO"
                color = "green"
            
            stress_data.append({
                'Método': method,
                'Impacto VaR': var_impact,
                'Prob. Prejuízo': loss_prob,
                'Nível Stress': stress_level,
                'Cor': color
            })
    
    if stress_data:
        stress_df = pd.DataFrame(stress_data)
        
        # Heatmap de stress
        fig = px.scatter(
            stress_df,
            x='Impacto VaR',
            y='Prob. Prejuízo',
            color='Nível Stress',
            size=[20] * len(stress_df),
            title='Mapa de Calor de Risco - Análise de Stress',
            color_discrete_map={'ALTO': 'red', 'MÉDIO': 'orange', 'BAIXO': 'green'},
            hover_data=['Método']
        )
        
        # Adicionar zonas de risco
        fig.add_shape(type="rect", x0=0, y0=0.3, x1=0.2, y1=1, fillcolor="red", opacity=0.1, line=dict(color="Red"))
        fig.add_shape(type="rect", x0=0, y0=0.15, x1=0.1, y1=0.3, fillcolor="orange", opacity=0.1, line=dict(color="Orange"))
        fig.add_shape(type="rect", x0=0, y0=0, x1=0.1, y1=0.15, fillcolor="green", opacity=0.1, line=dict(color="Green"))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        
        fig.update_xaxes(tickformat=".1%", title="Impacto do VaR (% do Portfólio)")
        fig.update_yaxes(tickformat=".0%", title="Probabilidade de Prejuízo")
        
        st.plotly_chart(fig, use_container_width=True)
        _dl_chart_btn("⬇️ Mapa de Stress (HTML)", fig, "stress_analysis", "dl_stress_scatter")

        # Recomendações baseadas no stress
        st.markdown("#### 💡 Recomendações de Gestão de Risco")
        
        high_stress_methods = [d['Método'] for d in stress_data if d['Nível Stress'] == 'ALTO']
        if high_stress_methods:
            st.error(f"**ALERTA:** Métodos {', '.join(high_stress_methods)} indicam alto risco. Considere:")
            st.write("• Reduzir exposição ao risco")
            st.write("• Implementar hedges de proteção")
            st.write("• Revisar alocação de ativos")
        
        medium_stress_methods = [d['Método'] for d in stress_data if d['Nível Stress'] == 'MÉDIO']
        if medium_stress_methods:
            st.warning(f"**ATENÇÃO:** Métodos {', '.join(medium_stress_methods)} indicam risco moderado.")
            st.write("• Monitorar posições regularmente")
            st.write("• Estabelecer stops de proteção")
        
        low_stress_methods = [d['Método'] for d in stress_data if d['Nível Stress'] == 'BAIXO']
        if low_stress_methods:
            st.success(f"**ESTÁVEL:** Métodos {', '.join(low_stress_methods)} indicam baixo risco.")

def display_export_options(simulation_results: dict, initial_investment: float):
    """Opções de exportação de dados"""
    st.markdown("#### 📤 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exportar para Excel", use_container_width=True, key="export_excel"):
            export_to_excel(simulation_results, initial_investment)
    
    with col2:
        if st.button("📄 Gerar Relatório PDF", use_container_width=True, key="export_pdf"):
            export_to_pdf(simulation_results, initial_investment)
    
    with col3:
        if st.button("🖼️ Exportar Gráficos", use_container_width=True, key="export_charts"):
            export_charts(simulation_results)
    
    # Exportação Rápida
    st.markdown("---")
    st.markdown("### 🚀 Exportação Rápida")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📋 Relatório Executivo", help="PDF resumido para tomada de decisão", key="executive_summary"):
            export_executive_summary(simulation_results, initial_investment)
    
    with quick_col2:
        if st.button("📈 Dados para Análise", help="Excel com todos os dados para análise detalhada", key="analysis_data"):
            export_analysis_data(simulation_results, initial_investment)
    
    with quick_col3:
        if st.button("🖼️ Portfolio de Gráficos", help="Todos os gráficos em alta resolução", key="all_charts"):
            export_all_charts(simulation_results)

def export_to_excel(simulation_results: dict, initial_investment: float):
    """Exporta resultados para Excel"""
    try:
        # Criar um arquivo Excel com múltiplas abas
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Aba de métricas resumidas
            metrics_data = []
            for method, results in simulation_results.items():
                metrics_data.append({
                    'Método': method,
                    'VaR (R$)': abs(results.get('var', 0)),
                    'CVaR (R$)': abs(results.get('cvar', 0)),
                    'Prob. Prejuízo': results.get('probability_loss', 0),
                    'Retorno Esperado (R$)': results.get('expected_return', 0),
                    'Simulações': results.get('simulations', 0),
                    'Horizonte (dias)': results.get('time_horizon', 0)
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Métricas_Resumidas', index=False)
            
            # Aba de dados detalhados por método
            for method, results in simulation_results.items():
                if 'final_values' in results and len(results['final_values']) > 0:
                    detailed_data = {
                        'Valores_Finais': results['final_values'],
                        'Perdas': initial_investment - results['final_values']
                    }
                    detailed_df = pd.DataFrame(detailed_data)
                    sheet_name = method[:31]  # Limite de 31 caracteres
                    detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Formatação
            workbook = writer.book
            worksheet = writer.sheets['Métricas_Resumidas']
            
            # Formatos
            currency_format = workbook.add_format({'num_format': 'R$ #,##0'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            number_format = workbook.add_format({'num_format': '#,##0'})
            
            # Aplicar formatação
            worksheet.set_column('B:C', 15, currency_format)
            worksheet.set_column('D:D', 15, percent_format)
            worksheet.set_column('E:E', 20, currency_format)
            worksheet.set_column('F:G', 12, number_format)
        
        output.seek(0)
        
        # Botão de download
        st.download_button(
            label="⬇️ Baixar Arquivo Excel",
            data=output,
            file_name=f"analise_risco_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
        
    except Exception as e:
        st.error(f"❌ Erro ao exportar para Excel: {e}")

def export_to_pdf(simulation_results: dict, initial_investment: float):
    """Relatório PDF consolidado: simulações + métricas por ativo + alertas + LSTM + Autoencoder."""
    try:
        if not simulation_results:
            st.error("❌ Nenhum dado de simulação disponível")
            return

        pdf = FPDF()

        def _header(pdf, title):
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.set_fill_color(26, 26, 46)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 12, title, 0, 1, "C", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

        def _section(pdf, title):
            pdf.set_font("Arial", "B", 11)
            pdf.set_fill_color(230, 230, 240)
            pdf.cell(0, 8, title, 0, 1, "L", fill=True)
            pdf.set_font("Arial", "", 10)
            pdf.ln(2)

        def _table_row(pdf, cells, widths, bold=False):
            pdf.set_font("Arial", "B" if bold else "", 9)
            for cell, w in zip(cells, widths):
                pdf.cell(w, 7, str(cell), 1)
            pdf.ln()

        # ── Capa ──────────────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Arial", "B", 20)
        pdf.ln(30)
        pdf.cell(0, 14, "QUANTUM RISK ANALYTICS", 0, 1, "C")
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Relatório Consolidado de Análise de Risco", 0, 1, "C")
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Data: {datetime.now().strftime('%d/%m/%Y  %H:%M')}", 0, 1, "C")
        pdf.cell(0, 8, f"Investimento inicial: R$ {initial_investment:,.2f}", 0, 1, "C")

        # ── Seção 1: Simulações ───────────────────────────────────────────────
        _header(pdf, "1. SIMULAÇÕES DE MONTE CARLO")
        _section(pdf, "Métricas por método de simulação")
        cols = ["Método", "VaR (R$)", "CVaR (R$)", "Prob. Perd.", "Ret. Esp. (R$)", "Simulações"]
        widths = [52, 28, 28, 22, 36, 24]
        _table_row(pdf, cols, widths, bold=True)
        for method, r in simulation_results.items():
            _table_row(pdf, [
                method[:30],
                f"R$ {abs(r.get('var', 0)):,.0f}",
                f"R$ {abs(r.get('cvar', 0)):,.0f}",
                f"{r.get('probability_loss', 0):.1%}",
                f"R$ {r.get('expected_return', 0):,.0f}",
                f"{r.get('simulations', 0):,}",
            ], widths)

        # ── Seção 2: Métricas por ativo ───────────────────────────────────────
        orch = st.session_state.get("orchestrator")
        report = st.session_state.get("analysis_report") or st.session_state.get("report")
        returns_df = st.session_state.get("returns_filtered") or st.session_state.get("returns")

        if returns_df is not None and not returns_df.empty:
            _header(pdf, "2. MÉTRICAS POR ATIVO")
            _section(pdf, "Retorno anual, volatilidade, Sharpe, Max DD, VaR, CVaR")
            cols2 = ["Ativo", "Ret. Anual", "Volat.", "Sharpe", "Max DD", "VaR 95%", "CVaR 95%"]
            widths2 = [40, 22, 18, 18, 18, 22, 22]
            _table_row(pdf, cols2, widths2, bold=True)
            rf = st.session_state.get("risk_free_rate", 0.1075)
            cl = st.session_state.get("confidence_level", 0.95)
            for asset in returns_df.columns:
                s = returns_df[asset].dropna()
                ann_ret = s.mean() * 252
                ann_vol = s.std() * np.sqrt(252)
                sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
                cum = (1 + s).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                var = np.percentile(s, (1 - cl) * 100)
                cvar = s[s <= var].mean()
                _table_row(pdf, [asset, f"{ann_ret:.1%}", f"{ann_vol:.1%}", f"{sharpe:.2f}",
                                  f"{mdd:.1%}", f"{var:.2%}", f"{cvar:.2%}"], widths2)

        # ── Seção 3: Alertas ──────────────────────────────────────────────────
        all_alerts = []
        if report and "all_alerts" in report:
            all_alerts = report["all_alerts"]
        if all_alerts:
            _header(pdf, "3. ALERTAS ATIVOS")
            sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            sorted_alerts = sorted(all_alerts, key=lambda x: sev_order.get(x.get("severity", "low"), 0), reverse=True)
            _section(pdf, f"Total: {len(sorted_alerts)} alertas")
            cols3 = ["Severidade", "Agente", "Score", "Mensagem"]
            widths3 = [22, 35, 15, 118]
            _table_row(pdf, cols3, widths3, bold=True)
            for a in sorted_alerts[:30]:  # limite p/ não explodir o PDF
                msg = str(a.get("message", ""))[:80]
                _table_row(pdf, [
                    a.get("severity", "").upper()[:10],
                    a.get("agent_id", "")[:20],
                    f"{a.get('score', 0):.2f}",
                    msg,
                ], widths3)
            if len(sorted_alerts) > 30:
                pdf.set_font("Arial", "I", 9)
                pdf.cell(0, 6, f"... e mais {len(sorted_alerts) - 30} alertas (veja exportação Excel).", 0, 1)

        # ── Seção 4: LSTM ─────────────────────────────────────────────────────
        lstm_res = {}
        if orch and hasattr(orch, "agent_lstm") and orch.agent_lstm:
            lstm_res = getattr(orch.agent_lstm, "lstm_results", {}) or {}
        if lstm_res:
            _header(pdf, "4. LSTM — PREVISÃO DE TENDÊNCIA")
            _section(pdf, "Resumo por ativo")
            cols4 = ["Ativo", "Tendência", "Média/dia", "MAE", "d+1", "d+2", "d+3", "d+4", "d+5"]
            widths4 = [30, 20, 20, 18, 18, 18, 18, 18, 20]
            _table_row(pdf, cols4, widths4, bold=True)
            for _a, _r in lstm_res.items():
                fc = _r.get("forecast", [None]*5)
                _table_row(pdf, [
                    _a[:18],
                    _r.get("trend", "").upper(),
                    f"{_r.get('trend_avg', 0):+.3%}",
                    f"{_r.get('mae', 0):.4f}",
                    f"{fc[0]:.3%}" if len(fc) > 0 and fc[0] is not None else "-",
                    f"{fc[1]:.3%}" if len(fc) > 1 and fc[1] is not None else "-",
                    f"{fc[2]:.3%}" if len(fc) > 2 and fc[2] is not None else "-",
                    f"{fc[3]:.3%}" if len(fc) > 3 and fc[3] is not None else "-",
                    f"{fc[4]:.3%}" if len(fc) > 4 and fc[4] is not None else "-",
                ], widths4)

        # ── Seção 5: Autoencoder ──────────────────────────────────────────────
        ae_res = {}
        if orch and hasattr(orch, "agent_autoencoder") and orch.agent_autoencoder:
            ae_res = getattr(orch.agent_autoencoder, "autoencoder_results", {}) or {}
        if ae_res:
            _header(pdf, "5. AUTOENCODER — DETECÇÃO DE ANOMALIAS")
            _section(pdf, "Resumo")
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 7, f"Anomalias detectadas: {ae_res.get('n_anomalies', 0)}", 0, 1)
            pdf.cell(0, 7, f"Anomalias últimos 5 dias: {ae_res.get('recent_anomalies', 0)}", 0, 1)
            pdf.cell(0, 7, f"Taxa de anomalia: {ae_res.get('contamination', 0):.2%}", 0, 1)
            pdf.cell(0, 7, f"Threshold MSE: {ae_res.get('threshold', 0):.6f}", 0, 1)
            if "errors_by_asset" in ae_res:
                pdf.ln(3)
                _section(pdf, "Erro de reconstrução por ativo")
                cols5 = ["Ativo", "MSE Médio", "Anômalo?"]
                widths5 = [60, 40, 40]
                _table_row(pdf, cols5, widths5, bold=True)
                mean_err = np.mean(list(ae_res["errors_by_asset"].values()))
                for asset, err in ae_res["errors_by_asset"].items():
                    _table_row(pdf, [asset, f"{err:.6f}", "SIM" if err > 2 * mean_err else "NÃO"], widths5)

        # ── Gerar e disponibilizar ─────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            pdf_data = open(tmp.name, "rb").read()
        os.unlink(tmp.name)

        st.download_button(
            label="⬇️ Baixar Relatório Consolidado (PDF)",
            data=pdf_data,
            file_name=f"relatorio_consolidado_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key="pdf_export_button",
        )

    except Exception as e:
        st.error(f"❌ Erro ao gerar PDF: {e}")

def export_charts(simulation_results: dict):
    """Exporta gráficos como imagens"""
    try:
        # Para cada método, criar e exportar gráficos
        for method, results in simulation_results.items():
            if 'final_values' in results and len(results['final_values']) > 0:
                # Criar gráfico de distribuição
                fig = px.histogram(
                    x=results['final_values'],
                    title=f"Distribuição - {method}",
                    labels={'x': 'Valor Final', 'y': 'Frequência'}
                )
                
                # Converter para imagem
                img_bytes = fig.to_image(format="png")
                
                # Botão de download para cada gráfico
                st.download_button(
                    label=f"📸 Baixar Gráfico {method}",
                    data=img_bytes,
                    file_name=f"grafico_{method}_{datetime.now().strftime('%Y%m%d')}.png",
                    mime="image/png",
                    key=f"chart_{method}"
                )
                
    except Exception as e:
        st.error(f"❌ Erro ao exportar gráficos: {e}")

def export_executive_summary(simulation_results: dict, initial_investment: float):
    """Exporta um relatório executivo resumido"""
    try:
        # Métricas consolidadas
        var_values = [abs(results['var']) for results in simulation_results.values() if 'var' in results]
        prob_loss_values = [results['probability_loss'] for results in simulation_results.values() if 'probability_loss' in results]
        
        avg_var = np.mean(var_values) if var_values else 0
        avg_prob_loss = np.mean(prob_loss_values) if prob_loss_values else 0
        
        # Criar PDF executivo
        pdf = FPDF()
        pdf.add_page()
        
        # Cabeçalho
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 15, 'RELATÓRIO EXECUTIVO - ANÁLISE DE RISCO', 0, 1, 'C')
        pdf.ln(10)
        
        # Resumo executivo
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'RESUMO EXECUTIVO', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8, f"""
        Esta análise apresenta os resultados de simulações de Monte Carlo para avaliação de risco do portfólio.
        
        Principais conclusões:
        • Perda esperada (VaR 95%): R$ {avg_var:,.0f} ({avg_var/initial_investment:.1%} do portfólio)
        • Probabilidade de prejuízo: {avg_prob_loss:.1%}
        • Número total de cenários simulados: {sum(r.get('simulations', 0) for r in simulation_results.values()):,}
        
        Recomendações:
        {'• ALERTA: Risco elevado detectado' if avg_prob_loss > 0.2 else '• Risco dentro dos limites aceitáveis'}
        """)
        
        # Salvar e disponibilizar download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf.output(tmp_file.name)
            
            with open(tmp_file.name, 'rb') as f:
                pdf_data = f.read()
        
        st.download_button(
            label="⬇️ Baixar Relatório Executivo",
            data=pdf_data,
            file_name=f"resumo_executivo_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="download_executive"
        )
        
        os.unlink(tmp_file.name)
        
    except Exception as e:
        st.error(f"❌ Erro ao gerar relatório executivo: {e}")

def export_analysis_data(simulation_results: dict, initial_investment: float):
    """Exporta dados completos para análise"""
    try:
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Dados brutos de todas as simulações
            all_final_values = []
            all_methods = []
            
            for method, results in simulation_results.items():
                if 'final_values' in results:
                    n_values = len(results['final_values'])
                    all_final_values.extend(results['final_values'])
                    all_methods.extend([method] * n_values)
            
            analysis_df = pd.DataFrame({
                'Método': all_methods,
                'Valor_Final': all_final_values,
                'Perda': initial_investment - np.array(all_final_values)
            })
            
            analysis_df.to_excel(writer, sheet_name='Dados_Completos', index=False)
            
            # Estatísticas por método
            stats_data = []
            for method, results in simulation_results.items():
                if 'final_values' in results:
                    final_vals = results['final_values']
                    stats_data.append({
                        'Método': method,
                        'Média': np.mean(final_vals),
                        'Mediana': np.median(final_vals),
                        'Desvio_Padrão': np.std(final_vals),
                        'VaR_95%': abs(results.get('var', 0)),
                        'CVaR_95%': abs(results.get('cvar', 0)),
                        'Prob_Prejuízo': results.get('probability_loss', 0)
                    })
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="⬇️ Baixar Dados Completos",
            data=output,
            file_name=f"dados_analise_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_analysis"
        )
        
    except Exception as e:
        st.error(f"❌ Erro ao exportar dados: {e}")

def export_all_charts(simulation_results: dict):
    """Exporta todos os gráficos"""
    try:
        # Criar um ZIP com todos os gráficos
        import zipfile
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for method, results in simulation_results.items():
                if 'final_values' in results:
                    # Gráfico de distribuição
                    fig1 = px.histogram(
                        x=results['final_values'],
                        title=f"Distribuição - {method}",
                        labels={'x': 'Valor Final', 'y': 'Frequência'}
                    )
                    img1_bytes = fig1.to_image(format="png")
                    zip_file.writestr(f"distribuicao_{method}.png", img1_bytes)
                    
                    # Gráfico de densidade
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(
                        x=results['final_values'], 
                        histnorm='probability density',
                        name='Densidade'
                    ))
                    img2_bytes = fig2.to_image(format="png")
                    zip_file.writestr(f"densidade_{method}.png", img2_bytes)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="⬇️ Baixar Todos os Gráficos (ZIP)",
            data=zip_buffer,
            file_name=f"graficos_risco_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            key="download_all_charts"
        )
        
    except Exception as e:
        st.error(f"❌ Erro ao exportar gráficos: {e}")

# =============================================
# FUNÇÕES EXISTENTES DO SISTEMA
# =============================================

def display_all_alerts_with_filters(report):
    """Exibe todos os alertas com sistema de filtros avançado"""
    if not report or 'all_alerts' not in report:
        st.info("📭 Nenhum alerta gerado ainda. Execute a análise multiagente primeiro.")
        return
    
    st.markdown('<div class="section-header">🚨 Sistema de Gestão de Alertas</div>', unsafe_allow_html=True)
    
    all_alerts = report['all_alerts']
    
    # Estatísticas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = len(all_alerts)
        st.metric("📊 Total de Alertas", total_alerts)
    
    with col2:
        critical_high = len([a for a in all_alerts if a['severity'] in ['critical', 'high']])
        st.metric("🔥 Críticos/Altos", critical_high)
    
    with col3:
        agents_involved = len(set(a['agent_id'] for a in all_alerts))
        st.metric("🤖 Agentes", agents_involved)
    
    with col4:
        recent_alerts = len([a for a in all_alerts if a.get('timestamp') and 
                           (datetime.now() - a['timestamp']).days < 1])
        st.metric("🕐 Últimas 24h", recent_alerts)
    
    # Sistema de Filtros
    st.markdown("### 🔍 Filtros Avançados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Filtro por Severidade
        severity_options = ['critical', 'high', 'medium', 'low']
        selected_severities = st.multiselect(
            "Nível de Severidade:",
            options=severity_options,
            default=severity_options,
            key="severity_filter"
        )
    
    with col2:
        # Filtro por Agente
        all_agents = sorted(set(a['agent_id'] for a in all_alerts))
        selected_agents = st.multiselect(
            "Agentes:",
            options=all_agents,
            default=all_agents,
            key="agent_filter"
        )
    
    with col3:
        # Filtro por Data
        date_filter = st.selectbox(
            "Período:",
            ["Todos", "Últimas 24h", "Última semana", "Último mês"],
            key="date_filter"
        )
    
    with col4:
        # Filtro por Conteúdo
        search_text = st.text_input("🔎 Buscar texto:", placeholder="Ex: volatilidade, correlação...")
    
    # Aplicar filtros
    filtered_alerts = all_alerts
    
    # Filtro de severidade
    if selected_severities:
        filtered_alerts = [a for a in filtered_alerts if a['severity'] in selected_severities]
    
    # Filtro de agente
    if selected_agents:
        filtered_alerts = [a for a in filtered_alerts if a['agent_id'] in selected_agents]
    
    # Filtro de data
    if date_filter != "Todos":
        now = datetime.now()
        if date_filter == "Últimas 24h":
            filtered_alerts = [a for a in filtered_alerts if 
                             (now - a.get('timestamp', now)).days < 1]
        elif date_filter == "Última semana":
            filtered_alerts = [a for a in filtered_alerts if 
                             (now - a.get('timestamp', now)).days < 7]
        elif date_filter == "Último mês":
            filtered_alerts = [a for a in filtered_alerts if 
                             (now - a.get('timestamp', now)).days < 30]
    
    # Filtro de texto
    if search_text:
        filtered_alerts = [a for a in filtered_alerts if 
                         search_text.lower() in a['message'].lower() or
                         search_text.lower() in a['agent_id'].lower()]
    
    # Ordenação
    sort_option = st.selectbox(
        "Ordenar por:",
        ["Severidade (crítico primeiro)", "Data (mais recente)", "Agente", "Score"],
        key="sort_filter"
    )
    
    if sort_option == "Severidade (crítico primeiro)":
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        filtered_alerts.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
    elif sort_option == "Data (mais recente)":
        filtered_alerts.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
    elif sort_option == "Agente":
        filtered_alerts.sort(key=lambda x: x['agent_id'])
    elif sort_option == "Score":
        filtered_alerts.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Exibir resultados dos filtros
    st.markdown(f"### 📋 Alertas Filtrados ({len(filtered_alerts)} de {len(all_alerts)})")
    
    if not filtered_alerts:
        st.warning("🚫 Nenhum alerta corresponde aos filtros aplicados.")
        return
    
    # Agrupar por severidade para melhor organização
    severity_groups = {}
    for alert in filtered_alerts:
        severity = alert['severity']
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(alert)
    
    # Exibir por grupos de severidade
    for severity in ['critical', 'high', 'medium', 'low']:
        if severity in severity_groups:
            alerts = severity_groups[severity]
            
            severity_display = {
                'critical': '🔴 CRÍTICO',
                'high': '🟡 ALTO', 
                'medium': '🟠 MÉDIO',
                'low': '🟢 BAIXO'
            }
            
            with st.expander(f"{severity_display[severity]} ({len(alerts)} alertas)", expanded=severity in ['critical', 'high']):
                for alert in alerts:
                    display_single_alert(alert)

    # Exportar alertas
    st.markdown("---")
    _al_buf = BytesIO()
    _al_rows = [{
        "Severidade": a.get("severity", ""),
        "Agente": a.get("agent_id", ""),
        "Mensagem": a.get("message", ""),
        "Score": a.get("score", ""),
        "Timestamp": str(a.get("timestamp", "")),
    } for a in all_alerts]
    pd.DataFrame(_al_rows).to_excel(_al_buf, index=False, engine="xlsxwriter")
    _al_buf.seek(0)
    st.download_button(
        "⬇️ Exportar Alertas (Excel)",
        data=_al_buf,
        file_name=f"alertas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_alerts_excel",
    )

def display_single_alert(alert):
    """Exibe um único alerta de forma organizada"""
    severity_color = {
        'critical': '#ff4757',
        'high': '#ffa502', 
        'medium': '#ff9f43',
        'low': '#00d4aa'
    }
    
    severity_icon = {
        'critical': '🔴',
        'high': '🟡',
        'medium': '🟠', 
        'low': '🟢'
    }
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="padding: 0.5rem; border-left: 4px solid {severity_color[alert['severity']]}; 
                    background: rgba(255,255,255,0.05); border-radius: 4px; margin: 0.2rem 0;">
            <div style="display: flex; justify-content: between; align-items: start;">
                <div style="flex: 1;">
                    <strong>{severity_icon[alert['severity']]} {alert['agent_id']}</strong><br/>
                    <span style="font-size: 0.9rem;">{alert['message']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; border-radius: 5px; 
                    background: {severity_color[alert['severity']]}; 
                    color: white; font-weight: bold; margin-top: 0.5rem;">
            {alert['severity'].upper()}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if alert.get('timestamp'):
            time_str = alert['timestamp'].strftime("%H:%M")
            date_str = alert['timestamp'].strftime("%d/%m")
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; border: 1px solid #555; 
                        border-radius: 5px; margin-top: 0.5rem;">
                <div style="font-size: 0.8rem; color: #ccc;">{date_str}</div>
                <div style="font-size: 0.9rem; font-weight: bold;">{time_str}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detalhes expandíveis
    with st.expander("🔍 Detalhes Técnicos", expanded=False):
        if alert.get('data'):
            st.json(alert['data'])
        else:
            st.info("Sem dados adicionais")
    
    st.markdown("---")

def run_simulation_corrected(algorithm: str, returns_df: pd.DataFrame, initial_investment: float,
                           time_horizon: int, num_simulations: int, weights: Optional[Dict[str, float]] = None):
    """Executa simulação com parâmetros corretos - CORRIGIDA"""
    try:
        from src.agents.agent_simulation import AgentSimulation
        
        simulator = AgentSimulation()
        
        # ✅ CORREÇÃO: Garantir que estamos passando uma Series, não DataFrame
        if weights and len(weights) == len(returns_df.columns):
            weight_sum = sum(weights.values())
            normalized_weights = {k: v/weight_sum for k, v in weights.items()}
            # ✅ Usar .mul e .sum para evitar problemas com index
            weighted_returns = returns_df.mul(pd.Series(normalized_weights), axis=1)
            portfolio_returns = weighted_returns.sum(axis=1)
        else:
            # ✅ CORREÇÃO: Usar mean(axis=1) corretamente
            portfolio_returns = returns_df.mean(axis=1)
        
        # ✅ CORREÇÃO: Garantir que portfolio_returns é uma Series válida
        if portfolio_returns.empty:
            st.error("❌ Retornos do portfólio estão vazios")
            return {}
        
        portfolio_config = {
            'value': initial_investment,
            'confidence_level': 0.95,
            'time_horizon': time_horizon,
            'num_simulations': num_simulations,
            'weights': weights
        }
        
        # ✅ CORREÇÃO: Verificar se o método run_simulation existe
        if hasattr(simulator, 'run_simulation'):
            result = simulator.run_simulation(algorithm, portfolio_returns, portfolio_config)
            return result
        else:
            st.error(f"❌ Método run_simulation não encontrado no AgentSimulation")
            return {}
        
    except Exception as e:
        st.error(f"❌ Erro na simulação {algorithm}: {str(e)}")
        # ✅ CORREÇÃO: Log mais detalhado para debug
        import traceback
        st.error(f"🔍 Detalhes do erro: {traceback.format_exc()}")
        return {}

def run_simulation_simple_fallback(algorithm: str, returns_df: pd.DataFrame, initial_investment: float,
                                 time_horizon: int, num_simulations: int):
    """Fallback robusto para simulações quando o AgentSimulation falha"""
    try:
        # Calcular retorno simples do portfólio
        portfolio_returns = returns_df.mean(axis=1)
        
        if portfolio_returns.empty:
            st.warning("⚠️ Dados de retorno vazios no fallback")
            return {}
        
        # Estatísticas básicas
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        
        # Simulação Monte Carlo básica
        np.random.seed(42)  # Para reproducibilidade
        simulated_returns = np.random.normal(mu, sigma, (time_horizon, num_simulations))
        portfolio_values = initial_investment * np.cumprod(1 + simulated_returns, axis=0)
        
        final_values = portfolio_values[-1, :]
        portfolio_changes = final_values - initial_investment
        
        # Métricas de risco
        var_95 = np.percentile(portfolio_changes, 5)
        cvar_95 = portfolio_changes[portfolio_changes <= var_95].mean()
        prob_loss = np.mean(final_values < initial_investment)
        
        return {
            'var': float(var_95),
            'cvar': float(cvar_95) if not np.isnan(cvar_95) else float(var_95),
            'portfolio_value': float(initial_investment),
            'probability_loss': float(prob_loss),
            'expected_return': float(np.mean(final_values) - initial_investment),
            'final_values': final_values,
            'simulation_paths': portfolio_values,
            'method': f'{algorithm} (Fallback)',
            'simulations': num_simulations,
            'time_horizon': time_horizon
        }
        
    except Exception as e:
        st.error(f"❌ Erro no fallback da simulação {algorithm}: {str(e)}")
        return {}


def get_current_weights(selected_assets):
    """Calcula pesos atuais baseados na seleção - CORRIGIDA"""
    if not selected_assets:
        return {}
    equal_weight = 1.0 / len(selected_assets)
    return {asset: equal_weight for asset in selected_assets}


# Função para carregar imagem como base64
def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return None

# CSS personalizado - Estilo Wall Street Journal com elementos tech
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    * {
        font-family: 'EB Garamond', serif;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: #f0f2f6 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #1e2130 !important;
        min-width: 360px !important;
        width: 360px !important;
    }

    /* Permite redimensionar a sidebar normalmente */
    [data-testid="stSidebarResizeHandle"] {
        visibility: visible !important;
    }

    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    .main-header {
        font-size: 2.8rem;
        color: #00d4aa;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.5px;
        font-family: 'EB Garamond', serif;
        text-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #00d4aa;
        border-bottom: 2px solid #00d4aa;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
        font-family: 'EB Garamond', serif;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3746 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #00d4aa;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 212, 170, 0.2);
    }
    
    .metric-card {
        background-color: #1e2130;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #00d4aa;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .simulation-card {
        background-color: #1e2130;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        border: 1px solid #333;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border-left: 4px solid #ff4757;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ffa502 0%, #e67e22 100%);
        border-left: 4px solid #ff9f43;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border-left: 4px solid #fdcb6e;
        color: #2d3436;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        border-left: 4px solid #00d4aa;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 15px;
        padding-bottom: 15px;
        font-weight: 500;
        font-family: 'EB Garamond', serif;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #26293f;
        border-bottom: 3px solid #00d4aa;
    }
    
    /* Tech elements */
    .tech-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #00d4aa;
        box-shadow: 0 0 10px #00d4aa;
    }
    
    .status-inactive {
        background-color: #ff4757;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1e2130;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 1.5rem;
    }
    
    .logo-img {
        max-width: 200px;
        margin: 0 auto;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'EB Garamond', serif !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        font-family: 'EB Garamond', serif !important;
    }
    
    /* Code-like elements */
    .code-block {
        background: #1a1b26;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #a9b1d6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Logo do BACEN
try:
    # Caminho absoluto baseado no local do arquivo atual (app.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "bacon_logo.png")
    
    logo_base64 = get_image_as_base64(logo_path)
    if logo_base64:
        st.sidebar.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img"
                 style="max-width: 160px; display: block; margin: 0 auto 0.5rem auto;">
            <div style="color: #00d4aa; font-size: 1.3rem; font-weight: 600; margin-top: 0.5rem;">
                BANCO CENTRAL DO BRASIL
            </div>
            <div style="color: #888; font-size: 1rem; margin-top: 0.5rem;">Sistema Multiagente Quantitativo</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="logo-container">
            <div style="color: #00d4aa; font-size: 1.5rem; font-weight: 700;">BANCO CENTRAL DO BRASIL</div>
            <div style="color: #888; font-size: 1rem; margin-top: 0.5rem;">Sistema Multiagente Quantitativo</div>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown(f"""
    <div class="logo-container">
        <div style="color: #00d4aa; font-size: 1.5rem; font-weight: 700;">BANCO CENTRAL DO BRASIL</div>
        <div style="color: #888; font-size: 1rem; margin-top: 0.5rem;">Sistema Multiagente Quantitativo</div>
        <div style="color: red; font-size: 0.8rem;">(Logo não encontrado: {e})</div>
    </div>
    """, unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🤖 Quantum Risk Analytics - Multiagente</h1>', unsafe_allow_html=True)
st.markdown("---")

# =============================================
# INICIALIZAÇÃO DO SISTEMA MULTIAGENTE
# =============================================

@st.cache_resource
def initialize_multiagent_system():
    """Inicializa o sistema multiagente uma única vez"""
    try:
        from src.agents.dask_orchestrator import DaskMultiAgentOrchestrator
        
        orchestrator = DaskMultiAgentOrchestrator(
            use_dask=True,      # ✅ Dask ativado
            use_tensorflow=True # ✅ TensorFlow ativado
        )
        
        st.success("✅ Sistema Multiagente inicializado com sucesso!")
        return orchestrator
    except Exception as e:
        st.error(f"❌ Erro na inicialização do sistema multiagente: {e}")
        # Fallback: sistema sem TensorFlow
        try:
            from src.agents.dask_orchestrator import DaskMultiAgentOrchestrator
            orchestrator = DaskMultiAgentOrchestrator(
                use_dask=True,
                use_tensorflow=False
            )
            st.warning("⚠️ Sistema inicializado sem TensorFlow (modo fallback)")
            return orchestrator
        except Exception as e2:
            st.error(f"❌ Falha total na inicialização: {e2}")
            return None

# Inicializar sistema multiagente
if 'orchestrator' not in st.session_state:
    with st.spinner("🚀 Inicializando Sistema Multiagente..."):
        st.session_state.orchestrator = initialize_multiagent_system()

# Carregar configuração de alertas
from src.alerts.alert_config import load_config as _load_alert_cfg, save_config as _save_alert_cfg, \
    evaluate_alerts as _evaluate_alerts, alert_summary as _alert_summary

if 'alert_config' not in st.session_state:
    st.session_state.alert_config = _load_alert_cfg()
if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = []

# =============================================
# FUNÇÕES AUXILIARES
# =============================================

# ------------------------------------------------------------------
# Presets de ativos por categoria
# ------------------------------------------------------------------
ASSET_PRESETS = {
    "🇧🇷 B3":    ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA"],
    "🇺🇸 NYSE":  ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "₿ Crypto":  ["BTC-USD", "ETH-USD", "SOL-USD"],
    "🏢 FIIs":   ["KNRI11.SA", "HGLG11.SA", "MXRF11.SA", "XPML11.SA", "VISC11.SA"],
    "📊 ETFs":   ["SPY", "QQQ", "BOVA11.SA", "IVVB11.SA"],
    "💱 Forex":  ["USDBRL=X", "EURBRL=X", "JPYBRL=X"],
}

PERIOD_OPTIONS = ["1M", "3M", "6M", "1A", "2A", "5A", "Personalizado"]
PERIOD_MAP     = {"1M": "1mo", "3M": "3mo", "6M": "6mo",
                  "1A": "1y",  "2A": "2y",  "5A": "5y", "Personalizado": "custom"}

BACEN_SERIES = {
    "SELIC (%)":    11,
    "IPCA (%)":     433,
    "Câmbio BRL":   1,
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_portfolio_data(tickers_tuple: tuple, period: str,
                         custom_start: str = None, custom_end: str = None):
    """Baixa preços via yfinance com fallback automático de sufixo .SA para B3.
    Retorna (returns, prices_norm_100)."""
    if not tickers_tuple:
        return pd.DataFrame(), pd.DataFrame()

    import yfinance as yf
    import re

    def _download(tickers, **kw):
        raw = yf.download(tickers, **kw)
        if raw.empty:
            return pd.DataFrame()
        prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) \
                 else raw[["Close"]].rename(columns={"Close": tickers[0]})
        return prices

    kwargs = dict(auto_adjust=True, progress=False)
    if period == "custom" and custom_start and custom_end:
        kwargs.update(start=custom_start, end=custom_end)
    else:
        kwargs["period"] = period

    # Separar tickers que parecem B3 sem sufixo (ex: "PETR4" → tenta "PETR4.SA")
    _b3_pattern = re.compile(r"^[A-Z]{4}\d{1,2}$")
    primary   = list(tickers_tuple)
    sa_remap  = {}   # original → com .SA

    prices_all = pd.DataFrame()

    # 1ª tentativa: todos os tickers como vieram
    try:
        prices_all = _download(primary, **kwargs)
    except Exception:
        pass

    # Identificar os que não vieram e tentar com .SA
    missing = [t for t in primary
               if prices_all.empty or t not in prices_all.columns
               if _b3_pattern.match(t)]

    if missing:
        sa_tickers = [t + ".SA" for t in missing]
        sa_remap   = dict(zip(sa_tickers, missing))
        try:
            prices_sa = _download(sa_tickers, **kwargs)
            if not prices_sa.empty:
                prices_sa = prices_sa.rename(columns=sa_remap)
                prices_all = prices_sa if prices_all.empty \
                             else pd.concat([prices_all, prices_sa], axis=1)
        except Exception:
            pass

    if prices_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices_all = prices_all.dropna(axis=1, thresh=max(1, int(len(prices_all) * 0.5)))
    prices_all = prices_all.ffill().bfill().dropna()

    if prices_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices_norm = prices_all.div(prices_all.iloc[0]) * 100
    returns     = prices_all.pct_change().dropna()
    return returns, prices_norm


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmarks_br(selected_names: tuple, days: int = 365):
    """Busca benchmarks de renda fixa BR via BACEN. Retorna DataFrame normalizado a 100."""
    if not selected_names:
        return pd.DataFrame()
    try:
        from src.etl.benchmarks_br import BENCHMARK_FUNCS
        series = []
        for name in selected_names:
            if name in BENCHMARK_FUNCS:
                s = BENCHMARK_FUNCS[name](days)
                if not s.empty:
                    series.append(s)
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).ffill().dropna()
    except Exception:
        return pd.DataFrame()


INTRADAY_PERIODS   = ["Hoje", "2 dias", "5 dias"]
INTRADAY_INTERVALS = ["1m", "5m", "15m", "30m", "1h"]
_INTRADAY_PERIOD_MAP = {"Hoje": "1d", "2 dias": "2d", "5 dias": "5d"}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_data(tickers_tuple: tuple, period: str, interval: str):
    """Baixa dados intraday via yfinance. TTL=60s para aproximar tempo real.
    Retorna (returns, prices) com preços brutos (sem normalização — candlestick usa preço real)."""
    if not tickers_tuple:
        return pd.DataFrame(), pd.DataFrame()
    import yfinance as yf, re
    _b3 = re.compile(r"^[A-Z]{4}\d{1,2}$")
    tickers = list(tickers_tuple)
    kw = dict(period=period, interval=interval, auto_adjust=True, progress=False)

    def _dl(tkrs):
        raw = yf.download(tkrs, **kw)
        if raw.empty:
            return pd.DataFrame()
        return raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) \
               else raw[["Close"]].rename(columns={"Close": tkrs[0]})

    prices = _dl(tickers)

    # Fallback .SA para tickers B3 sem sufixo
    missing = [t for t in tickers
               if prices.empty or t not in prices.columns
               if _b3.match(t)]
    if missing:
        sa_map   = {t + ".SA": t for t in missing}
        prices_sa = _dl(list(sa_map.keys()))
        if not prices_sa.empty:
            prices_sa = prices_sa.rename(columns=sa_map)
            prices = prices_sa if prices.empty \
                     else pd.concat([prices, prices_sa], axis=1)

    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices = prices.dropna(axis=1, thresh=max(1, int(len(prices) * 0.3))).ffill().bfill()
    prices = prices.dropna()
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    returns = prices.pct_change().dropna()
    return returns, prices


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bacen_macro(series_codes: tuple):
    """Baixa séries do BACEN via API SGS. Retorna DataFrame com as séries."""
    import requests
    from datetime import date

    results = {}
    for name, code in series_codes:
        try:
            url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
                   f"?formato=json&dataInicial=01/01/2020&dataFinal={date.today().strftime('%d/%m/%Y')}")
            resp = requests.get(url, timeout=10)
            if resp.ok:
                df = pd.DataFrame(resp.json())
                df["data"] = pd.to_datetime(df["data"], dayfirst=True)
                df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
                df = df.set_index("data")["valor"].dropna()
                results[name] = df
        except Exception:
            pass

    if not results:
        return pd.DataFrame()

    macro = pd.DataFrame(results)
    macro.index = pd.to_datetime(macro.index)
    return macro.sort_index()

def get_current_weights(selected_assets):
    """Calcula pesos atuais baseados na seleção"""
    if not selected_assets:
        return {}
    equal_weight = 1.0 / len(selected_assets)
    return {asset: equal_weight for asset in selected_assets}

def display_agent_status():
    """Exibe status dos agentes"""
    st.markdown('<div class="section-header">🤖 Status dos Agentes</div>', unsafe_allow_html=True)
    
    agents = {
        "AgentMarket": "📊 Análise de Mercado (Volatilidade/Correlação)",
        "AgentClustering": "🎯 Clustering de Ativos (K-Means + PCA)",
        "AgentML": "🧠 Machine Learning (Isolation Forest + Random Forest)", 
        "AgentLSTM": "📈 Deep Learning (LSTM - Previsão Temporal)",
        "AgentAutoencoder": "🔍 Deep Learning (Autoencoder - Anomalias)",
        "AgentAlert": "⚠️ Sistema de Alertas Inteligentes"
    }
    
    cols = st.columns(3)
    for idx, (agent, description) in enumerate(agents.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="agent-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-indicator status-active"></span>
                    <h4 style="margin: 0; color: #00d4aa;">{agent}</h4>
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: #ccc;">{description}</p>
                <div style="margin-top: 0.5rem;">
                    <span class="tech-badge">ONLINE</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_alert_report(report):
    """Exibe relatório de alertas de forma avançada"""
    if not report:
        return
    
    st.markdown('<div class="section-header">🚨 Relatório de Alertas Consolidado</div>', unsafe_allow_html=True)
    
    # Métricas rápidas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_alerts = report.get('total_alerts', 0)
        st.metric("📊 Total de Alertas", total_alerts)
    
    with col2:
        critical = report.get('severity_breakdown', {}).get('critical', 0)
        high = report.get('severity_breakdown', {}).get('high', 0)
        st.metric("🔥 Críticos/Altos", critical + high, delta="Atenção" if critical > 0 else None)
    
    with col3:
        st.metric("🎯 Ativos com Alertas", len(report.get('asset_alerts', {})))
    
    with col4:
        st.metric("🤖 Agentes Ativos", len([k for k in report.keys() if 'alerts' in k.lower()]))
    
    with col5:
        orchestrator_type = report.get('orchestrator', 'Sequencial')
        st.metric("⚡ Orquestrador", orchestrator_type)
    
    # Alertas detalhados
    st.subheader("📋 Alertas Prioritários")
    
    critical_alerts = report.get('critical_alerts', [])
    if not critical_alerts:
        st.success("🎉 Nenhum alerta crítico detectado - Situação estável!")
        return
    
    # Agrupar alertas por severidade
    severity_groups = {}
    for alert in critical_alerts:
        severity = alert.get('severity', 'low')
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(alert)
    
    # Exibir por ordem de severidade
    for severity in ['critical', 'high', 'medium', 'low']:
        if severity in severity_groups:
            alerts = severity_groups[severity][:8]  # Limitar a 8 por grupo
            
            severity_display = {
                'critical': '🔴 CRÍTICO',
                'high': '🟡 ALTO', 
                'medium': '🟠 MÉDIO',
                'low': '🟢 BAIXO'
            }
            
            st.markdown(f"**{severity_display.get(severity, severity.upper())}** ({len(severity_groups[severity])} alertas)")
            
            for alert in alerts:
                alert_class = f"alert-{severity}"
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="agent-card {alert_class}">
                            <div style="display: flex; justify-content: between; align-items: start;">
                                <div style="flex: 1;">
                                    <strong>🤖 {alert.get('agent_id', 'Unknown')}</strong><br/>
                                    <span>{alert.get('message', 'No message')}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detalhes expandíveis
                        with st.expander("🔍 Detalhes Técnicos", expanded=False):
                            if alert.get('data'):
                                st.json(alert['data'])
                            else:
                                st.info("Sem dados adicionais")
                    
                    with col2:
                        severity_color = {
                            'critical': '#ff4757',
                            'high': '#ffa502', 
                            'medium': '#ff9f43',
                            'low': '#00d4aa'
                        }
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem; border-radius: 5px; 
                                    background: {severity_color.get(severity, '#666')}; 
                                    color: white; font-weight: bold;">
                            {severity.upper()}
                        </div>
                        """, unsafe_allow_html=True)

def display_agent_analytics(alerts):
    """Exibe analytics avançados dos agentes"""
    if not alerts:
        return
    
    st.markdown('<div class="section-header">📈 Analytics dos Agentes</div>', unsafe_allow_html=True)
    
    # Análise por agente
    agent_counts = {}
    severity_counts = {}
    asset_mentions = {}
    
    for alert in alerts:
        agent = alert.get('agent_id', 'Unknown')
        severity = alert.get('severity', 'low')
        
        # Contar por agente
        agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Contar por severidade
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Contar menções de ativos
        if 'data' in alert and 'asset' in alert['data']:
            asset = alert['data']['asset']
            asset_mentions[asset] = asset_mentions.get(asset, 0) + 1
        elif 'data' in alert and 'assets' in alert['data']:
            for asset in alert['data']['assets']:
                asset_mentions[asset] = asset_mentions.get(asset, 0) + 1
    
    # Visualizações
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de alertas por agente
        if agent_counts:
            fig_agents = px.pie(
                values=list(agent_counts.values()),
                names=list(agent_counts.keys()),
                title="📊 Distribuição de Alertas por Agente",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_agents.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig_agents, use_container_width=True)
    
    with col2:
        # Gráfico de severidade
        if severity_counts:
            fig_severity = px.bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                title="🚨 Alertas por Nível de Severidade",
                color=list(severity_counts.keys()),
                color_discrete_map={
                    'critical': '#ff4757',
                    'high': '#ffa502',
                    'medium': '#ff9f43', 
                    'low': '#00d4aa'
                }
            )
            fig_severity.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_severity, use_container_width=True)

# =============================================
# SIDEBAR — BUSCA DE ATIVOS & CONFIGURAÇÕES
# =============================================

st.sidebar.markdown('<div class="section-header">🔍 Busca de Ativos</div>', unsafe_allow_html=True)

# --- Toggle de Modo ---
app_mode = st.sidebar.radio(
    "Modo de operação:",
    ["📊 Análise de Risco", "⚡ Day Trader"],
    horizontal=True,
    key="app_mode",
)
if app_mode == "⚡ Day Trader":
    st.sidebar.info("⚡ **Day Trader**: dados intraday com atualização automática a cada 60s.", icon="⚡")
st.sidebar.markdown("---")

# --- Estado inicial dos tickers ---
if "tickers_text" not in st.session_state:
    st.session_state.tickers_text = "PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA, WEGE3.SA"
if "selected_period" not in st.session_state:
    st.session_state.selected_period = "3M"
if "custom_start" not in st.session_state:
    st.session_state.custom_start = None
if "custom_end" not in st.session_state:
    st.session_state.custom_end = None
if "portfolio_data" not in st.session_state:
    st.session_state.portfolio_data = (pd.DataFrame(), pd.DataFrame())
if "data_timestamp" not in st.session_state:
    st.session_state.data_timestamp = None
if "invalid_tickers" not in st.session_state:
    st.session_state.invalid_tickers = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "📊 Análise de Risco"
if "intraday_period" not in st.session_state:
    st.session_state.intraday_period = "Hoje"
if "intraday_interval" not in st.session_state:
    st.session_state.intraday_interval = "5m"
if "intraday_data" not in st.session_state:
    st.session_state.intraday_data = (pd.DataFrame(), pd.DataFrame())

# --- Presets rápidos ---
st.sidebar.markdown("**Pré-seleções rápidas:**")
_preset_items = list(ASSET_PRESETS.items())
# 3 colunas por linha (6 presets → 2 linhas de 3)
for _row in [_preset_items[:3], _preset_items[3:]]:
    _cols = st.sidebar.columns(3)
    for _col, (_label, _tickers_preset) in zip(_cols, _row):
        with _col:
            if st.button(_label, key=f"preset_{_label}", use_container_width=True):
                st.session_state.tickers_text = ", ".join(_tickers_preset)

# --- Busca por nome ---
@st.cache_data(ttl=120, show_spinner=False)
def search_ticker(query: str):
    """Busca tickers no Yahoo Finance pelo nome ou símbolo."""
    try:
        import yfinance as yf
        results = yf.Search(query, max_results=8)
        return [
            {
                "symbol": q.get("symbol", ""),
                "name":   q.get("shortname") or q.get("longname") or "",
                "exchange": q.get("exchange", ""),
            }
            for q in results.quotes
            if q.get("symbol")
        ]
    except Exception:
        return []

search_query = st.sidebar.text_input(
    "🔎 Buscar ativo por nome:",
    placeholder="ex: embraer, bitcoin, apple...",
    key="asset_search_input"
)

if search_query.strip():
    with st.sidebar:
        with st.spinner("Buscando..."):
            hits = search_ticker(search_query.strip())

    if hits:
        options = [f"{h['symbol']}  —  {h['name']}  [{h['exchange']}]" for h in hits]
        chosen = st.sidebar.selectbox(
            "Resultado — selecione e clique em Adicionar:",
            options=[""] + options,
            key="search_result_select"
        )
        if chosen and st.sidebar.button("➕ Adicionar ao portfólio", key="add_search_result"):
            ticker_to_add = chosen.split("  —  ")[0].strip()
            current = st.session_state.tickers_text.strip().rstrip(",")
            st.session_state.tickers_text = (
                f"{current}, {ticker_to_add}" if current else ticker_to_add
            )
    else:
        st.sidebar.caption("Nenhum resultado encontrado.")

st.sidebar.markdown("---")

# --- Input de tickers ---
tickers_input = st.sidebar.text_area(
    "Tickers no portfólio (separados por vírgula):",
    value=st.session_state.tickers_text,
    height=100,
    placeholder="PETR4.SA, VALE3.SA, AAPL, BTC-USD",
    key="tickers_text_area",
    help="B3: PETR4.SA  |  NYSE: AAPL  |  Crypto: BTC-USD  |  Forex: USDBRL=X"
)
st.session_state.tickers_text = tickers_input

if app_mode == "📊 Análise de Risco":
    # --- Seletor de período ---
    st.sidebar.markdown("**📅 Período de análise:**")
    period_label = st.sidebar.radio(
        "Período",
        options=PERIOD_OPTIONS,
        index=PERIOD_OPTIONS.index(st.session_state.selected_period),
        horizontal=True,
        label_visibility="collapsed",
        key="period_radio"
    )
    st.session_state.selected_period = period_label

    custom_start_str, custom_end_str = None, None
    if period_label == "Personalizado":
        c1, c2 = st.sidebar.columns(2)
        with c1:
            cs = st.date_input("Início", key="custom_start_input")
        with c2:
            ce = st.date_input("Fim", key="custom_end_input")
        custom_start_str = str(cs)
        custom_end_str   = str(ce)

    # --- Botão Atualizar ---
    col_btn, col_info = st.sidebar.columns([1, 1])
    with col_btn:
        fetch_btn = st.button("🔄 Atualizar Dados", type="primary",
                              use_container_width=True, key="fetch_data_btn")
    with col_info:
        if st.session_state.data_timestamp:
            st.caption(f"⏱ {st.session_state.data_timestamp}")

    # --- Parâmetros de risco ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Parâmetros de Risco**")
    risk_free_rate   = st.sidebar.slider("Taxa Livre de Risco (% a.a):", 0.0, 20.0, 11.75, 0.1) / 100
    confidence_level = st.sidebar.slider("Nível de Confiança VaR:", 0.90, 0.99, 0.95, 0.01)

    # --- Benchmarks de renda fixa BR ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏦 Benchmarks Renda Fixa BR**")
    st.sidebar.caption("Sobrepostos no gráfico de desempenho — dados via BACEN")

    from src.etl.benchmarks_br import BENCHMARK_FUNCS as _BF
    benchmark_selections = st.sidebar.multiselect(
        "Comparar portfólio com:",
        options=list(_BF.keys()),
        default=[],
        key="benchmark_select"
    )

    # --- Overlay macro BACEN (séries brutas) ---
    st.sidebar.markdown("**📊 Overlay Macro BACEN**")
    bacen_selections = st.sidebar.multiselect(
        "Séries brutas (eixo secundário):",
        options=list(BACEN_SERIES.keys()),
        default=[],
        key="bacen_overlay_select"
    )

    # --- Configurações avançadas ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Configurações Avançadas**")
    enable_clustering  = st.sidebar.checkbox("Ativar Clustering de Ativos", value=True)
    enable_ml_detection= st.sidebar.checkbox("Ativar Detecção ML", value=True)

else:  # ⚡ Day Trader
    # --- Período intraday ---
    st.sidebar.markdown("**📅 Período:**")
    intraday_period_label = st.sidebar.radio(
        "Período intraday",
        options=INTRADAY_PERIODS,
        index=INTRADAY_PERIODS.index(st.session_state.intraday_period),
        horizontal=True,
        label_visibility="collapsed",
        key="intraday_period_radio"
    )
    st.session_state.intraday_period = intraday_period_label

    # --- Intervalo intraday ---
    st.sidebar.markdown("**⏱ Intervalo:**")
    intraday_interval_sel = st.sidebar.radio(
        "Intervalo",
        options=INTRADAY_INTERVALS,
        index=INTRADAY_INTERVALS.index(st.session_state.intraday_interval),
        horizontal=True,
        label_visibility="collapsed",
        key="intraday_interval_radio"
    )
    st.session_state.intraday_interval = intraday_interval_sel

    # --- Botão Atualizar ---
    col_btn, col_info = st.sidebar.columns([1, 1])
    with col_btn:
        fetch_btn = st.button("🔄 Atualizar", type="primary",
                              use_container_width=True, key="fetch_data_btn")
    with col_info:
        if st.session_state.data_timestamp:
            st.caption(f"⏱ {st.session_state.data_timestamp}")

    # Defaults para variáveis usadas downstream
    period_label        = "1A"
    custom_start_str    = None
    custom_end_str      = None
    risk_free_rate      = 0.1175
    confidence_level    = 0.95
    benchmark_selections = []
    bacen_selections     = []
    enable_clustering    = False
    enable_ml_detection  = False

use_dask       = False  # Dask não instalado — modo sequencial
use_tensorflow = False  # placeholder

# --- Tech Stack badges ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🛠️ Tech Stack:**")
st.sidebar.markdown(
    '<span class="tech-badge">yfinance</span>'
    '<span class="tech-badge">Scikit-learn</span>'
    '<span class="tech-badge">Plotly</span>'
    '<span class="tech-badge">6 Agents</span>',
    unsafe_allow_html=True
)

# =============================================
# FETCH + FILTRAGEM DE DADOS
# =============================================

def _parse_tickers(raw: str) -> tuple:
    """Normaliza texto de tickers em uma tupla imutável (para cache key)."""
    import re
    tokens = re.split(r"[,\n\r]+", raw)
    cleaned = [t.strip().upper() for t in tokens if t.strip()]
    return tuple(dict.fromkeys(cleaned))  # Remove duplicatas, preserva ordem

# Auto-refresh de 60s ativo somente no modo Day Trader
if app_mode == "⚡ Day Trader":
    _refresh_count = st_autorefresh(interval=60_000, key="dt_autorefresh")

tickers_tuple = _parse_tickers(tickers_input)

if app_mode == "⚡ Day Trader":
    # Day Trader — sempre chama fetch_intraday_data; o cache TTL=60s evita downloads desnecessários.
    # O auto-refresh dispara a cada 60s coincidindo com a expiração do cache → dados frescos.
    _intraday_period   = _INTRADAY_PERIOD_MAP[st.session_state.intraday_period]
    _intraday_interval = st.session_state.intraday_interval

    if tickers_tuple:
        ret_i, prc_i = fetch_intraday_data(tickers_tuple, _intraday_period, _intraday_interval)
        if not ret_i.empty:
            st.session_state.intraday_data  = (ret_i, prc_i)
            st.session_state.data_timestamp = datetime.now().strftime("%H:%M:%S")
            fetched = set(ret_i.columns.tolist())
            st.session_state.invalid_tickers = [t for t in tickers_tuple if t not in fetched]
        elif fetch_btn:
            st.error("❌ Sem dados intraday. Verifique os tickers.")
    elif fetch_btn:
        st.warning("⚠️ Insira ao menos um ticker.")

    returns, prices = st.session_state.intraday_data

else:
    # Análise de Risco — fetch EOD (comportamento original)
    should_fetch = fetch_btn or st.session_state.portfolio_data[0].empty

    if should_fetch:
        if tickers_tuple:
            yf_period = PERIOD_MAP[period_label]
            with st.spinner(f"Baixando dados de {len(tickers_tuple)} ativo(s)..."):
                ret, prc = fetch_portfolio_data(
                    tickers_tuple, yf_period, custom_start_str, custom_end_str
                )
            if not ret.empty:
                st.session_state.portfolio_data  = (ret, prc)
                st.session_state.data_timestamp  = datetime.now().strftime("%H:%M:%S")
                fetched = set(ret.columns.tolist())
                st.session_state.invalid_tickers = [t for t in tickers_tuple if t not in fetched]
            else:
                st.error("❌ Nenhum dado retornado. Verifique os tickers e tente novamente.")
        else:
            st.warning("⚠️ Insira ao menos um ticker válido.")

    returns, prices = st.session_state.portfolio_data

# Alertas de tickers inválidos
if st.session_state.invalid_tickers:
    _inv = ", ".join(f"`{t}`" for t in st.session_state.invalid_tickers)
    st.warning(
        f"⚠️ Ticker(s) não encontrado(s): {_inv}\n\n"
        "Possíveis causas: ticker inexistente, erro de digitação ou ativo fora de cobertura do Yahoo Finance. "
        "Ativos B3 precisam do sufixo `.SA` (ex: `EMBR3.SA` para Embraer, `PETR4.SA` para Petrobras). "
        "Use a **busca por nome** na barra lateral para encontrar o ticker correto."
    )

if returns.empty:
    st.info("""
    **👈 Comece adicionando ativos na barra lateral.**

    Exemplos:
    - Ações B3: `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`
    - NYSE/NASDAQ: `AAPL`, `NVDA`, `MSFT`
    - Crypto: `BTC-USD`, `ETH-USD`
    - Forex: `USDBRL=X`
    - ETFs: `BOVA11.SA`, `SPY`

    Use os botões de pré-seleção na sidebar para começar rapidamente!
    """)
    st.stop()

# --- Filtro de ativos carregados (multiselect) ---
all_loaded = returns.columns.tolist()
selected_assets = st.sidebar.multiselect(
    "**📌 Ativos no portfólio:**",
    options=all_loaded,
    default=all_loaded,
    key="asset_multiselect"
)
if not selected_assets:
    selected_assets = all_loaded

returns_filtered  = returns[selected_assets]
prices_filtered   = prices[selected_assets]

# Retorno ponderado do portfólio (pesos iguais)
portfolio_returns = returns_filtered.mean(axis=1)

# --- Avaliar alertas contra configuração do usuário ---
st.session_state.active_alerts = _evaluate_alerts(returns_filtered, st.session_state.alert_config)

# Badge de alertas na sidebar
_summary = _alert_summary(st.session_state.active_alerts)
_total   = sum(_summary.values())
if _total > 0:
    _color = "#e74c3c" if (_summary["critical"] + _summary["high"]) > 0 else "#f39c12"
    st.sidebar.markdown(
        f'<div style="background:{_color};color:white;padding:6px 10px;border-radius:6px;'
        f'text-align:center;font-weight:bold;margin-bottom:8px;">'
        f'🔔 {_total} alerta{"s" if _total > 1 else ""} ativo{"s" if _total > 1 else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# --- Benchmarks e Macro (somente no modo Análise de Risco) ---
_period_days = {"1M": 31, "3M": 92, "6M": 183, "1A": 365,
                "2A": 730, "5A": 1825, "Personalizado": 365}
_bench_days = _period_days.get(st.session_state.get("selected_period", "1A"), 365)

benchmark_df = pd.DataFrame()
if benchmark_selections and app_mode == "📊 Análise de Risco":
    with st.spinner("Carregando benchmarks de renda fixa..."):
        benchmark_df = fetch_benchmarks_br(
            tuple(benchmark_selections), days=_bench_days
        )

macro_df = pd.DataFrame()
if bacen_selections and app_mode == "📊 Análise de Risco":
    selected_codes = tuple((k, BACEN_SERIES[k]) for k in bacen_selections)
    with st.spinner("Carregando dados do BACEN..."):
        macro_df = fetch_bacen_macro(selected_codes)

# =============================================
# LAYOUT PRINCIPAL COM ABAS
# =============================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard", 
    "🤖 Sistema Multiagente", 
    "📊 Análise de Risco",
    "🎯 Simulações", 
    "📈 Analytics",
    "⚙️ Configurações"
])

with tab1:
    if app_mode == "⚡ Day Trader":
        # ── DAY TRADER VIEW ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">⚡ Day Trader — Dados Intraday</div>',
                    unsafe_allow_html=True)

        # Banner de aviso sobre delay B3
        st.warning(
            "⏱ **Atenção:** Dados de ações B3 possuem atraso de ~15 minutos (política da B3/yfinance). "
            "Ativos internacionais (NYSE, Crypto) costumam ter delay menor. "
            "**Não use para execução de ordens — apenas análise.**",
            icon="⚠️"
        )

        # Timestamp e intervalo ativos
        _ts  = st.session_state.data_timestamp or "—"
        _per = st.session_state.intraday_period
        _ivl = st.session_state.intraday_interval
        st.caption(f"Última atualização: **{_ts}** · Período: **{_per}** · Intervalo: **{_ivl}** · Auto-refresh: 60s")

        # Cards de preço por ativo
        if not prices.empty:
            _assets_dt = prices_filtered.columns.tolist()
            _n = len(_assets_dt)
            _cols_cards = st.columns(min(_n, 4))
            for _i, _asset in enumerate(_assets_dt):
                _col_idx = _i % min(_n, 4)
                with _cols_cards[_col_idx]:
                    _s = prices_filtered[_asset].dropna()
                    if len(_s) >= 2:
                        _last  = float(_s.iloc[-1])
                        _open  = float(_s.iloc[0])
                        _high  = float(_s.max())
                        _low   = float(_s.min())
                        _chg   = (_last - _open) / _open if _open != 0 else 0
                        st.metric(
                            label=_asset,
                            value=f"{_last:.2f}",
                            delta=f"{_chg:+.2%}  (H:{_high:.2f} L:{_low:.2f})"
                        )

            # Candlestick por ativo (expander por ativo)
            import yfinance as yf
            _yfkw = dict(
                period=_INTRADAY_PERIOD_MAP[_per],
                interval=_ivl,
                auto_adjust=True,
                progress=False
            )
            for _asset in _assets_dt:
                with st.expander(f"📊 Candlestick — {_asset}", expanded=(_asset == _assets_dt[0])):
                    try:
                        _ohlc = yf.download([_asset], **_yfkw)
                        if _ohlc.empty:
                            _ticker_sa = _asset if _asset.endswith(".SA") else _asset + ".SA"
                            _ohlc = yf.download([_ticker_sa], **_yfkw)
                        if not _ohlc.empty:
                            if isinstance(_ohlc.columns, pd.MultiIndex):
                                _ohlc.columns = _ohlc.columns.get_level_values(0)
                            _fig_cs = go.Figure(go.Candlestick(
                                x=_ohlc.index,
                                open=_ohlc["Open"],
                                high=_ohlc["High"],
                                low=_ohlc["Low"],
                                close=_ohlc["Close"],
                                name=_asset,
                                increasing_line_color="#2ecc71",
                                decreasing_line_color="#e74c3c",
                            ))
                            _fig_cs.update_layout(
                                height=350,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f0f2f6'),
                                xaxis=dict(gridcolor='#333', rangeslider=dict(visible=False)),
                                yaxis=dict(gridcolor='#333'),
                                margin=dict(l=10, r=10, t=30, b=10),
                            )
                            st.plotly_chart(_fig_cs, use_container_width=True)
                        else:
                            st.caption(f"Sem dados OHLC para {_asset}.")
                    except Exception as _e:
                        st.caption(f"Erro ao carregar candlestick: {_e}")
        else:
            st.info("👈 Clique **Atualizar** na sidebar para carregar dados intraday.")

    else:
        # ── ANÁLISE DE RISCO VIEW (padrão) ──────────────────────────────────
        st.markdown('<div class="section-header">📈 Visão Geral do Portfólio</div>',
                    unsafe_allow_html=True)

        # Métricas rápidas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = (1 + portfolio_returns).prod() - 1
            st.metric("📈 Retorno Total", f"{total_return:.2%}")

        with col2:
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            st.metric("📉 Volatilidade Anual", f"{annual_vol:.2%}")

        with col3:
            sharpe = (portfolio_returns.mean() * 252 - risk_free_rate) / (annual_vol + 1e-8)
            st.metric("🎯 Sharpe Ratio", f"{sharpe:.2f}")

        with col4:
            var_95 = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            st.metric(f"⚠️ VaR {confidence_level:.0%}", f"{var_95:.2%}")

        # ── Painel de alertas ativos ─────────────────────────────────────────
        _alerts = st.session_state.active_alerts
        if _alerts:
            _sev_colors = {"critical": "#c0392b", "high": "#e74c3c",
                           "medium": "#f39c12", "low": "#27ae60"}
            _sev_icons  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            _summ = _alert_summary(_alerts)
            _crit_high = _summ["critical"] + _summ["high"]
            _label = (f"🚨 {len(_alerts)} alerta{'s' if len(_alerts) > 1 else ''} ativo{'s' if len(_alerts) > 1 else ''}"
                      f" — {_crit_high} crítico/alto"  if _crit_high else
                      f"🔔 {len(_alerts)} alerta{'s' if len(_alerts) > 1 else ''} ativo{'s' if len(_alerts) > 1 else ''}")
            with st.expander(_label, expanded=_crit_high > 0):
                for _a in _alerts:
                    _c = _sev_colors.get(_a["severity"], "#999")
                    st.markdown(
                        f'<div style="border-left:4px solid {_c};padding:6px 12px;'
                        f'margin:4px 0;border-radius:0 4px 4px 0;background:rgba(0,0,0,0.15)">'
                        f'{_sev_icons.get(_a["severity"],"•")} '
                        f'<strong>{_a["asset"]}</strong> — {_a["message"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.caption("Configure os limites na aba ⚙️ Configurações → Alertas de Risco.")
        else:
            st.success("✅ Nenhum alerta ativo — portfólio dentro dos limites configurados.", icon="✅")

        # Status dos agentes
        display_agent_status()

        # Gráficos principais
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Desempenho Normalizado (Base 100)")
            fig_prices = go.Figure()
            for asset in prices_filtered.columns:
                fig_prices.add_trace(go.Scatter(
                    x=prices_filtered.index,
                    y=prices_filtered[asset],
                    name=asset,
                    line=dict(width=2)
                ))
            bench_colors = ["#f39c12", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22"]
            if not benchmark_df.empty:
                for i, col in enumerate(benchmark_df.columns):
                    fig_prices.add_trace(go.Scatter(
                        x=benchmark_df.index,
                        y=benchmark_df[col],
                        name=col,
                        line=dict(width=2, dash="dash",
                                  color=bench_colors[i % len(bench_colors)]),
                        opacity=0.85
                    ))
            if not macro_df.empty:
                for col in macro_df.columns:
                    fig_prices.add_trace(go.Scatter(
                        x=macro_df.index,
                        y=macro_df[col],
                        name=col,
                        line=dict(width=1, dash="dot"),
                        yaxis="y2",
                        opacity=0.7
                    ))
            fig_prices.update_layout(
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f2f6'),
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(gridcolor='#333', title="Base 100"),
                yaxis2=dict(overlaying="y", side="right", gridcolor='#222',
                            title="BACEN", showgrid=False) if not macro_df.empty else {},
                legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_prices, use_container_width=True)
            _dl_chart_btn("⬇️ Preços (HTML)", fig_prices, "precos_normalizados", "dl_prices", use_container_width=True)

        with col2:
            st.subheader("📈 Retornos Acumulados")
            cumulative_returns = (1 + returns_filtered).cumprod()
            fig_cumulative = go.Figure()
            for asset in cumulative_returns.columns:
                fig_cumulative.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[asset],
                    name=asset,
                    line=dict(width=2)
                ))
            fig_cumulative.update_layout(
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f2f6'),
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(gridcolor='#333')
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
            _dl_chart_btn("⬇️ Retornos Acumulados (HTML)", fig_cumulative, "retornos_acumulados", "dl_cumulative", use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">🤖 Sistema Multiagente Quantitativo</div>', unsafe_allow_html=True)

    # ── Seleção de agentes ────────────────────────────────────────────────────
    _ALL_AGENTS = {
        # label, perfis sugeridos
        "AgentMarket":          ("📈 Market — volatilidade, correlação, regime",         ["quant", "credito", "tesoureiro", "research"]),
        "AgentClustering":      ("🎯 Clustering — K-Means + PCA por similaridade",      ["quant"]),
        "AgentML":              ("🤖 ML — Isolation Forest + Random Forest",             ["quant", "credito"]),
        "AgentSimulation":      ("🎲 Simulation — Monte Carlo / Bootstrap / GARCH",      ["quant", "tesoureiro"]),
        "AgentLSTM":            ("🧠 LSTM — previsão de tendência (MLP temporal)",       ["quant", "research"]),
        "AgentAutoencoder":     ("🔬 Autoencoder — detecção de anomalias",               ["quant"]),
        "AgentFundamental":     ("📊 Fundamental — P/E, P/B, EV/EBITDA, DCF",           ["research"]),
        "AgentCredit":          ("🏦 Credit — Dív.Líq/EBITDA, ICR, Liquidez, FCF",      ["credito"]),
        "AgentDividend":        ("💰 Dividend — DY, payout, consistência, CAGR 5Y",      ["research"]),
        "AgentPeerComparison":  ("🔁 Peer — ranking relativo por setor (z-score)",       ["research"]),
        "AgentMacroSensitivity":("🏛️ Macro — beta a CDI, IPCA, câmbio (rolling OLS)",   ["tesoureiro", "credito"]),
        "AgentScenario":        ("⚡ Cenário — stress SELIC+300bps / IPCA+4% / FX+20%", ["tesoureiro", "credito"]),
        "AgentCVM":             ("🗂️ CVM — séries contábeis B3 via CVM Dados Abertos",  ["research", "credito"]),
        "AgentScreener":        ("🏆 Screener — ranking composto (Fund+Créd+Div)",       ["research", "credito"]),
    }

    # Perfis de sugestão (não restritivo — apenas pré-marca checkboxes)
    _PROFILES = {
        "Todos":                None,
        "📊 Research (Equity)": ["AgentFundamental", "AgentDividend", "AgentPeerComparison", "AgentMarket", "AgentLSTM", "AgentCVM", "AgentScreener"],
        "🏦 Crédito":           ["AgentCredit", "AgentScenario", "AgentMacroSensitivity", "AgentMarket", "AgentML", "AgentCVM", "AgentScreener"],
        "🏛️ Tesoureiro":        ["AgentMacroSensitivity", "AgentScenario", "AgentSimulation", "AgentMarket"],
        "🔍 Quant / Risco":     ["AgentMarket", "AgentML", "AgentSimulation", "AgentLSTM", "AgentAutoencoder", "AgentClustering"],
    }

    with st.expander("⚙️ Selecionar agentes para executar", expanded=False):
        # Seletor de perfil
        _profile_key = st.selectbox(
            "Sugestão por perfil (não restringe seleção):",
            list(_PROFILES.keys()),
            key="agent_profile_sel",
            help="Selecionar um perfil pré-marca os agentes mais relevantes, mas você pode ajustar livremente.",
        )
        _profile_suggestion = _PROFILES[_profile_key]

        if _profile_suggestion is not None and st.button("Aplicar sugestão de perfil", key="apply_profile"):
            for k in _ALL_AGENTS:
                st.session_state[f"agent_chk_{k}"] = k in _profile_suggestion

        st.caption("Todos os agentes podem ser usados independentemente do perfil.")
        st.markdown("---")

        _sel_cols = st.columns(2)
        _agent_enabled = {}
        for _i, (_key, (_label, _profiles)) in enumerate(_ALL_AGENTS.items()):
            # badge de perfis sugeridos
            _badge = " ".join(
                {"research": "📊", "credito": "🏦", "tesoureiro": "🏛️", "quant": "🔍"}.get(p, "") for p in _profiles
            ).strip()
            _display = f"{_label}  {_badge}" if _badge else _label
            with _sel_cols[_i % 2]:
                _default = True if _profile_suggestion is None else (_key in _profile_suggestion)
                _agent_enabled[_key] = st.checkbox(
                    _display,
                    value=st.session_state.get(f"agent_chk_{_key}", _default),
                    key=f"agent_chk_{_key}",
                )
        _agents_to_run = [k for k, v in _agent_enabled.items() if v]
        if not _agents_to_run:
            st.warning("Selecione ao menos um agente.")

    # ── Botões de controle ────────────────────────────────────────────────────
    _btn_col, _rst_col = st.columns([2, 1])

    with _btn_col:
        _n_sel = len(_agents_to_run) if _agents_to_run else 0
        _run_label = (f"🚀 Executar {_n_sel} agente{'s' if _n_sel != 1 else ''}"
                      if _n_sel < len(_ALL_AGENTS)
                      else "🚀 Executar Análise Completa")
        if st.button(_run_label, type="primary", use_container_width=True,
                     key="run_agents_main", disabled=not _agents_to_run):
            if st.session_state.orchestrator:
                with st.spinner(f"🤖 Executando {_n_sel} agente(s)..."):
                    try:
                        _agent_portfolio_config = {
                            "macro_df":   macro_df if not macro_df.empty else None,
                            "value":      st.session_state.get("initial_investment", 100_000),
                            "weights":    {tk: 1.0 / len(prices_filtered.columns)
                                           for tk in prices_filtered.columns},
                            "confidence_level": st.session_state.get("confidence_level", 0.95),
                        }
                        alerts = st.session_state.orchestrator.run_analysis(
                            prices_filtered,
                            portfolio_config=_agent_portfolio_config,
                            enabled_agents=_agents_to_run if _n_sel < len(_ALL_AGENTS) else None,
                        )
                        report = st.session_state.orchestrator.generate_report(alerts)
                        st.session_state.last_agent_report = report
                        st.session_state.last_agent_alerts = alerts
                        st.success(f"✅ {_n_sel} agente(s) concluídos! Veja os resultados abaixo.")
                    except Exception as e:
                        st.error(f"❌ Erro na execução: {str(e)}")
            else:
                st.error("❌ Sistema multiagente não inicializado!")

    with _rst_col:
        if st.button("🔄 Reinicializar", use_container_width=True, key="restart_agents"):
            with st.spinner("Reinicializando..."):
                st.session_state.orchestrator = initialize_multiagent_system()
                st.session_state.pop("last_agent_report", None)
                st.session_state.pop("last_agent_alerts", None)
                st.rerun()
    
    # Exibir resultados se disponíveis
    if hasattr(st.session_state, 'last_agent_report'):
        (tab_alertas, tab_analytics, tab_lstm, tab_ae,
         tab_fund, tab_credit, tab_div, tab_peer, tab_macro, tab_scenario,
         tab_cvm, tab_screener) = st.tabs([
            "🚨 Alertas", "📊 Analytics", "🧠 LSTM", "🔬 Autoencoder",
            "📊 Fundamentals", "🏦 Crédito",
            "💰 Dividendos", "🔁 Peer", "🏛️ Macro", "⚡ Cenários",
            "🗂️ CVM", "🏆 Screener",
        ])

        with tab_alertas:
            display_all_alerts_with_filters(st.session_state.last_agent_report)

        with tab_analytics:
            if hasattr(st.session_state, 'last_agent_alerts'):
                display_agent_analytics(st.session_state.last_agent_alerts)

        # ── LSTM ─────────────────────────────────────────────────────────────
        with tab_lstm:
            st.markdown("### 🧠 LSTM — Previsão de Tendência (MLP Temporal)")
            st.caption(
                "Rede neural treinada com janela deslizante de 20 dias. "
                "Prevê retornos dos próximos 5 dias por ativo."
            )
            _lstm_res = st.session_state.orchestrator.agent_lstm.lstm_results \
                        if st.session_state.orchestrator else {}

            if not _lstm_res:
                st.info("Execute a análise multiagente para ver as previsões LSTM.")
            else:
                _trend_icons = {"alta": "📈", "baixa": "📉", "lateral": "➡️"}
                _trend_colors= {"alta": "#2ecc71", "baixa": "#e74c3c", "lateral": "#f39c12"}

                # Cards de tendência por ativo
                _asset_list = list(_lstm_res.keys())
                _cols = st.columns(min(len(_asset_list), 4))
                for _i, _asset in enumerate(_asset_list):
                    _r = _lstm_res[_asset]
                    with _cols[_i % min(len(_asset_list), 4)]:
                        _icon  = _trend_icons.get(_r["trend"], "")
                        _color = _trend_colors.get(_r["trend"], "#fff")
                        st.markdown(
                            f'<div style="border:2px solid {_color};border-radius:8px;'
                            f'padding:10px;text-align:center;margin:4px 0">'
                            f'<b>{_asset}</b><br/>'
                            f'<span style="font-size:1.4rem">{_icon}</span><br/>'
                            f'<b style="color:{_color}">{_r["trend"].upper()}</b><br/>'
                            f'<small>{_r["trend_avg"]:+.3%}/dia</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # Gráfico: Real vs Previsto (período de teste)
                _sel = st.selectbox("Ativo para detalhar:", _asset_list, key="lstm_asset_sel")
                _r   = _lstm_res[_sel]

                _fig_lstm = go.Figure()
                _x_test = list(range(len(_r["actual"])))
                _fig_lstm.add_trace(go.Scatter(
                    x=_x_test, y=_r["actual"],
                    name="Real", line=dict(color="#00d4aa", width=2)
                ))
                _fig_lstm.add_trace(go.Scatter(
                    x=_x_test, y=_r["predicted"],
                    name="Previsto (teste)", line=dict(color="#f39c12", width=2, dash="dash")
                ))

                # Forecast: continua após o último ponto de teste
                _fc_x = list(range(len(_r["actual"]), len(_r["actual"]) + len(_r["forecast"])))
                _fig_lstm.add_trace(go.Scatter(
                    x=_fc_x, y=_r["forecast"],
                    name=f"Previsão ({len(_r['forecast'])} dias)",
                    line=dict(color="#9b59b6", width=2, dash="dot"),
                    mode="lines+markers",
                ))
                _fig_lstm.add_vrect(
                    x0=len(_r["actual"]) - 0.5, x1=len(_r["actual"]) + len(_r["forecast"]) - 0.5,
                    fillcolor="rgba(155,89,182,0.08)", line_width=0,
                    annotation_text="Previsão", annotation_position="top left",
                )
                _fig_lstm.update_layout(
                    title=f"LSTM — {_sel}: Real vs Previsto vs Forecast",
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f2f6"),
                    xaxis=dict(title="Dias (período de teste)", gridcolor="#333"),
                    yaxis=dict(title="Retorno diário", tickformat=".2%", gridcolor="#333"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(_fig_lstm, use_container_width=True)
                st.caption(f"MAE no conjunto de teste: **{_r['mae']:.4f}** ({_r['test_size']} dias)")
                _dl_chart_btn("⬇️ Gráfico LSTM (HTML)", _fig_lstm, f"lstm_{_sel}", "dl_lstm_chart")

                # Excel: todos os ativos LSTM
                _lstm_buf = BytesIO()
                with pd.ExcelWriter(_lstm_buf, engine="xlsxwriter") as _lw:
                    # Resumo: tendência por ativo
                    _summary = [{
                        "Ativo": _a,
                        "Tendência": _lstm_res[_a]["trend"],
                        "Média/dia (%)": _lstm_res[_a]["trend_avg"],
                        "MAE": _lstm_res[_a]["mae"],
                        "Dias de teste": _lstm_res[_a]["test_size"],
                        "Forecast d+1": _lstm_res[_a]["forecast"][0] if _lstm_res[_a]["forecast"] else None,
                        "Forecast d+2": _lstm_res[_a]["forecast"][1] if len(_lstm_res[_a]["forecast"]) > 1 else None,
                        "Forecast d+3": _lstm_res[_a]["forecast"][2] if len(_lstm_res[_a]["forecast"]) > 2 else None,
                        "Forecast d+4": _lstm_res[_a]["forecast"][3] if len(_lstm_res[_a]["forecast"]) > 3 else None,
                        "Forecast d+5": _lstm_res[_a]["forecast"][4] if len(_lstm_res[_a]["forecast"]) > 4 else None,
                    } for _a in _lstm_res]
                    pd.DataFrame(_summary).to_excel(_lw, sheet_name="Resumo LSTM", index=False)
                    # Série temporal por ativo
                    for _a, _ar in _lstm_res.items():
                        _n = min(len(_ar["actual"]), len(_ar["predicted"]))
                        pd.DataFrame({
                            "Real": _ar["actual"][:_n],
                            "Previsto": _ar["predicted"][:_n],
                        }).to_excel(_lw, sheet_name=(_a[:28] + "_lstm")[:31], index=False)
                _lstm_buf.seek(0)
                st.download_button(
                    "⬇️ Dados LSTM (Excel — todos os ativos)",
                    data=_lstm_buf,
                    file_name=f"lstm_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_lstm_excel",
                )

        # ── Autoencoder ───────────────────────────────────────────────────────
        with tab_ae:
            st.markdown("### 🔬 Autoencoder — Detecção de Anomalias")
            st.caption(
                "MLP com camada bottleneck treinada para reconstruir retornos normais. "
                "Alto erro de reconstrução = padrão anômalo."
            )
            _ae_res = st.session_state.orchestrator.agent_autoencoder.autoencoder_results \
                      if st.session_state.orchestrator else {}

            if not _ae_res:
                st.info("Execute a análise multiagente para ver os resultados do Autoencoder.")
            else:
                # Métricas resumidas
                _c1, _c2, _c3, _c4 = st.columns(4)
                with _c1:
                    st.metric("Anomalias detectadas", _ae_res["n_anomalies"])
                with _c2:
                    st.metric("Últimos 5 dias", _ae_res["recent_anomalies"],
                              delta="ATENÇÃO" if _ae_res["recent_anomalies"] > 0 else "Normal",
                              delta_color="inverse" if _ae_res["recent_anomalies"] > 0 else "off")
                with _c3:
                    st.metric("Taxa de anomalia", f"{_ae_res['contamination']:.1%}")
                with _c4:
                    st.metric("Threshold MSE", f"{_ae_res['threshold']:.6f}")

                # Série de erro de reconstrução ao longo do tempo
                _errors  = np.array(_ae_res["errors"])
                _dates   = _ae_res["dates"]
                _is_anom = np.array(_ae_res["is_anomaly"])

                _fig_ae = go.Figure()
                _fig_ae.add_trace(go.Scatter(
                    x=_dates, y=_errors,
                    name="Erro de reconstrução (MSE)",
                    line=dict(color="#00d4aa", width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
                ))
                _fig_ae.add_hline(
                    y=_ae_res["threshold"],
                    line=dict(color="#e74c3c", dash="dash", width=1.5),
                    annotation_text="Threshold",
                )
                # Marcar anomalias
                _anom_dates  = [_dates[i] for i, a in enumerate(_is_anom) if a]
                _anom_errors = [_errors[i] for i, a in enumerate(_is_anom) if a]
                if _anom_dates:
                    _fig_ae.add_trace(go.Scatter(
                        x=_anom_dates, y=_anom_errors,
                        mode="markers", name="Anomalia",
                        marker=dict(color="#e74c3c", size=8, symbol="x"),
                    ))
                _fig_ae.update_layout(
                    title="Erro de Reconstrução — Autoencoder",
                    height=380,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f2f6"),
                    xaxis=dict(gridcolor="#333"),
                    yaxis=dict(gridcolor="#333", title="MSE"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(_fig_ae, use_container_width=True)

                # Contribuição por ativo
                st.markdown("#### Contribuição por Ativo")
                _err_assets = _ae_res["errors_by_asset"]
                _fig_bar = go.Figure(go.Bar(
                    x=list(_err_assets.keys()),
                    y=list(_err_assets.values()),
                    marker_color=[
                        "#e74c3c" if e > 2 * np.mean(list(_err_assets.values())) else "#00d4aa"
                        for e in _err_assets.values()
                    ],
                ))
                _fig_bar.update_layout(
                    title="Erro médio de reconstrução por ativo",
                    height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f2f6"),
                    yaxis=dict(gridcolor="#333", title="MSE médio"),
                    xaxis=dict(gridcolor="#333"),
                )
                st.plotly_chart(_fig_bar, use_container_width=True)
                st.caption("Barras vermelhas = ativos com padrão de retorno mais atípico (MSE > 2× média).")

                # Downloads Autoencoder
                _dl_chart_btn("⬇️ Erro de Reconstrução (HTML)", _fig_ae, "autoencoder_erros", "dl_ae_chart")

                _ae_buf = BytesIO()
                with pd.ExcelWriter(_ae_buf, engine="xlsxwriter") as _aw:
                    # Série temporal de erros
                    pd.DataFrame({
                        "Data": list(_ae_res["dates"]),
                        "Erro MSE": list(_ae_res["errors"]),
                        "Anomalia": [bool(b) for b in _ae_res["is_anomaly"]],
                    }).to_excel(_aw, sheet_name="Erros Temporais", index=False)
                    # Erro por ativo
                    pd.DataFrame(
                        list(_ae_res["errors_by_asset"].items()),
                        columns=["Ativo", "MSE Médio"]
                    ).to_excel(_aw, sheet_name="Erro por Ativo", index=False)
                    # Resumo
                    pd.DataFrame([{
                        "N anomalias": _ae_res["n_anomalies"],
                        "Anomalias últimos 5 dias": _ae_res["recent_anomalies"],
                        "Taxa de anomalia": _ae_res["contamination"],
                        "Threshold MSE": _ae_res["threshold"],
                    }]).to_excel(_aw, sheet_name="Resumo", index=False)
                _ae_buf.seek(0)
                st.download_button(
                    "⬇️ Dados Autoencoder (Excel)",
                    data=_ae_buf,
                    file_name=f"autoencoder_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_ae_excel",
                )

        # ── AgentFundamental ─────────────────────────────────────────────────
        with tab_fund:
            st.markdown("### 📊 Análise Fundamentalista")
            st.caption("P/E, P/B, EV/EBITDA, margem EBITDA, crescimento de receita e DCF simplificado via yfinance.")
            _orch = st.session_state.orchestrator
            _fund_res = {}
            if _orch and hasattr(_orch, "agent_fundamental"):
                _fund_res = getattr(_orch.agent_fundamental, "fundamental_results", {}) or {}

            if not _fund_res:
                st.info("Execute a análise multiagente com **AgentFundamental** selecionado para ver os múltiplos.")
            else:
                # Tabela resumo de múltiplos
                _fund_rows = []
                for _tk, _fr in _fund_res.items():
                    _fund_rows.append({
                        "Ativo":            _fr.get("name", _tk),
                        "Ticker":           _tk,
                        "Setor":            _fr.get("sector", "N/D"),
                        "P/E":              _fr.get("pe_ratio"),
                        "P/B":              _fr.get("pb_ratio"),
                        "EV/EBITDA":        _fr.get("ev_ebitda"),
                        "Marg. EBITDA":     _fr.get("ebitda_margin"),
                        "Cresc. Receita":   _fr.get("revenue_growth_yoy"),
                        "DY":               _fr.get("dividend_yield"),
                        "DCF Upside":       _fr.get("dcf_upside"),
                        "Market Cap (R$M)": (_fr.get("market_cap") or 0) / 1e6,
                    })
                _fund_df = pd.DataFrame(_fund_rows).set_index("Ticker")

                def _fmt_fund(df):
                    fmt = {
                        "P/E":            "{:.1f}",
                        "P/B":            "{:.2f}",
                        "EV/EBITDA":      "{:.1f}",
                        "Marg. EBITDA":   "{:.1%}",
                        "Cresc. Receita": "{:.1%}",
                        "DY":             "{:.2%}",
                        "DCF Upside":     "{:.1%}",
                        "Market Cap (R$M)": "{:,.0f}",
                    }
                    styled = df.style
                    for col, f in fmt.items():
                        if col in df.columns:
                            styled = styled.format(f, subset=[col], na_rep="N/D")
                    if "DCF Upside" in df.columns:
                        styled = styled.map(
                            lambda v: "color: #2ecc71" if isinstance(v, float) and v > 0.10
                            else ("color: #e74c3c" if isinstance(v, float) and v < -0.10 else ""),
                            subset=["DCF Upside"],
                        )
                    return styled

                st.dataframe(_fmt_fund(_fund_df), use_container_width=True)

                # Gráficos
                _valid = [r for r in _fund_rows if r.get("EV/EBITDA") is not None]
                if _valid:
                    _fc1, _fc2 = st.columns(2)
                    with _fc1:
                        _fig_pe = px.bar(
                            pd.DataFrame(_valid),
                            x="Ticker", y="EV/EBITDA",
                            color="EV/EBITDA",
                            color_continuous_scale="RdYlGn_r",
                            title="EV/EBITDA por Ativo",
                        )
                        _fig_pe.update_layout(
                            height=350, plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f2f6"),
                        )
                        st.plotly_chart(_fig_pe, use_container_width=True)
                        _dl_chart_btn("⬇️ EV/EBITDA (HTML)", _fig_pe, "ev_ebitda", "dl_ev_ebitda")

                    with _fc2:
                        _dcf_valid = [r for r in _fund_rows if r.get("DCF Upside") is not None]
                        if _dcf_valid:
                            _fig_dcf = px.bar(
                                pd.DataFrame(_dcf_valid),
                                x="Ticker", y="DCF Upside",
                                color="DCF Upside",
                                color_continuous_scale="RdYlGn",
                                title="DCF Upside/Downside vs. Preço Atual",
                            )
                            _fig_dcf.update_layout(
                                height=350, plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f2f6"),
                                yaxis=dict(tickformat=".0%"),
                            )
                            _fig_dcf.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                            st.plotly_chart(_fig_dcf, use_container_width=True)
                            _dl_chart_btn("⬇️ DCF Upside (HTML)", _fig_dcf, "dcf_upside", "dl_dcf_upside")

                # Excel export
                _fund_buf = BytesIO()
                with pd.ExcelWriter(_fund_buf, engine="xlsxwriter") as _fw:
                    _fund_df.reset_index().to_excel(_fw, sheet_name="Múltiplos", index=False)
                    # DCF detail
                    _dcf_rows = [{
                        "Ticker": _tk,
                        "FCF": _fr.get("fcf"),
                        "Market Cap": _fr.get("market_cap"),
                        "DCF Value (WACC=12%, g=4%)": _fr.get("dcf_value"),
                        "DCF Upside": _fr.get("dcf_upside"),
                    } for _tk, _fr in _fund_res.items()]
                    pd.DataFrame(_dcf_rows).to_excel(_fw, sheet_name="DCF", index=False)
                _fund_buf.seek(0)
                st.download_button(
                    "⬇️ Fundamentals (Excel)",
                    data=_fund_buf,
                    file_name=f"fundamentals_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_fund_excel",
                )
                st.caption(
                    "Cobertura via yfinance: excelente para PETR4.SA, VALE3.SA, ITUB4.SA etc. "
                    "Small/mid caps B3 podem ter dados parciais."
                )

        # ── AgentCredit ──────────────────────────────────────────────────────
        with tab_credit:
            st.markdown("### 🏦 Análise de Crédito")
            st.caption("Dívida Líq./EBITDA, ICR, Liquidez Corrente, FCF Yield e score proprietário (0–100).")
            _credit_res = {}
            if _orch and hasattr(_orch, "agent_credit"):
                _credit_res = getattr(_orch.agent_credit, "credit_results", {}) or {}

            if not _credit_res:
                st.info("Execute a análise multiagente com **AgentCredit** selecionado para ver o score de crédito.")
            else:
                _credit_rows = []
                for _tk, _cr in _credit_res.items():
                    _credit_rows.append({
                        "Ativo":             _cr.get("name", _tk),
                        "Ticker":            _tk,
                        "Setor":             _cr.get("sector", "N/D"),
                        "Score (0-100)":     _cr.get("credit_score"),
                        "Rating":            _cr.get("credit_rating", "N/D"),
                        "Dív.Líq./EBITDA":   _cr.get("net_debt_ebitda"),
                        "ICR (Cob.Juros)":   _cr.get("interest_coverage"),
                        "Liquidez Corr.":    _cr.get("current_ratio"),
                        "FCF Yield":         _cr.get("fcf_yield"),
                    })
                _credit_df = pd.DataFrame(_credit_rows).set_index("Ticker")

                # Score cards no topo
                _scored = [r for r in _credit_rows if r.get("Score (0-100)") is not None]
                if _scored:
                    _card_cols = st.columns(min(len(_scored), 4))
                    _rating_color = {
                        "BAIXO RISCO": "#2ecc71", "RISCO MODERADO": "#f39c12",
                        "RISCO ELEVADO": "#e67e22", "ALTO RISCO": "#e74c3c",
                    }
                    for _ci, _cr_row in enumerate(_scored[:4]):
                        with _card_cols[_ci % min(len(_scored), 4)]:
                            _col = _rating_color.get(_cr_row["Rating"], "#aaa")
                            st.markdown(
                                f'<div style="border:2px solid {_col};border-radius:8px;'
                                f'padding:10px;text-align:center;margin:4px 0">'
                                f'<b>{_cr_row["Ativo"]}</b><br/>'
                                f'<span style="font-size:1.6rem;font-weight:bold;color:{_col}">'
                                f'{_cr_row["Score (0-100)"]:.0f}</span><br/>'
                                f'<small style="color:{_col}">{_cr_row["Rating"]}</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    st.markdown("---")

                # Tabela formatada
                def _fmt_credit(df):
                    fmt = {
                        "Score (0-100)": "{:.0f}",
                        "Dív.Líq./EBITDA": "{:.2f}x",
                        "ICR (Cob.Juros)": "{:.2f}x",
                        "Liquidez Corr.": "{:.2f}x",
                        "FCF Yield": "{:.2%}",
                    }
                    styled = df.style
                    for col, f in fmt.items():
                        if col in df.columns:
                            styled = styled.format(f, subset=[col], na_rep="N/D")
                    if "Score (0-100)" in df.columns:
                        styled = styled.map(
                            lambda v: (
                                "background-color: rgba(46,204,113,0.2)" if isinstance(v, float) and v >= 70
                                else "background-color: rgba(231,76,60,0.2)" if isinstance(v, float) and v < 40
                                else ""
                            ),
                            subset=["Score (0-100)"],
                        )
                    return styled

                st.dataframe(_fmt_credit(_credit_df), use_container_width=True)

                # Gráfico de score
                if _scored:
                    _fig_score = px.bar(
                        pd.DataFrame(_scored).sort_values("Score (0-100)"),
                        x="Score (0-100)", y="Ticker",
                        orientation="h",
                        color="Score (0-100)",
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 100],
                        title="Score de Crédito por Ativo (0 = alto risco, 100 = baixo risco)",
                        hover_data=["Rating", "Díd.Líq./EBITDA", "ICR (Cob.Juros)"] if "Díd.Líq./EBITDA" in pd.DataFrame(_scored).columns else ["Rating"],
                    )
                    _fig_score.add_vline(x=70, line_dash="dash", line_color="#2ecc71",
                                         annotation_text="Baixo Risco")
                    _fig_score.add_vline(x=40, line_dash="dash", line_color="#e74c3c",
                                         annotation_text="Alto Risco")
                    _fig_score.update_layout(
                        height=max(300, len(_scored) * 55),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f0f2f6"),
                        xaxis=dict(range=[0, 100], gridcolor="#333"),
                    )
                    st.plotly_chart(_fig_score, use_container_width=True)
                    _dl_chart_btn("⬇️ Score de Crédito (HTML)", _fig_score, "credit_score", "dl_credit_chart")

                # Excel export
                _cr_buf = BytesIO()
                with pd.ExcelWriter(_cr_buf, engine="xlsxwriter") as _cw:
                    _credit_df.reset_index().to_excel(_cw, sheet_name="Score de Crédito", index=False)
                    # Score breakdown
                    _bk_rows = [{
                        "Ticker": _tk,
                        "Score Total": _cr_d.get("credit_score"),
                        "Rating": _cr_d.get("credit_rating"),
                        **{f"Pts_{k}": v for k, v in (_cr_d.get("score_details") or {}).items()}
                    } for _tk, _cr_d in _credit_res.items()]
                    pd.DataFrame(_bk_rows).to_excel(_cw, sheet_name="Score Breakdown", index=False)
                _cr_buf.seek(0)
                st.download_button(
                    "⬇️ Análise de Crédito (Excel)",
                    data=_cr_buf,
                    file_name=f"credito_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_credit_excel",
                )

        # ── AgentDividend ────────────────────────────────────────────────────
        with tab_div:
            st.markdown("### 💰 Análise de Dividendos")
            st.caption("DY histórico, crescimento 5Y, payout ratio e consistência de pagamentos via yfinance.")
            _orch = st.session_state.orchestrator
            _div_res = getattr(getattr(_orch, "agent_dividend", None), "dividend_results", {}) or {}

            if not _div_res:
                st.info("Execute a análise com **AgentDividend** para ver o histórico de dividendos.")
            else:
                _div_rows = [{
                    "Ticker":       _tk,
                    "Nome":         _dr.get("name", _tk),
                    "DY Trailing":  _dr.get("dy_trailing"),
                    "CAGR 5Y":      _dr.get("dividend_cagr_5y"),
                    "Consistência": _dr.get("consistency_score"),
                    "Anos Pagando": _dr.get("years_paying"),
                    "Payout Ratio": _dr.get("payout_ratio"),
                    "Div. 12m":     _dr.get("trailing_dividends_1y"),
                } for _tk, _dr in _div_res.items()]
                _div_df = pd.DataFrame(_div_rows).set_index("Ticker")

                st.dataframe(_div_df.style.format({
                    "DY Trailing":  "{:.2%}", "CAGR 5Y": "{:.2%}",
                    "Consistência": "{:.0%}", "Payout Ratio": "{:.2%}",
                    "Div. 12m":     "{:.4f}",
                }, na_rep="N/D"), use_container_width=True)

                _dc1, _dc2 = st.columns(2)
                _dy_valid = [r for r in _div_rows if r.get("DY Trailing") is not None]
                if _dy_valid:
                    with _dc1:
                        _fig_dy = px.bar(pd.DataFrame(_dy_valid).sort_values("DY Trailing", ascending=False),
                            x="Ticker", y="DY Trailing", color="DY Trailing",
                            color_continuous_scale="RdYlGn", title="Dividend Yield por Ativo")
                        _fig_dy.update_layout(height=350, yaxis=dict(tickformat=".1%"),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#f0f2f6"))
                        st.plotly_chart(_fig_dy, use_container_width=True)
                        _dl_chart_btn("⬇️ DY (HTML)", _fig_dy, "dividend_yield", "dl_dy_chart")

                _cagr_valid = [r for r in _div_rows if r.get("CAGR 5Y") is not None]
                if _cagr_valid:
                    with _dc2:
                        _fig_cagr = px.bar(pd.DataFrame(_cagr_valid).sort_values("CAGR 5Y"),
                            x="Ticker", y="CAGR 5Y", color="CAGR 5Y",
                            color_continuous_scale="RdYlGn", title="Crescimento de Dividendos CAGR 5Y")
                        _fig_cagr.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                        _fig_cagr.update_layout(height=350, yaxis=dict(tickformat=".1%"),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#f0f2f6"))
                        st.plotly_chart(_fig_cagr, use_container_width=True)
                        _dl_chart_btn("⬇️ CAGR Dividendos (HTML)", _fig_cagr, "div_cagr", "dl_cagr_chart")

                _div_buf = BytesIO()
                pd.DataFrame(_div_rows).to_excel(_div_buf, index=False, engine="xlsxwriter")
                _div_buf.seek(0)
                st.download_button("⬇️ Dividendos (Excel)", _div_buf,
                    f"dividendos_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_div_excel")

        # ── AgentPeerComparison ──────────────────────────────────────────────
        with tab_peer:
            st.markdown("### 🔁 Comparação entre Pares (Peer Comparison)")
            st.caption("Ranking relativo de múltiplos (P/E, P/B, EV/EBITDA, DY) por setor. Z-score > |2| indica outlier.")
            _peer_res = getattr(getattr(_orch, "agent_peer", None), "peer_results", {}) or {}

            if not _peer_res:
                st.info("Execute a análise com **AgentPeerComparison** para ver o ranking setorial.")
            else:
                _comparison = _peer_res.get("comparison", {})
                _ticker_info = _peer_res.get("ticker_info", {})
                _metrics_lbl = {"pe": "P/E", "pb": "P/B", "ev_ebitda": "EV/EBITDA", "dy": "DY"}

                for _sector, _comp in _comparison.items():
                    if _comp.get("solo"):
                        st.markdown(f"**{_sector}** — apenas 1 ativo, sem comparação possível.")
                        continue
                    with st.expander(f"**{_sector}** ({len(_comp['members'])} ativos)", expanded=True):
                        _peer_rows = []
                        for _tk in _comp["members"]:
                            _row = {"Ticker": _tk, "Nome": _ticker_info.get(_tk, {}).get("name", _tk)}
                            for _m, _lbl in _metrics_lbl.items():
                                if _m in _comp:
                                    _r = _comp[_m]["rankings"].get(_tk, {})
                                    _row[f"{_lbl} Valor"] = _r.get("value")
                                    _row[f"{_lbl} Z"]     = _r.get("zscore")
                            _peer_rows.append(_row)

                        _peer_tbl = pd.DataFrame(_peer_rows).set_index("Ticker")
                        _z_cols = [c for c in _peer_tbl.columns if c.endswith(" Z")]

                        def _color_z(v):
                            if not isinstance(v, float): return ""
                            if v > 1.5:  return "color: #e74c3c"
                            if v < -1.5: return "color: #2ecc71"
                            return ""

                        _peer_num = _peer_tbl.select_dtypes(include="number").columns.tolist()
                        _peer_fmt = {c: "{:.2f}" for c in _peer_num}
                        styled = _peer_tbl.style.format(_peer_fmt, na_rep="—")
                        if _z_cols:
                            styled = styled.map(_color_z, subset=_z_cols)
                        st.dataframe(styled, use_container_width=True)

                        # Radar de z-scores por ativo
                        _radar_data = []
                        for _m, _lbl in _metrics_lbl.items():
                            if _m in _comp:
                                for _tk, _r in _comp[_m]["rankings"].items():
                                    _radar_data.append({"Métrica": _lbl, "Ticker": _tk, "Z-score": _r["zscore"]})
                        if _radar_data:
                            _fig_radar = px.bar(pd.DataFrame(_radar_data), x="Métrica", y="Z-score",
                                color="Ticker", barmode="group",
                                title=f"Z-score por Métrica — {_sector}",
                                color_discrete_sequence=px.colors.qualitative.Set2)
                            _fig_radar.add_hline(y=2,  line_dash="dash", line_color="red",   opacity=0.5)
                            _fig_radar.add_hline(y=-2, line_dash="dash", line_color="green", opacity=0.5)
                            _fig_radar.update_layout(height=320,
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#f0f2f6"))
                            st.plotly_chart(_fig_radar, use_container_width=True)

                _peer_buf = BytesIO()
                with pd.ExcelWriter(_peer_buf, engine="xlsxwriter") as _pw:
                    _all_peer_rows = []
                    for _sector, _comp in _comparison.items():
                        if _comp.get("solo"): continue
                        for _tk in _comp["members"]:
                            _row = {"Setor": _sector, "Ticker": _tk,
                                    "Nome": _ticker_info.get(_tk, {}).get("name", _tk)}
                            for _m, _lbl in _metrics_lbl.items():
                                if _m in _comp:
                                    _r = _comp[_m]["rankings"].get(_tk, {})
                                    _row[f"{_lbl}"] = _r.get("value")
                                    _row[f"{_lbl}_Zscore"] = _r.get("zscore")
                                    _row[f"{_lbl}_Pct"] = _r.get("pct_rank")
                            _all_peer_rows.append(_row)
                    if _all_peer_rows:
                        pd.DataFrame(_all_peer_rows).to_excel(_pw, sheet_name="Peer Comparison", index=False)
                _peer_buf.seek(0)
                st.download_button("⬇️ Peer Comparison (Excel)", _peer_buf,
                    f"peer_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_peer_excel")

        # ── AgentMacroSensitivity ────────────────────────────────────────────
        with tab_macro:
            st.markdown("### 🏛️ Sensibilidade Macroeconômica")
            st.caption("Beta rolling 63d de cada ativo a CDI, IPCA e câmbio via regressão OLS histórica.")
            _macro_res = getattr(getattr(_orch, "agent_macro", None), "macro_sensitivity_results", {}) or {}

            if not _macro_res or not _macro_res.get("assets"):
                st.info("Execute a análise com **AgentMacroSensitivity** para ver os betas macroeconômicos.")
            else:
                _factors  = _macro_res.get("factors", [])
                _assets   = _macro_res.get("assets", {})

                # Tabela de betas rolling
                _beta_rows = []
                for _tk, _fdata in _assets.items():
                    _row = {"Ticker": _tk}
                    for _f in _factors:
                        _fd = _fdata.get(_f, {})
                        _row[f"β {_f} (full)"]    = _fd.get("beta_full")
                        _row[f"β {_f} (63d)"]     = _fd.get("beta_rolling_63d")
                        _row[f"R² {_f}"]          = _fd.get("r2")
                    _beta_rows.append(_row)

                _beta_df = pd.DataFrame(_beta_rows).set_index("Ticker")
                st.dataframe(_beta_df.style.format("{:.4f}", na_rep="N/D")
                             .background_gradient(cmap="RdYlGn_r", subset=[c for c in _beta_df.columns if "β" in c]),
                             use_container_width=True)

                # Heatmap dos betas rolling
                _hm_data = {_f: [_assets.get(_tk, {}).get(_f, {}).get("beta_rolling_63d", np.nan)
                                  for _tk in _assets] for _f in _factors}
                _hm_df = pd.DataFrame(_hm_data, index=list(_assets.keys()))
                if not _hm_df.empty:
                    _fig_hm = px.imshow(_hm_df, text_auto=".3f", aspect="auto",
                        color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                        title="Heatmap de Beta Rolling 63d (ativo × fator macro)")
                    _fig_hm.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f2f6"))
                    st.plotly_chart(_fig_hm, use_container_width=True)
                    _dl_chart_btn("⬇️ Heatmap Macro (HTML)", _fig_hm, "macro_heatmap", "dl_macro_hm")

                _mac_buf = BytesIO()
                _beta_df.reset_index().to_excel(_mac_buf, index=False, engine="xlsxwriter")
                _mac_buf.seek(0)
                st.download_button("⬇️ Betas Macro (Excel)", _mac_buf,
                    f"macro_sensitivity_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_macro_excel")

        # ── AgentScenario ────────────────────────────────────────────────────
        with tab_scenario:
            st.markdown("### ⚡ Stress Test — Análise de Cenários")
            st.caption("Impacto estimado no portfólio sob choques de SELIC, IPCA e câmbio.")
            _scen_res = getattr(getattr(_orch, "agent_scenario", None), "scenario_results", {}) or {}

            if not _scen_res or not _scen_res.get("scenarios"):
                st.info("Execute a análise com **AgentScenario** para ver o stress test.")
            else:
                _scenarios = _scen_res["scenarios"]
                _port_val  = _scen_res.get("portfolio_value", 100_000)

                # Cards de resumo por cenário
                _scol = st.columns(len(_scenarios))
                _sev_color = {"SEVERO": "#e74c3c", "MODERADO": "#e67e22",
                              "LEVE": "#f1c40f", "POSITIVO": "#2ecc71"}
                for _si, (_sn, _sd) in enumerate(_scenarios.items()):
                    with _scol[_si]:
                        _col = _sev_color.get(_sd["severity"], "#aaa")
                        _pct = _sd["portfolio_impact_pct"]
                        _brl = _sd["portfolio_impact_brl"]
                        st.markdown(
                            f'<div style="border:2px solid {_col};border-radius:8px;'
                            f'padding:10px;text-align:center;margin:4px 0">'
                            f'<b>{_sn}</b><br/>'
                            f'<span style="font-size:1.4rem;font-weight:bold;color:{_col}">'
                            f'{_pct:+.1%}</span><br/>'
                            f'<small>R$ {_brl:+,.0f}</small><br/>'
                            f'<small style="color:{_col}">{_sd["severity"]}</small>'
                            f'</div>',
                            unsafe_allow_html=True)

                st.markdown("---")

                # Gráfico de barras: impacto por cenário
                _sc_rows = [{"Cenário": _sn, "Impacto (%)": _sd["portfolio_impact_pct"],
                             "Severidade": _sd["severity"]}
                            for _sn, _sd in _scenarios.items()]
                _fig_sc = px.bar(pd.DataFrame(_sc_rows), x="Cenário", y="Impacto (%)",
                    color="Impacto (%)", color_continuous_scale="RdYlGn",
                    title=f"Impacto dos Cenários (portfólio R$ {_port_val:,.0f})")
                _fig_sc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                _fig_sc.update_layout(height=380, yaxis=dict(tickformat=".1%"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f2f6"))
                st.plotly_chart(_fig_sc, use_container_width=True)
                _dl_chart_btn("⬇️ Stress Test (HTML)", _fig_sc, "stress_scenarios", "dl_scenario_chart")

                # Tabela de impacto por ativo para o pior cenário
                _worst_name = min(_scenarios, key=lambda k: _scenarios[k]["portfolio_impact_pct"])
                _worst = _scenarios[_worst_name]
                st.markdown(f"#### Impacto por ativo — pior cenário: *{_worst_name}*")
                _asset_rows = [{"Ticker": _tk, "Impacto (%)": _imp}
                               for _tk, _imp in _worst["asset_impacts"].items()]
                _asset_df = pd.DataFrame(_asset_rows).sort_values("Impacto (%)").set_index("Ticker")
                st.dataframe(_asset_df.style.format({"Impacto (%)": "{:.2%}"})
                             .background_gradient(cmap="RdYlGn", subset=["Impacto (%)"]),
                             use_container_width=True)

                # Excel
                _sc_buf = BytesIO()
                with pd.ExcelWriter(_sc_buf, engine="xlsxwriter") as _sw:
                    pd.DataFrame(_sc_rows).to_excel(_sw, sheet_name="Cenários", index=False)
                    pd.DataFrame([
                        {"Ticker": _tk, "Cenário": _sn, "Impacto (%)": _sd["asset_impacts"].get(_tk, 0)}
                        for _sn, _sd in _scenarios.items()
                        for _tk in _sd["asset_impacts"]
                    ]).to_excel(_sw, sheet_name="Impacto por Ativo", index=False)
                    pd.DataFrame([
                        {"Ticker": _tk, "Fator": _f, "Beta": _b}
                        for _tk, _fb in _scen_res.get("asset_betas", {}).items()
                        for _f, _b in _fb.items()
                    ]).to_excel(_sw, sheet_name="Betas Estimados", index=False)
                _sc_buf.seek(0)
                st.download_button("⬇️ Stress Test (Excel)", _sc_buf,
                    f"stress_test_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_scenario_excel")

        # ── CVM ──────────────────────────────────────────────────────────────
        with tab_cvm:
            st.markdown("### 🗂️ CVM Dados Abertos — Séries Contábeis B3")
            st.caption("Fonte: dados.cvm.gov.br · Cobertura: 100% das empresas de capital aberto (incluindo small/mid caps).")
            _cvm_res = getattr(getattr(_orch, "agent_cvm", None), "cvm_results", {}) or {}

            if not _cvm_res:
                st.info("Execute a análise com **AgentCVM** para ver os dados contábeis via CVM.")
            else:
                _cvm_rows = []
                for _tk, _d in _cvm_res.items():
                    _cvm_rows.append({
                        "Ticker":           _tk,
                        "Receita Líq. (R$)":_d.get("receita_liquida"),
                        "EBITDA (R$)":      _d.get("ebitda"),
                        "Margem EBITDA":    _d.get("margem_ebitda"),
                        "Trend Margem":     _d.get("margem_ebitda_trend"),
                        "CAGR Receita 3a":  _d.get("cagr_receita_3a"),
                        "Dív. Líq./EBITDA": _d.get("net_debt_ebitda"),
                        "ICR":              _d.get("interest_coverage"),
                        "Liq. Corrente":    _d.get("current_ratio"),
                        "FCF (R$)":         _d.get("fcf"),
                        "FCF Yield":        _d.get("fcf_yield"),
                        "ROE":              _d.get("roe"),
                        "ROA":              _d.get("roa"),
                    })
                _cvm_df = pd.DataFrame(_cvm_rows).set_index("Ticker")
                _pct_cols = ["Margem EBITDA", "Trend Margem", "CAGR Receita 3a", "FCF Yield", "ROE", "ROA"]
                _fmt = {c: "{:.1%}" for c in _pct_cols if c in _cvm_df.columns}
                for _c in ["Dív. Líq./EBITDA", "ICR", "Liq. Corrente"]:
                    if _c in _cvm_df.columns:
                        _fmt[_c] = "{:.2f}x"
                for _c in ["Receita Líq. (R$)", "EBITDA (R$)", "FCF (R$)"]:
                    if _c in _cvm_df.columns:
                        _fmt[_c] = "R$ {:,.0f}"
                st.dataframe(_cvm_df.style.format(_fmt, na_rep="—"), use_container_width=True)

                # Gráfico de barras — CAGR Receita por ticker
                _cvm_chart_rows = [{"Ticker": _tk, "CAGR Receita 3a": _d.get("cagr_receita_3a")}
                                   for _tk, _d in _cvm_res.items() if _d.get("cagr_receita_3a") is not None]
                if _cvm_chart_rows:
                    _fig_cvm = px.bar(pd.DataFrame(_cvm_chart_rows), x="Ticker", y="CAGR Receita 3a",
                        color="CAGR Receita 3a", color_continuous_scale="RdYlGn",
                        title="CAGR de Receita 3 Anos (Fonte: CVM)")
                    _fig_cvm.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                    _fig_cvm.update_layout(height=380, yaxis=dict(tickformat=".1%"),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f0f2f6"))
                    st.plotly_chart(_fig_cvm, use_container_width=True)
                    _dl_chart_btn("⬇️ CAGR Receita (HTML)", _fig_cvm, "cvm_cagr", "dl_cvm_chart")

                # Excel
                _cvm_buf = BytesIO()
                with pd.ExcelWriter(_cvm_buf, engine="xlsxwriter") as _cw:
                    _cvm_df.to_excel(_cw, sheet_name="CVM Métricas")
                _cvm_buf.seek(0)
                st.download_button("⬇️ CVM Dados (Excel)", _cvm_buf,
                    f"cvm_dados_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_cvm_excel")

        # ── Screener ─────────────────────────────────────────────────────────
        with tab_screener:
            st.markdown("### 🏆 Screener — Ranking Composto de Ativos")
            st.caption("Score 0–100 combinando Fundamentals (40%), Crédito (40%) e Dividendos (20%). Requer AgentFundamental + AgentCredit + AgentDividend.")
            _sc_res = getattr(getattr(_orch, "agent_screener", None), "screener_results", {}) or {}

            if not _sc_res:
                st.info("Execute a análise com **AgentScreener** (e AgentFundamental, AgentCredit, AgentDividend) para ver o ranking.")
            else:
                _sc_list = sorted(_sc_res.values(), key=lambda x: x["composite_score"], reverse=True)

                # Cards top-3
                _top3 = _sc_list[:3]
                _rec_color = {"COMPRA": "#2ecc71", "NEUTRO": "#f1c40f", "EVITAR": "#e74c3c"}
                if _top3:
                    st.markdown("#### Top 3")
                    _tcols = st.columns(len(_top3))
                    for _ti, _ts in enumerate(_top3):
                        with _tcols[_ti]:
                            _rc = _rec_color.get(_ts["recommendation"], "#aaa")
                            st.markdown(
                                f'<div style="border:2px solid {_rc};border-radius:8px;'
                                f'padding:12px;text-align:center">'
                                f'<b style="font-size:1.1rem">{_ts["ticker"]}</b><br/>'
                                f'<small>{_ts.get("name","")}</small><br/>'
                                f'<span style="font-size:1.8rem;font-weight:bold;color:{_rc}">'
                                f'{_ts["composite_score"]:.0f}</span><br/>'
                                f'<span style="color:{_rc};font-weight:bold">{_ts["recommendation"]}</span>'
                                f'</div>', unsafe_allow_html=True)

                st.markdown("---")

                # Tabela completa
                _sc_rows_display = []
                for _s in _sc_list:
                    _sc_rows_display.append({
                        "Ticker":       _s["ticker"],
                        "Nome":         _s.get("name", ""),
                        "Setor":        _s.get("sector", "N/D"),
                        "Score":        _s["composite_score"],
                        "Recomendação": _s["recommendation"],
                        "Fund.":        _s["score_components"].get("fundamental"),
                        "Crédito":      _s["score_components"].get("credit"),
                        "Dividendo":    _s["score_components"].get("dividend"),
                        "P/E":          _s.get("pe_ratio"),
                        "EV/EBITDA":    _s.get("ev_ebitda"),
                        "Margem EBITDA":_s.get("ebitda_margin"),
                        "Cresc. Receita":_s.get("revenue_growth_yoy"),
                        "DY":           _s.get("dividend_yield"),
                        "Credit Rating":_s.get("credit_rating", "N/D"),
                    })
                _sc_df = pd.DataFrame(_sc_rows_display).set_index("Ticker")
                _sc_fmt = {
                    "Score": "{:.0f}", "Fund.": "{:.0f}", "Crédito": "{:.0f}", "Dividendo": "{:.0f}",
                    "P/E": "{:.1f}x", "EV/EBITDA": "{:.1f}x",
                    "Margem EBITDA": "{:.1%}", "Cresc. Receita": "{:.1%}", "DY": "{:.1%}",
                }
                def _color_rec(val):
                    return f"color: {_rec_color.get(val, '#aaa')};font-weight:bold"
                st.dataframe(
                    _sc_df.style.format(_sc_fmt, na_rep="—").map(_color_rec, subset=["Recomendação"]),
                    use_container_width=True)

                # Gráfico de barras por score
                _fig_sc2 = px.bar(
                    pd.DataFrame(_sc_rows_display), x="Ticker", y="Score",
                    color="Recomendação",
                    color_discrete_map={"COMPRA": "#2ecc71", "NEUTRO": "#f1c40f", "EVITAR": "#e74c3c"},
                    title="Ranking de Ativos — Score Composto")
                _fig_sc2.add_hline(y=65, line_dash="dash", line_color="#2ecc71", annotation_text="Compra (65)")
                _fig_sc2.add_hline(y=45, line_dash="dash", line_color="#e74c3c", annotation_text="Evitar (<45)")
                _fig_sc2.update_layout(height=400, yaxis=dict(range=[0, 100]),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f2f6"))
                st.plotly_chart(_fig_sc2, use_container_width=True)
                _dl_chart_btn("⬇️ Screener Chart (HTML)", _fig_sc2, "screener_ranking", "dl_screener_chart")

                # Excel
                _sc_buf2 = BytesIO()
                with pd.ExcelWriter(_sc_buf2, engine="xlsxwriter") as _sw2:
                    _sc_df.to_excel(_sw2, sheet_name="Ranking")
                _sc_buf2.seek(0)
                st.download_button("⬇️ Screener (Excel)", _sc_buf2,
                    f"screener_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_screener_excel")

    else:
        st.info("""
        **👆 Execute a análise multiagente para ver os resultados**

        O sistema irá analisar:
        - Volatilidade e correlações em tempo real
        - Agrupamentos de ativos por similaridade
        - Anomalias e padrões usando machine learning
        - Previsões com rede neural MLP temporal (proxy LSTM)
        - Autoencoder para detecção de anomalias nos retornos
        - Alertas prioritários inteligentes
        """)

with tab3:
    # Análise de Risco (existente)
    st.markdown('<div class="section-header">📊 Análise Detalhada de Risco</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Métricas por Ativo")
        
        def calculate_asset_metrics(returns_series, risk_free):
            annual_return = returns_series.mean() * 252
            annual_vol = returns_series.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
            
            # Drawdown
            cumulative = (1 + returns_series).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # VaR e CVaR
            var = np.percentile(returns_series, (1 - confidence_level) * 100)
            cvar = returns_series[returns_series <= var].mean()
            
            return [annual_return, annual_vol, sharpe, max_drawdown, var, cvar]
        
        metrics_data = []
        for asset in returns_filtered.columns:
            metrics = calculate_asset_metrics(returns_filtered[asset].dropna(), risk_free_rate)
            metrics_data.append([asset] + metrics)
        
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=['Ativo', 'Retorno Anual', 'Volatilidade', 'Sharpe', 'Max DD', f'VaR {confidence_level:.0%}', f'CVaR {confidence_level:.0%}']
        ).set_index('Ativo')
        
        st.dataframe(metrics_df.style.format({
            'Retorno Anual': '{:.2%}',
            'Volatilidade': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max DD': '{:.2%}',
            f'VaR {confidence_level:.0%}': '{:.2%}',
            f'CVaR {confidence_level:.0%}': '{:.2%}'
        }))

        # Excel: métricas por ativo + correlação
        _met_buf = BytesIO()
        with pd.ExcelWriter(_met_buf, engine="xlsxwriter") as _w:
            metrics_df.reset_index().to_excel(_w, sheet_name="Métricas por Ativo", index=False)
            returns_filtered.corr().to_excel(_w, sheet_name="Correlação")
            returns_filtered.describe().to_excel(_w, sheet_name="Retornos Descritivos")
        _met_buf.seek(0)
        st.download_button(
            "⬇️ Métricas por Ativo (Excel)",
            data=_met_buf,
            file_name=f"metricas_ativos_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_asset_metrics",
            use_container_width=True,
        )

    with col2:
        st.subheader("🔄 Matriz de Correlação")
        corr_matrix = returns_filtered.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Correlação entre Ativos'
        )
        fig_corr.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f2f6', family='EB Garamond')
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        _dl_chart_btn("⬇️ Correlação (HTML)", fig_corr, "correlacao", "dl_corr")

    # ✅ GRÁFICO DE DRAWDOWN READICIONADO
    st.subheader("📉 Análise de Drawdown do Portfólio")
    
    # Calcular drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown_series = (cumulative_returns - rolling_max) / rolling_max
    
    # Gráfico de drawdown
    fig_drawdown = go.Figure()
    
    # Área do drawdown
    fig_drawdown.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        fill='tozeroy',
        fillcolor='rgba(255, 71, 87, 0.3)',  # Vermelho transparente
        line=dict(color='#ff4757', width=2),
        name='Drawdown',
        hovertemplate='<b>Drawdown</b>: %{y:.2%}<extra></extra>'
    ))
    
    # Linha zero
    fig_drawdown.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="white",
        opacity=0.5
    )
    
    # Máximo drawdown
    max_drawdown = drawdown_series.min()
    max_drawdown_date = drawdown_series.idxmin()
    
    # Adicionar anotação do máximo drawdown
    fig_drawdown.add_trace(go.Scatter(
        x=[max_drawdown_date],
        y=[max_drawdown],
        mode='markers+text',
        marker=dict(size=12, color='#ff4757'),
        text=[f'Max DD: {max_drawdown:.2%}'],
        textposition="top center",
        name='Máximo Drawdown',
        hovertemplate=f'<b>Máximo Drawdown</b><br>Data: {max_drawdown_date.strftime("%Y-%m-%d")}<br>Valor: {max_drawdown:.2%}<extra></extra>'
    ))
    
    fig_drawdown.update_layout(
        height=400,
        title=f"Drawdown do Portfólio (Máximo: {max_drawdown:.2%})",
        yaxis_tickformat='.1%',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f0f2f6', family='EB Garamond'),
        xaxis=dict(gridcolor='#333', title='Data'),
        yaxis=dict(gridcolor='#333', title='Drawdown'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_drawdown, use_container_width=True)
    _dl_chart_btn("⬇️ Drawdown (HTML)", fig_drawdown, "drawdown", "dl_drawdown")

    # Métricas de drawdown
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📉 Máximo Drawdown", f"{max_drawdown:.2%}")
    
    with col2:
        current_drawdown = drawdown_series.iloc[-1] if len(drawdown_series) > 0 else 0
        st.metric("📊 Drawdown Atual", f"{current_drawdown:.2%}")
    
    with col3:
        # Duração atual do drawdown
        if current_drawdown < 0:
            # Encontrar início do drawdown atual
            current_drawdown_start = drawdown_series[drawdown_series == 0].last_valid_index()
            if current_drawdown_start is not None:
                duration = (drawdown_series.index[-1] - current_drawdown_start).days
            else:
                duration = len(drawdown_series)
            st.metric("⏱️ Duração Atual", f"{duration} dias")
        else:
            st.metric("⏱️ Duração Atual", "0 dias")
    
    with col4:
        # Número de drawdowns > 5%
        significant_drawdowns = (drawdown_series < -0.05).sum()
        st.metric("⚠️ Drawdowns > 5%", f"{significant_drawdowns}")

with tab4:
    # ✅ CORREÇÃO: Simulações Avançadas com funções corrigidas
    st.markdown('<div class="section-header">🎯 Simulações Avançadas de Monte Carlo</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simulation-card">
    <h3>🔄 Algoritmos de Simulação</h3>
    <ul>
    <li><b>Monte Carlo Clássico</b>: Simulação básica com distribuição normal</li>
    <li><b>Bootstrapping</b>: Reamostragem dos dados históricos</li>
    <li><b>Merton Jump</b>: Inclui saltos para eventos extremos</li>
    <li><b>GARCH</b>: Volatilidade variável no tempo</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Configurações de simulação
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_simulations = st.selectbox(
            "Número de Simulações:",
            [500, 1000, 2000, 5000],
            index=1,
            key="num_simulations_select_tab4"  # ✅ KEY ÚNICA
        )
    
    with col2:
        time_horizon = st.selectbox(
            "Horizonte Temporal (dias):",
            [30, 90, 180, 252, 504],
            index=3,
            key="time_horizon_select_tab4"  # ✅ KEY ÚNICA
        )
    
    with col3:
        initial_investment = st.number_input(
            "Investimento Inicial (R$):",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            key="initial_investment_input_tab4"  # ✅ KEY ÚNICA
        )
    
    algorithm = st.selectbox(
        "Selecione o algoritmo de simulação:",
        [
            "Monte Carlo Clássico",
            "Bootstrapping", 
            "Merton Jump Diffusion",
            "GARCH"
        ],
        key="algorithm_select_tab4"  # ✅ KEY ÚNICA
    )
    
    # ✅ CORREÇÃO: Botão com key única e controle de estado
    if st.button("🎲 Executar Simulação", type="primary", key="run_simulation_button_tab4"):
        with st.spinner(f"Executando {algorithm}..."):
            result = run_simulation_corrected(
                algorithm=algorithm,
                returns_df=returns_filtered,
                initial_investment=initial_investment,
                time_horizon=time_horizon,
                num_simulations=num_simulations,
                weights=get_current_weights(selected_assets)
            )
            if result:
                existing = st.session_state.get('current_simulation_results') or {}
                existing[algorithm] = result
                st.session_state.current_simulation_results = existing
                st.success(f"✅ {algorithm} concluído!")
            else:
                st.warning("⚠️ Simulação não retornou resultados")
    
    # ✅ CORREÇÃO: Exibir resultados se existirem na sessão
    if 'current_simulation_results' in st.session_state and st.session_state.current_simulation_results:
        st.markdown("---")
        display_simulation_results(st.session_state.current_simulation_results, initial_investment)
    
    # ✅ CORREÇÃO: Simulação multi-método robusta
    st.markdown("---")
    st.markdown('<div class="section-header">🔄 Simulação com Múltiplos Métodos</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Executar Todas as Simulações", key="run_all_simulations"):
        with st.spinner("Executando todas as simulações..."):
            try:
                # Lista de algoritmos para executar
                algorithms = [
                    "Monte Carlo Clássico",
                    "Bootstrapping", 
                    "Merton Jump Diffusion",
                    "GARCH"
                ]
                
                simulation_results = {}
                
                # Executar cada algoritmo
                for algorithm in algorithms:
                    with st.spinner(f"Executando {algorithm}..."):
                        # Tentar método principal
                        result = run_simulation_corrected(
                            algorithm=algorithm,
                            returns_df=returns_filtered,
                            initial_investment=initial_investment,
                            time_horizon=time_horizon,
                            num_simulations=num_simulations,
                            weights=get_current_weights(selected_assets)
                        )
                        
                        # Se falhar, usar fallback
                        if not result:
                            st.warning(f"⚠️ {algorithm} falhou, usando fallback...")
                            result = run_simulation_simple_fallback(
                                algorithm=algorithm,
                                returns_df=returns_filtered,
                                initial_investment=initial_investment,
                                time_horizon=time_horizon,
                                num_simulations=num_simulations
                            )
                        
                        if result:
                            simulation_results[algorithm] = result
                            st.success(f"✅ {algorithm} concluído")
                        else:
                            st.error(f"❌ {algorithm} não retornou resultados")
                
                # Armazenar resultados
                if simulation_results:
                    st.session_state.current_simulation_results = simulation_results
                    st.session_state.last_simulation_results = simulation_results
                    st.success(f"✅ {len(simulation_results)} métodos executados com sucesso!")
                    display_simulation_results(simulation_results, initial_investment)
                else:
                    st.error("❌ Nenhuma simulação foi bem-sucedida")
                    
            except Exception as e:
                st.error(f"❌ Erro nas simulações multi-método: {str(e)}")
                import traceback
                st.code(f"Detalhes: {traceback.format_exc()}")

with tab5:
    # Analytics Avançados
    st.markdown('<div class="section-header">📊 Analytics e Visualizações</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Retornos vs Volatilidade")
        annual_returns = returns_filtered.mean() * 252
        annual_volatility = returns_filtered.std() * np.sqrt(252)
        
        fig_scatter = px.scatter(
            x=annual_volatility,
            y=annual_returns,
            text=returns_filtered.columns,
            title="Retorno vs Risco por Ativo"
        )
        
        fig_scatter.update_traces(
            marker=dict(size=20, opacity=0.7, color='#00d4aa'),
            textposition='top center'
        )
        
        fig_scatter.update_layout(
            height=400,
            xaxis_title="Volatilidade Anual",
            yaxis_title="Retorno Anual",
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f2f6'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("📉 Análise de Drawdown")
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_portfolio.expanding().max()
        drawdown_series = (cumulative_portfolio - rolling_max) / rolling_max
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values,
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.3)',
            line=dict(color='#00d4aa', width=2),
            name='Drawdown'
        ))
        fig_dd.update_layout(
            height=400,
            title="Drawdown do Portfólio",
            yaxis_tickformat='.1%',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f2f6'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with tab6:
    # ── Configuração de Alertas de Risco ────────────────────────────────────
    st.markdown('<div class="section-header">🔔 Alertas de Risco — Configuração</div>',
                unsafe_allow_html=True)

    _cfg = st.session_state.alert_config
    _p   = _cfg.get("portfolio", {})

    st.markdown("#### Limites do Portfólio")
    st.caption("Alertas são avaliados em tempo real toda vez que os dados são carregados ou atualizados.")

    _c1, _c2 = st.columns(2)

    with _c1:
        new_var = st.slider(
            "VaR 95% diário máximo (%)",
            min_value=1.0, max_value=20.0,
            value=float(_p.get("var_threshold", 0.05)) * 100,
            step=0.5,
            help="Alerta quando o VaR 95% diário do portfólio superar este valor.",
            key="alert_var_slider"
        ) / 100

        new_vol = st.slider(
            "Volatilidade anual máxima (%)",
            min_value=5.0, max_value=100.0,
            value=float(_p.get("volatility_threshold", 0.30)) * 100,
            step=1.0,
            help="Alerta quando a volatilidade anualizada do portfólio superar este valor.",
            key="alert_vol_slider"
        ) / 100

    with _c2:
        new_dd = st.slider(
            "Drawdown atual máximo (%)",
            min_value=2.0, max_value=50.0,
            value=float(_p.get("drawdown_threshold", 0.15)) * 100,
            step=1.0,
            help="Alerta quando o drawdown corrente do portfólio superar este valor.",
            key="alert_dd_slider"
        ) / 100

        new_corr = st.slider(
            "Correlação média máxima",
            min_value=0.30, max_value=1.00,
            value=float(_p.get("correlation_threshold", 0.80)),
            step=0.05,
            help="Acima deste valor, a diversificação é considerada comprometida.",
            key="alert_corr_slider"
        )

    # ── Overrides por ativo ──────────────────────────────────────────────────
    st.markdown("#### Limites por Ativo (opcional)")
    st.caption("Deixe em branco para usar os limites do portfólio como padrão.")

    _asset_overrides = dict(_cfg.get("assets", {}))
    _loaded_assets   = returns_filtered.columns.tolist() if not returns_filtered.empty else []

    if _loaded_assets:
        _sel_asset = st.selectbox("Configurar ativo:", ["— selecione —"] + _loaded_assets,
                                  key="alert_asset_select")
        if _sel_asset != "— selecione —":
            _ao = _asset_overrides.get(_sel_asset, {})
            _ac1, _ac2 = st.columns(2)
            with _ac1:
                _a_var = st.number_input(
                    f"VaR máximo — {_sel_asset} (%)",
                    min_value=0.5, max_value=30.0,
                    value=float(_ao.get("var_threshold", new_var)) * 100,
                    step=0.5, key="alert_asset_var"
                ) / 100
            with _ac2:
                _a_vol = st.number_input(
                    f"Volatilidade máxima — {_sel_asset} (%)",
                    min_value=5.0, max_value=150.0,
                    value=float(_ao.get("volatility_threshold", new_vol)) * 100,
                    step=1.0, key="alert_asset_vol"
                ) / 100

            _col_set, _col_rem = st.columns(2)
            with _col_set:
                if st.button(f"💾 Salvar para {_sel_asset}", use_container_width=True,
                             key="alert_asset_save"):
                    _asset_overrides[_sel_asset] = {
                        "var_threshold":        _a_var,
                        "volatility_threshold": _a_vol,
                    }
                    st.success(f"Limite personalizado salvo para {_sel_asset}.")
            with _col_rem:
                if st.button(f"🗑️ Remover override de {_sel_asset}", use_container_width=True,
                             key="alert_asset_remove"):
                    _asset_overrides.pop(_sel_asset, None)
                    st.info(f"Override removido — {_sel_asset} usará os limites do portfólio.")
    else:
        st.info("Carregue ativos na sidebar para configurar limites individuais.")

    # Overrides ativos
    if _asset_overrides:
        with st.expander(f"Overrides ativos ({len(_asset_overrides)} ativo(s))", expanded=False):
            for _tk, _ov in _asset_overrides.items():
                st.caption(
                    f"**{_tk}**: VaR ≤ {_ov.get('var_threshold', new_var):.2%}  |  "
                    f"Vol ≤ {_ov.get('volatility_threshold', new_vol):.2%}"
                )

    # ── Salvar / Resetar ────────────────────────────────────────────────────
    st.markdown("---")
    _btn1, _btn2 = st.columns(2)

    with _btn1:
        if st.button("💾 Salvar configuração", type="primary",
                     use_container_width=True, key="alert_save_btn"):
            new_cfg = {
                "portfolio": {
                    "var_threshold":         new_var,
                    "volatility_threshold":  new_vol,
                    "drawdown_threshold":    new_dd,
                    "correlation_threshold": new_corr,
                },
                "assets": _asset_overrides,
            }
            _save_alert_cfg(new_cfg)
            st.session_state.alert_config  = new_cfg
            st.session_state.active_alerts = _evaluate_alerts(returns_filtered, new_cfg)
            st.success("✅ Configuração salva em `data/alert_config.json`.")

    with _btn2:
        if st.button("🔄 Restaurar padrões", use_container_width=True, key="alert_reset_btn"):
            from src.alerts.alert_config import DEFAULT_CONFIG
            _save_alert_cfg({"portfolio": dict(DEFAULT_CONFIG["portfolio"]), "assets": {}})
            st.session_state.alert_config  = _load_alert_cfg()
            st.session_state.active_alerts = _evaluate_alerts(returns_filtered,
                                                               st.session_state.alert_config)
            st.info("Configuração restaurada para os valores padrão.")

    # ── Preview de alertas atuais ────────────────────────────────────────────
    st.markdown("#### Preview — alertas com a configuração atual")
    _preview_cfg = {
        "portfolio": {
            "var_threshold":         new_var,
            "volatility_threshold":  new_vol,
            "drawdown_threshold":    new_dd,
            "correlation_threshold": new_corr,
        },
        "assets": _asset_overrides,
    }
    _preview_alerts = _evaluate_alerts(returns_filtered, _preview_cfg)
    if _preview_alerts:
        _sev_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        for _a in _preview_alerts:
            st.markdown(f"{_sev_icons.get(_a['severity'], '•')} {_a['message']}")
    else:
        st.success("Nenhum alerta seria disparado com estes limites.", icon="✅")

    st.markdown("---")

    # ── Configurações Técnicas (conteúdo original) ───────────────────────────
    st.markdown('<div class="section-header">⚙️ Configurações Técnicas</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Stack Tecnológico")
        
        st.markdown("""
        <div class="code-block">
        # Framework Multiagente
        🤖 Orquestrador: Sequencial (7 agentes)
        📊 Scikit-learn: Isolation Forest, K-Means, PCA
        📈 yfinance: dados em tempo real (TTL 5min)
        🎲 Simuladores: MC, Bootstrap, Merton, GARCH

        # Roadmap
        ⚡ Dask: paralelismo (futuro — > 20 ativos)
        🧠 LSTM / Autoencoder: em desenvolvimento
        🔔 Alertas persistentes: em desenvolvimento
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔧 Agentes Ativos")
        agents_info = {
            "AgentMarket": "Análise de volatilidade, correlação, regimes",
            "AgentClustering": "K-Means + PCA para agrupamento",
            "AgentML": "Isolation Forest + Random Forest", 
            "AgentLSTM": "Redes neurais LSTM temporais",
            "AgentAutoencoder": "Detecção de anomalias não supervisionada",
            "AgentAlert": "Sistema de priorização inteligente"
        }
        
        for agent, desc in agents_info.items():
            st.markdown(f"""
            <div class="metric-card">
                <strong>{agent}</strong><br/>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Status do Sistema")
        
        # Informações do sistema
        if st.session_state.orchestrator:
            system_status  = "✅ OPERACIONAL"
            exec_mode      = "Dask Paralelo" if st.session_state.orchestrator.use_dask else "Sequencial"
            exec_icon      = "✅" if st.session_state.orchestrator.use_dask else "⚡"
        else:
            system_status  = "❌ OFFLINE"
            exec_mode      = "—"
            exec_icon      = ""

        st.metric("Sistema Multiagente", system_status)
        st.metric("Modo de execução", f"{exec_icon} {exec_mode}")
        st.metric("Agentes Ativos", "7/7")
        st.caption("Dask entra automaticamente quando instalado e > 20 ativos em análise.")
        
        st.subheader("🔍 Diagnóstico")
        if st.button("🩺 Executar Diagnóstico", use_container_width=True, key="run_diagnostic"):
            try:
                # Teste simples do sistema
                test_data = prices_filtered.iloc[-100:]  # Últimos 100 dias
                test_alerts = st.session_state.orchestrator.run_analysis(test_data)
                
                if test_alerts:
                    st.success(f"✅ Sistema operacional - {len(test_alerts)} alertas de teste gerados")
                else:
                    st.warning("⚠️ Sistema operacional mas sem alertas de teste")
                    
            except Exception as e:
                st.error(f"❌ Falha no diagnóstico: {str(e)}")

# =============================================
# RODAPÉ
# =============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-family: "EB Garamond", serif;'>
    <b>🤖 Quantum Risk Analytics - Sistema Multiagente</b> | 
    Desenvolvido com Python + yfinance + Scikit-learn |
    Dados: Yahoo Finance + Banco Central do Brasil |
    🚀 7 Agentes Especializados em Operação
    </div>
    """, 
    unsafe_allow_html=True
)

# =============================================
# INICIALIZAÇÃO DE VARIÁVEIS DE SESSÃO
# =============================================

if 'last_simulation_results' not in st.session_state:
    st.session_state.last_simulation_results = None

if 'last_agent_report' not in st.session_state:
    st.session_state.last_agent_report = None

if 'last_agent_alerts' not in st.session_state:
    st.session_state.last_agent_alerts = None

if 'current_simulation_results' not in st.session_state:
    st.session_state.current_simulation_results = None	
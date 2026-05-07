import streamlit as st
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
    """Gera relatório PDF profissional - CORRIGIDA"""
    try:
        # ✅ CORREÇÃO: Verificar se existem dados para exportar
        if not simulation_results:
            st.error("❌ Nenhum dado disponível para exportação")
            return
            
        # Criar PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Cabeçalho
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'RELATÓRIO DE ANÁLISE DE RISCO - QUANTUM RISK ANALYTICS', 0, 1, 'C')
        pdf.ln(5)
        
        # Data e informações básicas
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1)
        pdf.cell(0, 10, f"Investimento Inicial: R$ {initial_investment:,.2f}", 0, 1)
        pdf.ln(10)
        
        # Métricas principais
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'MÉTRICAS PRINCIPAIS', 0, 1)
        pdf.ln(5)
        
        # ✅ CORREÇÃO: Tabela de métricas com verificação
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 10, 'Método', 1)
        pdf.cell(30, 10, 'VaR (R$)', 1)
        pdf.cell(30, 10, 'Prob. Prejuízo', 1)
        pdf.cell(40, 10, 'Retorno Esperado (R$)', 1)
        pdf.cell(30, 10, 'Simulações', 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 10)
        for method, results in simulation_results.items():
            # ✅ CORREÇÃO: Verificar se as métricas existem
            var_value = abs(results.get('var', 0))
            prob_loss = results.get('probability_loss', 0)
            expected_return = results.get('expected_return', 0)
            simulations = results.get('simulations', 0)
            
            pdf.cell(60, 10, str(method), 1)
            pdf.cell(30, 10, f"R$ {var_value:,.0f}", 1)
            pdf.cell(30, 10, f"{prob_loss:.2%}", 1)
            pdf.cell(40, 10, f"R$ {expected_return:,.0f}", 1)
            pdf.cell(30, 10, f"{simulations:,}", 1)
            pdf.ln()
        
        # Salvar PDF temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf.output(tmp_file.name)
            
            with open(tmp_file.name, 'rb') as f:
                pdf_data = f.read()
        
        # ✅ CORREÇÃO: Botão de download com key única
        st.download_button(
            label="⬇️ Baixar Relatório PDF",
            data=pdf_data,
            file_name=f"relatorio_risco_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key="pdf_export_button"  # ✅ KEY ÚNICA
        )
        
        # Limpar arquivo temporário
        os.unlink(tmp_file.name)
        
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

# =============================================
# FUNÇÕES AUXILIARES
# =============================================

def load_data():
    """Carrega os dados do portfólio"""
    try:
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        prices = pd.read_parquet('data/processed/macro_portfolio_prices.parquet')
        return returns, prices
    except:
        try:
            returns = pd.read_csv('data/processed/macro_portfolio_returns.csv', index_col=0, parse_dates=True)
            prices = pd.read_csv('data/processed/macro_portfolio_prices.csv', index_col=0, parse_dates=True)
            return returns, prices
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            # Criar dados de exemplo para demonstração
            st.info("🔄 Criando dados de exemplo para demonstração...")
            dates = pd.date_range('2020-01-01', periods=500, freq='D')
            np.random.seed(42)
            example_data = {
                'SELIC': np.random.normal(0.0003, 0.005, 500),
                'USD_BRL': np.random.normal(0.0005, 0.008, 500),
                'PETROLEO_BRENT': np.random.normal(0.0008, 0.012, 500),
                'IBOVESPA': np.random.normal(0.001, 0.015, 500),
                'CRYPTO_BTC': np.random.normal(0.002, 0.025, 500)
            }
            returns = pd.DataFrame(example_data, index=dates)
            prices = (1 + returns).cumprod() * 100
            return returns, prices

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
# CARREGAMENTO DE DADOS
# =============================================

# Carregar dados
returns, prices = load_data()

if returns is None or prices is None:
    st.error("""
    **Dados não encontrados!**
    
    Execute primeiro o pipeline completo:
    ```bash
    python src/etl/data_collector_bcb.py
    python src/metrics/risk_calculator.py
    ```
    """)
    st.stop()

# =============================================
# SIDEBAR - CONFIGURAÇÕES
# =============================================

st.sidebar.markdown('<div class="section-header">⚙️ Configurações do Sistema</div>', unsafe_allow_html=True)

# Configurações de Tech Stack
st.sidebar.markdown("**🛠️ Tech Stack Ativo:**")
col_tech1, col_tech2 = st.sidebar.columns(2)
with col_tech1:
    st.markdown('<span class="tech-badge">Dask</span>', unsafe_allow_html=True)
    st.markdown('<span class="tech-badge">TensorFlow</span>', unsafe_allow_html=True)
with col_tech2:
    st.markdown('<span class="tech-badge">Scikit-learn</span>', unsafe_allow_html=True) 
    st.markdown('<span class="tech-badge">6 Agents</span>', unsafe_allow_html=True)

# Seleção de ativos
st.sidebar.markdown("**📊 Configuração do Portfólio**")
selected_assets = st.sidebar.multiselect(
    "Selecione os ativos para análise:",
    options=returns.columns.tolist(),
    default=returns.columns.tolist()[:5] if len(returns.columns) >= 5 else returns.columns.tolist()
)

# Parâmetros de risco
st.sidebar.markdown("**🎯 Parâmetros de Risco**")
risk_free_rate = st.sidebar.slider("Taxa Livre de Risco (% a.a):", 0.0, 20.0, 11.75, 0.1) / 100
confidence_level = st.sidebar.slider("Nível de Confiança VaR:", 0.90, 0.99, 0.95, 0.01)

# Configurações avançadas
st.sidebar.markdown("**🔧 Configurações Avançadas**")
use_dask = st.sidebar.checkbox("Usar Dask (Processamento Paralelo)", value=True)
use_tensorflow = st.sidebar.checkbox("Usar TensorFlow (Deep Learning)", value=True)
enable_clustering = st.sidebar.checkbox("Ativar Clustering de Ativos", value=True)
enable_ml_detection = st.sidebar.checkbox("Ativar Detecção ML", value=True)

# Filtro de período
st.sidebar.markdown("**📅 Período de Análise**")
min_date = returns.index.min().date()
max_date = returns.index.max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Data Início", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("Data Fim", value=max_date, min_value=min_date, max_value=max_date)

# =============================================
# FILTRAGEM DE DADOS
# =============================================

# Filtrar dados
if selected_assets:
    returns_filtered = returns[selected_assets].loc[str(start_date):str(end_date)]
    prices_filtered = prices[selected_assets].loc[str(start_date):str(end_date)]
else:
    returns_filtered = returns.loc[str(start_date):str(end_date)]
    prices_filtered = prices.loc[str(start_date):str(end_date)]

# Calcular retorno do portfólio
weights = get_current_weights(selected_assets)
if weights:
    weight_sum = sum(weights.values())
    normalized_weights = {k: v/weight_sum for k, v in weights.items()}
    portfolio_returns = (returns_filtered * pd.Series(normalized_weights)).sum(axis=1)
else:
    portfolio_returns = returns_filtered.mean(axis=1)

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
    # Dashboard Principal
    st.markdown('<div class="section-header">📈 Visão Geral do Portfólio</div>', unsafe_allow_html=True)
    
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
    
    # Status dos agentes
    display_agent_status()
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Evolução dos Preços")
        fig_prices = go.Figure()
        for asset in prices_filtered.columns:
            fig_prices.add_trace(go.Scatter(
                x=prices_filtered.index,
                y=prices_filtered[asset],
                name=asset,
                line=dict(width=2)
            ))
        fig_prices.update_layout(
            height=400, 
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f2f6'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig_prices, use_container_width=True)
    
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

with tab2:
    # Sistema Multiagente
    st.markdown('<div class="section-header">🤖 Sistema Multiagente Quantitativo</div>', unsafe_allow_html=True)
    
    # Controles de execução
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info("""
        **🎯 Sistema com 6 Agentes Especializados:**
        - AgentMarket: Análise de mercado em tempo real
        - AgentClustering: Agrupamento inteligente de ativos  
        - AgentML: Modelos de machine learning
        - AgentLSTM: Redes neurais para previsão temporal
        - AgentAutoencoder: Detecção avançada de anomalias
        - AgentAlert: Sistema de priorização de alertas
        """)
    
    with col2:
        # ✅ CORREÇÃO: Botão que mantém na mesma aba
        if st.button("🚀 Executar Análise Completa", type="primary", use_container_width=True, key="run_agents_main"):
            if st.session_state.orchestrator:
                # Manter na mesma aba
                st.session_state.active_tab = "🤖 Sistema Multiagente"
                
                with st.spinner("🤖 Executando análise multiagente coordenada..."):
                    try:
                        progress_bar = st.progress(0)
                        
                        # Simular progresso
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Executar análise
                        alerts = st.session_state.orchestrator.run_analysis(prices_filtered)
                        report = st.session_state.orchestrator.generate_report(alerts)
                        
                        # Armazenar resultados
                        st.session_state.last_agent_report = report
                        st.session_state.last_agent_alerts = alerts
                        
                        st.success("✅ Análise multiagente concluída com sucesso!")
                        
                        # ✅ CORREÇÃO: Forçar rerun para atualizar a interface
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Erro na execução: {str(e)}")
            else:
                st.error("❌ Sistema multiagente não inicializado!")
    
    with col3:
        if st.button("🔄 Reinicializar Sistema", use_container_width=True, key="restart_agents"):
            with st.spinner("Reinicializando sistema..."):
                st.session_state.orchestrator = initialize_multiagent_system()
                if 'last_agent_report' in st.session_state:
                    del st.session_state.last_agent_report
                if 'last_agent_alerts' in st.session_state:
                    del st.session_state.last_agent_alerts
                st.rerun()
    
    # Exibir resultados se disponíveis
    if hasattr(st.session_state, 'last_agent_report'):
        # Abas para diferentes visualizações de alertas
        tab_alertas, tab_analytics = st.tabs(["🚨 Alertas Detalhados", "📊 Analytics"])
        
        with tab_alertas:
            display_all_alerts_with_filters(st.session_state.last_agent_report)
        
        with tab_analytics:
            if hasattr(st.session_state, 'last_agent_alerts'):
                display_agent_analytics(st.session_state.last_agent_alerts)
    else:
        st.info("""
        **👆 Execute a análise multiagente para ver os resultados**
        
        O sistema irá analisar:
        - Volatilidade e correlações em tempo real
        - Agrupamentos de ativos por similaridade  
        - Anomalias e padrões usando machine learning
        - Previsões com redes neurais LSTM
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
        # Manter na mesma aba
        st.session_state.active_tab = "🎯 Simulações"
        
        with st.spinner(f"Executando {algorithm}..."):
            try:
                # Importar e executar simulação
                from src.agents.agent_simulation import AgentSimulation

                # Configurar parâmetros do portfólio
                portfolio_config = {
                    'value': initial_investment,
                    'confidence_level': 0.95,
                    'time_horizon': time_horizon,
                    'weights': get_current_weights(selected_assets),
                    'num_simulations': num_simulations
                }

                # Executar agente de simulação
                agent_sim = AgentSimulation()
                simulation_alerts = agent_sim.process_data(prices_filtered, portfolio_config)
                
                # ✅ CORREÇÃO: Armazenar resultados na sessão
                if hasattr(agent_sim, 'simulation_results') and agent_sim.simulation_results:
                    st.session_state.current_simulation_results = agent_sim.simulation_results
                    st.success(f"✅ Simulação {algorithm} concluída com sucesso!")
                    
                    # ✅ CORREÇÃO: Forçar atualização da interface
                    st.rerun()
                else:
                    st.warning("⚠️ Simulação executada, mas nenhum resultado foi gerado")
                    
            except Exception as e:
                st.error(f"❌ Erro na simulação: {str(e)}")
    
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
                    st.session_state.last_simulation_results = simulation_results
                    st.success(f"✅ {len(simulation_results)} métodos executados com sucesso!")
                    
                    # Exibir resultados consolidados
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
    # Configurações Técnicas
    st.markdown('<div class="section-header">⚙️ Configurações Técnicas</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Stack Tecnológico")
        
        st.markdown("""
        <div class="code-block">
        # Framework Multiagente
        🤖 Dask Orchestrator: Ativo
        🧠 TensorFlow: Ativo
        📊 Scikit-learn: Ativo
        🔍 6 Agentes Especializados
        
        # Processamento
        ⚡ Paralelismo: Dask Distributed
        🎯 ML: Ensemble + Deep Learning
        📈 Análise: Tempo Real + Histórica
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
            system_status = "✅ OPERACIONAL"
            dask_status = "✅ ATIVO" if st.session_state.orchestrator.use_dask else "❌ INATIVO"
            tf_status = "✅ ATIVO" if st.session_state.orchestrator.use_tensorflow else "❌ INATIVO"
        else:
            system_status = "❌ OFFLINE"
            dask_status = "❌ OFFLINE"
            tf_status = "❌ OFFLINE"
        
        st.metric("Sistema Multiagente", system_status)
        st.metric("Framework Dask", dask_status)
        st.metric("TensorFlow", tf_status)
        st.metric("Agentes Ativos", "6/6")
        
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
    Desenvolvido com Python + Dask + TensorFlow | 
    Dados: Banco Central do Brasil | 
    🚀 6 Agentes Especializados em Operação
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
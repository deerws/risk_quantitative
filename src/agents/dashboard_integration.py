# src/agents/dashboard_integration.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

def get_current_weights():
    """Função auxiliar para obter pesos atuais"""
    return {'SELIC': 0.25, 'USD_BRL': 0.25, 'PETROLEO_BRENT': 0.25, 'IBOVESPA': 0.25}

def display_alert_report(report: Dict[str, Any]):
    """Exibe relatório de alertas no dashboard"""
    
    st.subheader("📊 Relatório de Alertas Consolidado")
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Alertas", report['total_alerts'])
    
    with col2:
        critical = report['severity_breakdown'].get('critical', 0)
        high = report['severity_breakdown'].get('high', 0)
        st.metric("Alertas Críticos/Altos", critical + high)
    
    with col3:
        st.metric("Ativos com Alertas", len(report['asset_alerts']))
    
    with col4:
        st.metric("Orquestrador", report.get('orchestrator', 'N/A'))
    
    # Alertas detalhados
    st.subheader("🚨 Alertas Prioritários")
    
    if not report['critical_alerts']:
        st.success("✅ Nenhum alerta crítico detectado")
        st.info("💡 Dica: Tente com dados mais voláteis ou aumente os thresholds")
        return
    
    for alert in report['critical_alerts'][:8]:  # Mostrar mais alertas
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{alert['agent_id']}** - {alert['message']}")
                if alert['data']:
                    with st.expander("Detalhes técnicos"):
                        st.json(alert['data'], expanded=False)
            
            with col2:
                severity = alert['severity']
                color = {
                    'critical': 'red',
                    'high': 'orange', 
                    'medium': 'yellow',
                    'low': 'green'
                }.get(severity, 'gray')
                
                st.markdown(
                    f"<div style='color: {color}; font-weight: bold; font-size: 1.1em; padding: 0.5rem; border: 1px solid {color}; border-radius: 5px; text-align: center;'>"
                    f"{severity.upper()}</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")

def display_agent_dashboard(orchestrator, market_data: pd.DataFrame, portfolio_returns: pd.DataFrame):
    """Adiciona seção de multiagentes ao dashboard existente"""
    
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Sistema Multiagente - Alertas em Tempo Real</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Monitoramento de Agentes")
        
        agent_status = {
            "AgentMarket": "🟢 Ativo - Análise de Volatilidade/Correlação",
            "AgentClustering": "🟢 Ativo - Clustering de Ativos", 
            "AgentML": "🟢 Ativo - Detecção de Anomalias",
            "AgentAlert": "🟢 Ativo - Priorização"
        }
        
        for agent, status in agent_status.items():
            st.write(f"**{agent}**: {status}")
    
    with col2:
        use_dask = st.checkbox("Usar Dask (Paralelo)", value=False)
        st.info("Dask para processamento paralelo (experimental)")
    
    with col3:
        if st.button("🚀 Executar Análise Multiagente", type="primary", use_container_width=True):
            with st.spinner("Executando análise coordenada entre agentes..."):
                try:
                    # Configurar orquestrador
                    orchestrator.use_dask = use_dask
                    
                    # Executar orquestrador
                    alerts = orchestrator.run_analysis(market_data)
                    report = orchestrator.generate_report(alerts)
                    
                    # Armazenar no session state para persistência
                    st.session_state.last_agent_report = report
                    
                    # Exibir resultados
                    display_alert_report(report)
                    
                except Exception as e:
                    st.error(f"❌ Erro na análise multiagente: {str(e)}")
                    st.info("💡 Dica: Tente desativar o Dask se houver problemas")
    
    # Mostrar último relatório se existir
    if hasattr(st.session_state, 'last_agent_report'):
        if st.button("📋 Mostrar Último Relatório"):
            display_alert_report(st.session_state.last_agent_report)
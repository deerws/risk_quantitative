# scripts/test_quant_agents.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_quant_test_data():
    """Cria dados de teste com características financeiras realistas"""
    print("📊 Gerando dados quantitativos de teste...")
    
    dates = pd.date_range('2020-01-01', periods=800, freq='D')
    np.random.seed(42)
    
    # Criar 3 clusters de ativos com características diferentes
    assets = {
        'cluster_1': ['SELIC', 'CDI', 'TESOURO_1Y'],  # Renda fixa - baixa vol
        'cluster_2': ['USD_BRL', 'EUR_BRL', 'GBP_BRL'],  # Forex - média vol  
        'cluster_3': ['PETROLEO_BRENT', 'IBOVESPA', 'SMALL_CAPS'],  # Commodities/Ações - alta vol
        'cluster_4': ['CRYPTO_BTC', 'CRYPTO_ETH', 'TECH_STOCKS']  # Alta vol/risco
    }
    
    all_data = {}
    
    # Cluster 1: Baixa volatilidade (1-10%)
    for asset in assets['cluster_1']:
        vol = np.random.uniform(0.01, 0.10) / np.sqrt(252)
        returns = np.random.normal(0.0002, vol, 800)
        all_data[asset] = 100 * (1 + returns).cumprod()
    
    # Cluster 2: Média volatilidade (10-25%)
    for asset in assets['cluster_2']:
        vol = np.random.uniform(0.10, 0.25) / np.sqrt(252)
        returns = np.random.normal(0.0003, vol, 800)
        all_data[asset] = 100 * (1 + returns).cumprod()
    
    # Cluster 3: Alta volatilidade (25-50%)
    for asset in assets['cluster_3']:
        vol = np.random.uniform(0.25, 0.50) / np.sqrt(252)
        # Adicionar alguns outliers para testar anomalias
        returns = np.random.normal(0.0005, vol, 800)
        outlier_indices = np.random.choice(800, 10, replace=False)
        returns[outlier_indices] *= 3  # Criar outliers
        all_data[asset] = 100 * (1 + returns).cumprod()
    
    # Cluster 4: Muito alta volatilidade (50-100%)
    for asset in assets['cluster_4']:
        vol = np.random.uniform(0.50, 1.00) / np.sqrt(252)
        returns = np.random.normal(0.001, vol, 800)
        all_data[asset] = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame(all_data, index=dates)
    print(f"   ✅ Dados criados: {df.shape} - {len(assets)} clusters simulados")
    return df

def test_quant_agents():
    """Testa os agentes quantitativos"""
    print("🧪 INICIANDO TESTE DOS AGENTES QUANTITATIVOS")
    print("=" * 60)
    
    try:
        from src.agents.dask_orchestrator import DaskMultiAgentOrchestrator
        
        # Criar dados de teste
        test_data = create_quant_test_data()
        print(f"   Período: {test_data.index[0]} a {test_data.index[-1]}")
        print(f"   Ativos: {list(test_data.columns)}")
        
        # Inicializar orquestrador
        print("🚀 Inicializando orquestrador Dask...")
        orchestrator = DaskMultiAgentOrchestrator(use_dask=True)
        
        # Executar análise
        print("🔍 Executando análise quantitativa...")
        start_time = time.time()
        alerts = orchestrator.run_analysis(test_data)
        end_time = time.time()
        
        report = orchestrator.generate_report(alerts)
        
        # Exibir resultados
        print("\n📈 RESULTADOS DA ANÁLISE QUANTITATIVA:")
        print("=" * 60)
        print(f"⏱️  Tempo de execução: {end_time - start_time:.2f} segundos")
        print(f"🏗️  Orquestrador: {report['orchestrator']}")
        print(f"📊 Total de alertas: {report['total_alerts']}")
        print(f"📝 Resumo: {report['summary']}")
        
        # Detalhes por severidade
        print("\n📊 Distribuição de alertas:")
        for severity, count in report['severity_breakdown'].items():
            print(f"   {severity.upper()}: {count}")
        
        # Alertas por tipo de agente
        print(f"\n🤖 Alertas por Agente:")
        print(f"   AgentClustering: {len(report['cluster_alerts'])}")
        print(f"   AgentML: {len(report['ml_alerts'])}")
        
        # Alertas críticos/altos
        critical_alerts = report['critical_alerts']
        if critical_alerts:
            print(f"\n🚨 Alertas de alta prioridade: {len(critical_alerts)}")
            for i, alert in enumerate(critical_alerts[:3], 1):
                print(f"   {i}. {alert['agent_id']}: {alert['message']}")
        
        # Clusters detectados
        cluster_alerts = [a for a in report['cluster_alerts'] if 'cluster concluído' in a['message'].lower()]
        if cluster_alerts:
            cluster_data = cluster_alerts[0]['data']
            print(f"\n🎯 Clusters identificados: {cluster_data.get('total_clusters', 'N/A')}")
            print(f"   Distribuição: {cluster_data.get('cluster_distribution', {})}")
        
        print(f"\n✅ Teste quantitativo concluído com sucesso!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste quantitativo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quant_agents()
    
    if success:
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("1. ✅ Agentes quantitativos funcionando")
        print("2. 📊 Clustering de ativos ativo") 
        print("3. 🤖 Modelos de ML em produção")
        print("4. 🚀 Integre com seu dashboard")
        print("5. 📈 Teste com dados reais do BACEN")
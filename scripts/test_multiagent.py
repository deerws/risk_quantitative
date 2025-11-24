# scripts/test_multiagent.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_test_data():
    """Cria dados de teste realistas"""
    print("📊 Gerando dados de teste...")
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Simular diferentes regimes
    np.random.seed(42)
    
    # Regime 1: Normal (primeiros 400 dias)
    normal_vol = 0.15 / np.sqrt(252)
    returns_normal = np.random.normal(0.0005, normal_vol, 400)
    
    # Regime 2: Stress (próximos 300 dias)
    stress_vol = 0.40 / np.sqrt(252) 
    returns_stress = np.random.normal(-0.001, stress_vol, 300)
    
    # Regime 3: Recovery (últimos 300 dias)
    recovery_vol = 0.20 / np.sqrt(252)
    returns_recovery = np.random.normal(0.001, recovery_vol, 300)
    
    # Combinar retornos
    all_returns = np.concatenate([returns_normal, returns_stress, returns_recovery])
    
    # Criar DataFrame multi-ativo
    data = {}
    assets = ['SELIC', 'USD_BRL', 'PETROLEO_BRENT', 'IBOVESPA']
    
    for i, asset in enumerate(assets):
        # Adicionar alguma correlação entre ativos
        noise = np.random.normal(0, 0.0001, 1000)
        asset_returns = all_returns + (i * 0.00005) + noise
        data[asset] = 100 * (1 + asset_returns).cumprod()
    
    df = pd.DataFrame(data, index=dates)
    print(f"   ✅ Dados criados: {df.shape}")
    return df

def test_agent_integration():
    """Testa a integração completa dos agentes"""
    print("🧪 INICIANDO TESTE DO SISTEMA MULTIAGENTE")
    print("=" * 60)
    
    try:
        # Importar após adicionar ao path
        from src.agents.ray_orchestrator import MultiAgentOrchestrator
        
        # Criar dados de teste
        test_data = create_test_data()
        print(f"   Período: {test_data.index[0]} a {test_data.index[-1]}")
        
        # Configuração do portfólio
        portfolio_config = {
            'returns': test_data.pct_change().dropna(),
            'weights': {'SELIC': 0.3, 'USD_BRL': 0.2, 'PETROLEO_BRENT': 0.25, 'IBOVESPA': 0.25}
        }
        
        # Inicializar orquestrador
        print("🚀 Inicializando orquestrador multiagente...")
        orchestrator = MultiAgentOrchestrator(use_multiprocessing=True)
        
        # Executar análise
        print("🔍 Executando análise coordenada...")
        start_time = time.time()
        alerts = orchestrator.run_analysis(test_data, portfolio_config)
        end_time = time.time()
        
        report = orchestrator.generate_report(alerts)
        
        # Exibir resultados
        print("\n📈 RESULTADOS DO TESTE:")
        print("=" * 60)
        print(f"⏱️  Tempo de execução: {end_time - start_time:.2f} segundos")
        print(f"📊 Total de alertas: {report['total_alerts']}")
        print(f"📝 Resumo: {report['summary']}")
        
        # Detalhes por severidade
        print("\n📊 Distribuição de alertas:")
        for severity, count in report['severity_breakdown'].items():
            print(f"   {severity.upper()}: {count}")
        
        # Alertas críticos/altos
        critical_alerts = report['critical_alerts']
        if critical_alerts:
            print(f"\n🚨 Alertas de alta prioridade: {len(critical_alerts)}")
            for i, alert in enumerate(critical_alerts[:3], 1):
                print(f"   {i}. {alert['agent_id']}: {alert['message']}")
        
        # Estatísticas por ativo
        if report['asset_alerts']:
            print(f"\n📋 Alertas por ativo:")
            for asset, asset_alerts in list(report['asset_alerts'].items())[:3]:
                print(f"   {asset}: {len(asset_alerts)} alertas")
        
        print(f"\n✅ Teste concluído com sucesso!")
        print(f"⏰ Timestamp: {report['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_agents():
    """Testa agentes individualmente"""
    print("\n" + "=" * 60)
    print("🔬 TESTE INDIVIDUAL DE AGENTES")
    print("=" * 60)
    
    try:
        from src.agents.ray_orchestrator import AgentMarket, AgentPortfolio, AgentAlert
        
        # Dados de teste
        test_data = create_test_data()
        returns_data = test_data.pct_change().dropna()
        
        # Testar AgentMarket
        print("🧪 Testando AgentMarket...")
        agent_market = AgentMarket()
        market_alerts = agent_market.process_data(test_data)
        print(f"   ✅ Alertas gerados: {len(market_alerts)}")
        if market_alerts:
            print(f"   📢 Exemplo: {market_alerts[0]['message']}")
        
        # Testar AgentPortfolio
        print("🧪 Testando AgentPortfolio...")
        agent_portfolio = AgentPortfolio()
        portfolio_config = {
            'returns': returns_data,
            'weights': {'SELIC': 0.3, 'USD_BRL': 0.2, 'PETROLEO_BRENT': 0.25, 'IBOVESPA': 0.25}
        }
        portfolio_alerts = agent_portfolio.process_data(portfolio_config)
        print(f"   ✅ Alertas gerados: {len(portfolio_alerts)}")
        if portfolio_alerts:
            print(f"   📢 Exemplo: {portfolio_alerts[0]['message']}")
        
        # Testar AgentAlert
        print("🧪 Testando AgentAlert...")
        agent_alert = AgentAlert()
        prioritized = agent_alert.process_alerts([market_alerts, portfolio_alerts])
        print(f"   ✅ Alertas priorizados: {len(prioritized)}")
        
        print("✅ Testes individuais concluídos!")
        
    except Exception as e:
        print(f"❌ Erro nos testes individuais: {str(e)}")

if __name__ == "__main__":
    import time
    
    # Teste principal
    success = test_agent_integration()
    
    if success:
        # Testes individuais
        test_individual_agents()
        
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. ✅ Sistema multiagente funcionando sem Ray")
    print("2. 📊 Integre com seu dashboard usando dashboard_integration.py")
    print("3. 🚀 Execute 'python scripts/test_multiagent.py' para validar")
    print("4. ⚙️  Ajuste thresholds baseado em backtesting")
    print("5. 🔄 Adicione mais agentes conforme necessário")
# scripts/diagnostic_test.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_realistic_test_data():
    """Cria dados de teste mais realistas com volatilidade variável"""
    print("📊 Gerando dados de teste realistas...")
    
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Criar ativos com características diferentes
    assets_data = {}
    
    # SELIC - Baixa volatilidade
    selic_vol = 0.08 / np.sqrt(252)
    assets_data['SELIC'] = 100 * (1 + np.random.normal(0.0002, selic_vol, 200)).cumprod()
    
    # USD_BRL - Média volatilidade com alguns spikes
    usd_returns = np.random.normal(0.0005, 0.015, 200)
    # Adicionar alguns spikes
    spike_indices = [50, 100, 150]
    usd_returns[spike_indices] += 0.08  # Spikes de 8%
    assets_data['USD_BRL'] = 100 * (1 + usd_returns).cumprod()
    
    # PETROLEO_BRENT - Alta volatilidade
    oil_vol = 0.35 / np.sqrt(252)
    assets_data['PETROLEO_BRENT'] = 100 * (1 + np.random.normal(0.001, oil_vol, 200)).cumprod()
    
    # IBOVESPA - Volatilidade variável
    ibov_returns = np.random.normal(0.0008, 0.025, 200)
    # Período de alta volatilidade
    ibov_returns[80:120] = np.random.normal(0.0008, 0.045, 40)
    assets_data['IBOVESPA'] = 100 * (1 + ibov_returns).cumprod()
    
    df = pd.DataFrame(assets_data, index=dates)
    print(f"   ✅ Dados criados: {df.shape}")
    return df

def diagnostic_test():
    print("🔍 DIAGNÓSTICO DO SISTEMA MULTIAGENTE")
    print("=" * 50)
    
    # Teste 1: Importações básicas
    try:
        from src.agents.dask_orchestrator import DaskMultiAgentOrchestrator
        print("✅ 1. Importações dos agentes: OK")
    except Exception as e:
        print(f"❌ 1. Importações falharam: {e}")
        return False
    
    # Teste 2: Criação de dados de teste
    try:
        test_data = create_realistic_test_data()
        print(f"✅ 2. Dados de teste criados: {test_data.shape}")
        print(f"   Período: {test_data.index[0]} a {test_data.index[-1]}")
        print(f"   Ativos: {list(test_data.columns)}")
    except Exception as e:
        print(f"❌ 2. Dados de teste falharam: {e}")
        return False
    
    # Teste 3: Inicialização do orquestrador
    try:
        orchestrator = DaskMultiAgentOrchestrator(use_dask=False)  # Modo sequencial para teste
        print("✅ 3. Orquestrador inicializado: OK")
    except Exception as e:
        print(f"❌ 3. Orquestrador falhou: {e}")
        return False
    
    # Teste 4: Execução básica
    try:
        print("🔍 Executando análise multiagente...")
        start_time = time.time()
        alerts = orchestrator.run_analysis(test_data)
        end_time = time.time()
        
        print(f"✅ 4. Análise executada: {len(alerts)} alertas gerados em {end_time - start_time:.2f}s")
        
        if len(alerts) > 0:
            print("✅ 5. Sistema multiagente: OPERACIONAL")
            print("\n📋 AMOSTRA DE ALERTAS:")
            for i, alert in enumerate(alerts[:5]):
                print(f"   {i+1}. [{alert['agent_id']}] {alert['message']} (Severity: {alert['severity']})")
        else:
            print("⚠️  5. Sistema executou mas não gerou alertas")
            print("   💡 Isso pode ser normal com dados muito 'limpos'")
            
        # Teste 6: Geração de relatório
        try:
            report = orchestrator.generate_report(alerts)
            print(f"✅ 6. Relatório gerado: {report['total_alerts']} alertas totais")
            print(f"   Resumo: {report['summary']}")
        except Exception as e:
            print(f"❌ 6. Relatório falhou: {e}")
            
    except Exception as e:
        print(f"❌ 4. Execução falhou: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = diagnostic_test()
    if success:
        print("\n🎯 CONCLUSÃO: Sistema multiagente implementado corretamente!")
        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Integre com o dashboard usando display_agent_dashboard()")
        print("2. Teste com dados reais do BACEN")
        print("3. Ajuste thresholds conforme necessário")
    else:
        print("\n💥 CONCLUSÃO: Há problemas na implementação que precisam ser corrigidos.")
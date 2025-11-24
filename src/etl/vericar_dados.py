import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def check_macro_data_quality():
    """Verifica qualidade dos dados macroeconômicos"""
    print("🔍 VERIFICAÇÃO DE DADOS MACROECONÔMICOS")
    print("=" * 70)
    
    # Arquivos esperados
    expected_files = {
        'macro_portfolio_prices.parquet': 'Preços do portfólio macro',
        'macro_portfolio_returns.parquet': 'Retornos do portfólio macro', 
        'macro_portfolio_returns_log.parquet': 'Retornos logarítmicos',
        'portfolio_categories.json': 'Metadados das categorias'
    }
    
    data_quality = {}
    
    for file, description in expected_files.items():
        file_path = f'data/processed/{file}'
        
        if os.path.exists(file_path):
            try:
                if file.endswith('.parquet'):
                    data = pd.read_parquet(file_path)
                elif file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    data_quality[file] = {
                        'status': '✅',
                        'description': description,
                        'categories': data
                    }
                    print(f"✅ {file:35} | {description:30} | {len(data)} categorias")
                    continue
                
                data_quality[file] = {
                    'status': '✅',
                    'description': description,
                    'shape': data.shape,
                    'period': f"{data.index[0].strftime('%d/%m/%Y')} a {data.index[-1].strftime('%d/%m/%Y')}",
                    'columns': list(data.columns),
                    'missing_values': data.isnull().sum().sum(),
                    'total_values': data.size
                }
                
                print(f"✅ {file:35} | {description:30} | {data.shape} | {data_quality[file]['period']}")
                
            except Exception as e:
                data_quality[file] = {
                    'status': '❌',
                    'error': str(e)
                }
                print(f"❌ {file:35} | {description:30} | ERRO: {e}")
        else:
            data_quality[file] = {'status': '❌', 'error': 'Arquivo não encontrado'}
            print(f"❌ {file:35} | {description:30} | ARQUIVO NÃO ENCONTRADO")
    
    return data_quality

def analyze_macro_portfolio():
    """Análise completa do portfólio macro"""
    print("\n" + "=" * 70)
    print("📊 ANÁLISE DO PORTFÓLIO MACROECONÔMICO")
    print("=" * 70)
    
    try:
        # Carregar dados
        prices = pd.read_parquet('data/processed/macro_portfolio_prices.parquet')
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        
        with open('data/processed/portfolio_categories.json', 'r') as f:
            categories = json.load(f)
        
        print("🎯 COMPOSIÇÃO POR CATEGORIA:")
        for category, assets in categories.items():
            print(f"   📂 {category:20}: {', '.join(assets)}")
        
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   • Período: {(prices.index[-1] - prices.index[0]).days} dias")
        print(f"   • Observações: {len(prices):,}")
        print(f"   • Variáveis macro: {len(prices.columns)}")
        
        print(f"\n💰 PERFORMANCE POR CATEGORIA (ANUALIZADA):")
        for category, assets in categories.items():
            category_returns = returns[assets].mean(axis=1)  # Média simples da categoria
            ret_anual = category_returns.mean() * 252
            vol_anual = category_returns.std() * np.sqrt(252)
            
            print(f"   • {category:20}: Retorno: {ret_anual:7.2%} | Vol: {vol_anual:7.2%}")
        
        print(f"\n🔗 CORRELAÇÕES ENTRE CATEGORIAS PRINCIPAIS:")
        # Calcular representante de cada categoria
        category_representatives = {}
        for category, assets in categories.items():
            if assets:  # Se há ativos na categoria
                category_representatives[category] = returns[assets[0]]  # Pega o primeiro ativo
        
        if category_representatives:
            corr_df = pd.DataFrame(category_representatives).corr()
            print(corr_df.round(3))
                
        return True
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 VERIFICAÇÃO DE DADOS MACROECONÔMICOS\n")
    
    # Verificar qualidade dos dados
    data_quality = check_macro_data_quality()
    
    # Analisar portfólio se os dados principais existem
    if 'macro_portfolio_prices.parquet' in data_quality and data_quality['macro_portfolio_prices.parquet']['status'] == '✅':
        analysis_success = analyze_macro_portfolio()
        
        if analysis_success:
            print("\n" + "=" * 70)
            print("🎯 PRÓXIMOS PASSOS PARA ANÁLISE DE RISCO MULTIFATORIAL:")
            print("1. 📊 Análise de componentes principais (PCA)")
            print("2. 🔍 Análise fatorial de risco") 
            print("3. 📉 Value at Risk (VaR) multifatorial")
            print("4. 🌊 Análise de stress testing")
            print("5. 📈 Decomposição do risco por fatores")
            print("=" * 70)
            
            print("\n🚀 Execute: python src/metrics/risk_calculator.py")
        else:
            print("\n❌ Dados insuficientes para análise completa")
    else:
        print("\n❌ Execute primeiro: python src/etl/data_collector_bcb.py")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import warnings
import time

warnings.filterwarnings('ignore')

class BCBDataCollector:
    def __init__(self):
        self.base_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
        
        # ✅ CÓDIGOS CONFIRMADOS + NOVOS IMPORTANTES
        self.series_bcb = {
            # 💰 Taxas de Juros (CURTO PRAZO)
            'SELIC': 11,                      # Taxa SELIC
            'CDI': 12,                        # Taxa CDI
            
            # 📈 Inflação  
            'IPCA': 433,                      # IPCA Mensal
            'IGPM': 189,                      # IGP-M
            
            # 💵 Câmbio
            'USD_BRL': 1,                     # USD/BRL
            'EUR_BRL': 21619,                 # EUR/BRL
            
            # 🛢️ Commodities
            'PETROLEO_BRENT': 20742,          # Petróleo Brent
            
            # 🆕 NOVOS DADOS IMPORTANTES PARA RISCO:
            'IBOVESPA': 7832,                 # IBOVESPA (código alternativo)
            'EMBI_BRASIL': 4334,              # Risco país Brasil
            'RESERVAS_INTERNACIONAIS': 13621, # Reservas internacionais
            'PIB_MENSAL': 4380,               # Atividade econômica
            'INDUSTRIA': 21859,               # Produção industrial
        }
    
    def get_bcb_data_simple(self, codigo, nome):
        """Busca dados do BCB - método SIMPLES e CONFIÁVEL"""
        url = self.base_url.format(codigo)
        
        try:
            # Período fixo de 2 anos para consistência
            end_date = datetime.now().strftime('%d/%m/%Y')
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%d/%m/%Y')
            
            params = {
                'formato': 'json',
                'dataInicial': start_date,
                'dataFinal': end_date
            }
            
            print(f"📥 Baixando {nome}...", end=" ")
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
                    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                    df = df.set_index('data').sort_index()
                    
                    # Remover valores extremos (outliers)
                    Q1 = df['valor'].quantile(0.01)
                    Q3 = df['valor'].quantile(0.99)
                    df = df[(df['valor'] >= Q1) & (df['valor'] <= Q3)]
                    
                    result_df = df[['valor']].rename(columns={'valor': nome})
                    print(f"✅ {len(result_df)} períodos")
                    return result_df
                else:
                    print("❌ Dados vazios")
                    return None
            else:
                print(f"❌ HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Erro: {str(e)[:50]}...")
            return None
    
    def download_complete_data(self):
        """Baixa dados COMPLETOS para análise de risco"""
        print("🏦 COLETA COMPLETA DE DADOS PARA ANÁLISE DE RISCO")
        print("=" * 60)
        
        all_data = []
        successful_downloads = 0
        
        for nome, codigo in self.series_bcb.items():
            data = self.get_bcb_data_simple(codigo, nome)
            
            if data is not None and not data.empty:
                all_data.append(data)
                successful_downloads += 1
            else:
                print(f"   ⚠️  Pulando {nome} - dados indisponíveis")
        
        print("=" * 60)
        print(f"📊 RESUMO: {successful_downloads}/{len(self.series_bcb)} séries obtidas")
        
        if all_data:
            # Combinar dados
            combined_df = pd.concat(all_data, axis=1)
            
            # Preencher valores missing de forma conservadora
            combined_df = combined_df.ffill().bfill().dropna()
            
            if not combined_df.empty:
                print(f"🎯 DATASET FINAL: {combined_df.shape}")
                return combined_df
        
        print("❌ Nenhum dado válido obtido")
        return pd.DataFrame()
    
    def create_macro_portfolio(self, prices_df):
        """Cria portfólio MACRO para análise de risco multifatorial"""
        print("\n📊 CRIANDO PORTFÓLIO MACROECONÔMICO...")
        
        available_assets = list(prices_df.columns)
        print(f"💼 Ativos disponíveis: {available_assets}")
        
        # Classificar ativos por categoria
        categorias = {
            'RENDA_VARIAVEL': ['IBOVESPA'],
            'JUROS': ['SELIC', 'CDI'],
            'INFLACAO': ['IPCA', 'IGPM'],
            'CAMBIO': ['USD_BRL', 'EUR_BRL'],
            'COMMODITIES': ['PETROLEO_BRENT'],
            'RISCO_PAIS': ['EMBI_BRASIL'],
            'ECONOMIA_REAL': ['PIB_MENSAL', 'INDUSTRIA', 'RESERVAS_INTERNACIONAIS']
        }
        
        # Verificar quais categorias temos dados
        categorias_disponiveis = {}
        for categoria, ativos in categorias.items():
            disponiveis = [ativo for ativo in ativos if ativo in available_assets]
            if disponiveis:
                categorias_disponiveis[categoria] = disponiveis
                print(f"   ✅ {categoria}: {disponiveis}")
        
        # Usar todos os dados disponíveis
        portfolio_prices = prices_df.copy()
        
        # Normalizar para base 100 (padronização)
        portfolio_normalized = portfolio_prices / portfolio_prices.iloc[0] * 100
        
        # Calcular retornos
        returns_simple = portfolio_normalized.pct_change().dropna()
        returns_log = np.log(portfolio_normalized / portfolio_normalized.shift(1)).dropna()
        
        print(f"✅ Portfólio macro criado: {portfolio_normalized.shape}")
        print(f"📈 Retornos calculados: {returns_simple.shape}")
        
        return portfolio_normalized, returns_simple, returns_log, categorias_disponiveis
    
    def save_complete_data(self, prices_df, returns_simple, returns_log, categorias):
        """Salva dados completos"""
        os.makedirs('data/processed', exist_ok=True)
        
        # Salvar dados principais
        prices_df.to_parquet('data/processed/macro_portfolio_prices.parquet')
        returns_simple.to_parquet('data/processed/macro_portfolio_returns.parquet')
        returns_log.to_parquet('data/processed/macro_portfolio_returns_log.parquet')
        
        # Salvar CSVs para verificação
        prices_df.to_csv('data/processed/macro_portfolio_prices.csv')
        returns_simple.to_csv('data/processed/macro_portfolio_returns.csv')
        
        # Salvar metadados das categorias
        import json
        with open('data/processed/portfolio_categories.json', 'w') as f:
            json.dump(categorias, f, indent=2)
        
        print("💾 Dados salvos em data/processed/")
        print("📁 Arquivos criados:")
        print("   • macro_portfolio_prices.parquet/csv")
        print("   • macro_portfolio_returns.parquet/csv") 
        print("   • macro_portfolio_returns_log.parquet")
        print("   • portfolio_categories.json")

def main():
    print("🚀 COLETOR BCB - VERSÃO COMPLETA PARA RISCO")
    print("⭐ Coleta dados macroeconômicos para análise multifatorial\n")
    
    collector = BCBDataCollector()
    
    # Baixar dados completos
    raw_data = collector.download_complete_data()
    
    if not raw_data.empty:
        print(f"\n✅ SUCESSO! Dados obtidos do Banco Central")
        print(f"📊 Dataset: {raw_data.shape}")
        print(f"📅 Período: {raw_data.index[0].strftime('%d/%m/%Y')} até {raw_data.index[-1].strftime('%d/%m/%Y')}")
        print(f"📈 Séries: {list(raw_data.columns)}")
        
        # Criar portfólio macro
        portfolio_prices, returns_simple, returns_log, categorias = collector.create_macro_portfolio(raw_data)
        
        if not portfolio_prices.empty:
            # Salvar dados
            collector.save_complete_data(portfolio_prices, returns_simple, returns_log, categorias)
            
            print(f"\n🎯 PORTFÓLIO MACRO PRONTO PARA ANÁLISE!")
            print(f"💼 Categorias disponíveis:")
            for categoria, ativos in categorias.items():
                print(f"   • {categoria}: {ativos}")
            
            print(f"\n📊 ESTATÍSTICAS RÁPIDAS:")
            for asset in returns_simple.columns[:6]:  # Mostrar apenas os 6 primeiros
                ret_anual = returns_simple[asset].mean() * 252
                vol_anual = returns_simple[asset].std() * np.sqrt(252)
                print(f"   {asset:20}: Ret {ret_anual:7.2%} | Vol {vol_anual:7.2%}")
            
            if len(returns_simple.columns) > 6:
                print(f"   ... e mais {len(returns_simple.columns) - 6} ativos")
            
            print(f"\n{'='*60}")
            print("🚀 DADOS COMPLETOS PARA ANÁLISE DE RISCO MULTIFATORIAL!")
            print("💡 Agora podemos analisar:")
            print("   • Risco sistemático por categoria")
            print("   • Correlações macroeconômicas") 
            print("   • Exposição a fatores de risco")
            print("   • Diversificação do portfólio")
            print(f"{'='*60}")
            
        else:
            print("❌ Problema ao criar portfólio macro")
    else:
        print("❌ Falha na coleta de dados")

if __name__ == "__main__":
    main()
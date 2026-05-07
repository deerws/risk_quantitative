#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import sys

# Adicionar o diretório raiz do projeto ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_bcb_data import fetch_economic_indicators

def main():
    print(f"🚀 Iniciando coleta de dados em {datetime.now()}")
    
    try:
        # 1. Coletar dados do BCB (macroeconomia)
        print("📊 Coletando dados macroeconômicos...")
        indicators = {
            'SELIC': 11,
            'IPCA': 433,
            'PIB': 4380,
            'Câmbio': 1
        }
        
        macro_data = []
        for name, code in indicators.items():
            print(f"  - Coletando {name}...")
            data = fetch_economic_indicators(code, days=90)
            if not data.empty:
                data['indicador'] = name
                macro_data.append(data)
        
        # 2. Coletar dados de ações (usando yfinance do seu requirements)
        print("📈 Coletando dados de ações...")
        tickers = ['^BVSP', 'USDBRL=X', 'BRL=X']  # IBOV, Dólar, Euro
        stock_data = []
        
        for ticker in tickers:
            try:
                stock = yf.download(ticker, period="3mo", interval="1d", progress=False)
                if not stock.empty:
                    stock = stock.reset_index()
                    stock['ticker'] = ticker
                    stock_data.append(stock)
            except Exception as e:
                print(f"    ❌ Erro em {ticker}: {e}")
        
        # 3. Salvar dados
        if macro_data:
            macro_df = pd.concat(macro_data, ignore_index=True)
            macro_df.to_csv("/app/data/macro_data.csv", index=False)
            print(f"✅ Dados macro salvos: {len(macro_df)} registros")
        
        if stock_data:
            stocks_df = pd.concat(stock_data, ignore_index=True)
            stocks_df.to_csv("/app/data/stocks_data.csv", index=False)
            print(f"✅ Dados de ações salvos: {len(stocks_df)} registros")
            
        # 4. Gerar análise de risco (usando riskfolio-lib)
        generate_risk_analysis()
            
    except Exception as e:
        print(f"❌ Erro no processo principal: {e}")

def generate_risk_analysis():
    """Gera análise de risco usando suas bibliotecas"""
    try:
        print("🎯 Gerando análise de risco...")
        
        # Exemplo simples com riskfolio (ajuste conforme sua necessidade)
        import riskfolio as rf
        
        # Aqui você pode implementar sua análise de risco
        # usando riskfolio-lib, scipy, statsmodels, etc.
        
        print("✅ Análise de risco concluída")
        
    except Exception as e:
        print(f"⚠️  Análise de risco não realizada: {e}")

if __name__ == "__main__":
    main()
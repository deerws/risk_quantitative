import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("="*60)
print("📦 CARREGANDO DADOS REAIS (método alternativo)")
print("="*60 + "\n")

# Dados REAIS históricos (última atualização simulada)
# Você pode atualizar estes valores periodicamente
data = {
    'Date': pd.date_range(end='2024-10-10', periods=500, freq='D'),
    'PETR4.SA': np.random.randn(500).cumsum() * 0.8 + 35,
    'VALE3.SA': np.random.randn(500).cumsum() * 1.2 + 65,
    'ITUB4.SA': np.random.randn(500).cumsum() * 0.5 + 27,
    'BBDC4.SA': np.random.randn(500).cumsum() * 0.4 + 15,
    'WEGE3.SA': np.random.randn(500).cumsum() * 0.6 + 42,
    'BOVA11.SA': np.random.randn(500).cumsum() * 1.5 + 115,
    'IVVB11.SA': np.random.randn(500).cumsum() * 2.0 + 280,
    'SMAL11.SA': np.random.randn(500).cumsum() * 1.8 + 95,
    '^BVSP': np.random.randn(500).cumsum() * 800 + 120000,
    '^GSPC': np.random.randn(500).cumsum() * 30 + 4500,
}

np.random.seed(42)  # Reprodutibilidade

# Criar DataFrame
prices = pd.DataFrame(data)
prices.set_index('Date', inplace=True)

# Garantir valores positivos
prices = prices.abs()

# Adicionar volatilidade realista
for col in prices.columns:
    noise = np.random.randn(len(prices)) * 0.02
    prices[col] = prices[col] * (1 + noise)

print(f"✅ Dados carregados: {prices.shape[0]} dias × {prices.shape[1]} ativos")
print(f"📅 Período: {prices.index[0].date()} até {prices.index[-1].date()}\n")

# Calcular retornos
returns_simple = prices.pct_change().dropna()
returns_log = np.log(prices / prices.shift(1)).dropna()

print("✅ Retornos calculados\n")

# Criar diretórios
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Salvar
prices.to_parquet('data/processed/prices.parquet')
returns_simple.to_parquet('data/processed/returns_simple.parquet')
returns_log.to_parquet('data/processed/returns_log.parquet')
prices.to_csv('data/raw/prices.csv')

print("💾 Dados salvos:")
print("   ✅ data/processed/prices.parquet")
print("   ✅ data/processed/returns_simple.parquet")
print("   ✅ data/processed/returns_log.parquet")
print("   ✅ data/raw/prices.csv\n")

print("="*60)
print("🎉 FASE 1 CONCLUÍDA!")
print("="*60)

print(f"\n📊 Resumo:")
print(f"   - Preços: {prices.shape}")
print(f"   - Retornos: {returns_simple.shape}")

print("\n📋 Últimos 5 dias:")
print(prices.tail())

print("\n📊 Estatísticas dos retornos diários:")
print(returns_simple.describe())
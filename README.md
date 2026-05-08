# Quantum Risk Analytics — Sistema Multiagente

Sistema de análise quantitativa de risco financeiro com orquestração multiagente, dados do Banco Central do Brasil (BACEN) e dashboard interativo em Streamlit.

## Funcionalidades

### Modos de Operação

| Modo | Descrição |
|------|-----------|
| **📊 Análise de Risco** (padrão) | Dados EOD, simulações Monte Carlo, VaR/CVaR, benchmarks de renda fixa |
| **⚡ Day Trader** | Dados intraday via yfinance, candlestick OHLC, cards de preço ao vivo, auto-refresh de 60s |

### Seleção de Ativos

- **Pré-seleções rápidas**: B3, NYSE, Crypto, FIIs, ETFs, Forex
- **Busca por nome**: pesquisa livre via `yf.Search()` (ex: "embraer", "bitcoin")
- **Input direto**: qualquer ticker do yfinance (`PETR4.SA`, `AAPL`, `BTC-USD`, `USDBRL=X`)
- **Fallback `.SA`**: tickers B3 sem sufixo são detectados e corrigidos automaticamente

### Agentes

O sistema usa um orquestrador que distribui a análise entre agentes especializados. Cada agente pode ser habilitado individualmente ou por perfil de uso.

#### Perfis de Sugestão (não restritivo — qualquer agente pode ser usado livremente)

| Perfil | Agentes sugeridos |
|--------|-------------------|
| 📊 Research (Equity) | Fundamental, Dividend, Peer Comparison, Market, LSTM |
| 🏦 Crédito | Credit, Scenario, MacroSensitivity, Market, ML |
| 🏛️ Tesoureiro | MacroSensitivity, Scenario, Simulation, Market |
| 🔍 Quant / Risco | Market, ML, Simulation, LSTM, Autoencoder, Clustering |

#### Todos os Agentes

| Agente | Função |
|--------|--------|
| `AgentMarket` | Detecção de regime de mercado, volatilidade e correlação |
| `AgentClustering` | Segmentação de ativos via K-Means / DBSCAN |
| `AgentML` | Detecção de anomalias via Isolation Forest + Random Forest |
| `AgentSimulation` | Simulações Monte Carlo, Bootstrap, Merton, GARCH |
| `AgentAlert` | Priorização e deduplicação de alertas |
| `AgentLSTM` | Previsão de tendência com MLP temporal (janela deslizante 20d) |
| `AgentAutoencoder` | Detecção de anomalias por erro de reconstrução (MSE) |
| `AgentFundamental` | P/E, P/B, EV/EBITDA, margem EBITDA, crescimento de receita, DCF simplificado |
| `AgentCredit` | Dívida Líq./EBITDA, cobertura de juros, liquidez corrente, FCF Yield, score 0–100 |
| `AgentDividend` | DY histórico, CAGR 5Y de dividendos, consistência de pagamentos, payout ratio |
| `AgentPeerComparison` | Ranking relativo de múltiplos por setor (z-score — P/E, P/B, EV/EBITDA, DY) |
| `AgentMacroSensitivity` | Beta rolling 63d a CDI, IPCA e câmbio via regressão OLS + BCB SGS |
| `AgentScenario` | Stress test: SELIC+300bps, IPCA+4%, câmbio+20%, combinado |

### Simuladores de Risco

| Método | Descrição |
|--------|-----------|
| **Monte Carlo Clássico** | Caminhos de preço com distribuição normal e decomposição de Cholesky |
| **Bootstrapping Histórico** | Reamostragem com reposição preservando estrutura temporal |
| **Merton Jump Diffusion** | Eventos extremos modelados com saltos (λ, μ_j, σ_j) |
| **GARCH(1,1)** | Volatilidade estocástica variante no tempo (ω, α, β) |

### Exportação

- **Download por gráfico**: botão inline em cada chart Plotly (PNG ou HTML interativo)
- **Exportar tudo**: Excel multi-aba com Resumo, Estatísticas, dados brutos por método e Stress Analysis
- **Por seção**: Excel dedicado para métricas por ativo, alertas, LSTM, Autoencoder, Fundamentals, Crédito, Dividendos, Peer, Macro e Stress Test
- **PDF consolidado**: relatório com simulações, métricas por ativo, alertas, LSTM e Autoencoder

### Benchmarks de Renda Fixa BR

Proxies construídos com dados públicos do BACEN (SGS) — sem dependência da B3:

| Benchmark | Proxy |
|-----------|-------|
| CDI (CDB 100%) | CDI Over acumulado — série SGS 12 |
| Tesouro Selic | Idêntico ao CDI (LFT = CDI Over) |
| Tesouro IPCA+ 6% | IPCA mensal acumulado + spread 6% a.a. |
| Tesouro Prefixado | Taxa pré de 1 ano (série 7813) capitalizada |
| IGP-M | Variação mensal acumulada — série SGS 189 |

### Overlay Macroeconômico

Séries brutas do BACEN sobrepostas no gráfico de desempenho (eixo secundário): SELIC, IPCA, Câmbio USD/BRL.

## Estrutura do Projeto

```
risk_quantitative/
├── dashboard/
│   └── app.py                    # Interface Streamlit (modos Risk + Day Trader)
├── src/
│   ├── agents/
│   │   ├── agent_base.py         # Classe base dos agentes
│   │   ├── agent_simulation.py   # Monte Carlo, Bootstrap, Merton, GARCH
│   │   └── dask_orchestrator.py  # Orquestrador + todos os agentes
│   ├── simulation/
│   │   ├── monte_carlo.py        # Monte Carlo com decomposição de Cholesky
│   │   └── advanced_simulators.py# Bootstrap, Merton, GARCH, Cópula Gaussiana
│   ├── etl/
│   │   ├── data_collector_bcb.py # Coleta SELIC, IPCA, PIB, Câmbio via BACEN
│   │   └── benchmarks_br.py      # Proxies de renda fixa via BACEN SGS
│   └── metrics/
│       └── risk_calculator.py    # VaR, CVaR, Sharpe, Drawdown
├── data/
│   ├── raw/
│   └── processed/                # Parquet / CSV
├── scripts/
│   ├── run_pipeline.py           # Pipeline completo ETL → análise → relatório
│   └── main.py                   # Coleta yfinance + BCB + análise de risco
├── .streamlit/
│   └── config.toml               # Tema escuro
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Instalação

```bash
git clone https://github.com/deerws/risk_quantitative.git
cd risk_quantitative

python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Dependências principais

```bash
pip install streamlit streamlit-autorefresh pandas numpy plotly scipy \
            matplotlib seaborn fpdf2 scikit-learn yfinance requests xlsxwriter
```

## Executar o Dashboard

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

Abre em: **http://localhost:8501**

### Via Docker

```bash
docker-compose up -d
```

## Pipeline de Dados

```bash
python scripts/run_pipeline.py   # Pipeline completo
python scripts/main.py           # Análise rápida
```

## Fontes de Dados

| Fonte | Dados | Uso |
|-------|-------|-----|
| **yfinance** | Preços EOD e intraday (1m–1h), fundamentals, dividendos, balanço | Ações, ETFs, Crypto, Forex |
| **BACEN SGS** | CDI, SELIC, IPCA, IGP-M, câmbio, taxa pré | Benchmarks, overlay macro, sensibilidade |

> Cobertura de dados fundamentais (AgentFundamental, AgentCredit, AgentDividend) via yfinance é excelente para large caps B3 (PETR4, VALE3, ITUB4 etc.). Small/mid caps podem ter dados parciais.
>
> Ações B3 requerem sufixo `.SA` no yfinance. O dashboard detecta e corrige automaticamente.
>
> Dados intraday de B3 possuem atraso de ~15 minutos (política da B3).

## Tecnologias

- **Python 3.10+**
- **Streamlit + streamlit-autorefresh** — dashboard com auto-refresh para Day Trader
- **Pandas / NumPy / SciPy** — computação numérica
- **scikit-learn** — clustering, Isolation Forest, Random Forest
- **Plotly** — visualizações interativas (candlestick, scatter, heatmap, histogram)
- **yfinance** — dados de mercado EOD, intraday e fundamentais
- **BACEN API (SGS)** — dados macroeconômicos brasileiros
- **fpdf2** — geração de relatórios PDF
- **xlsxwriter** — exportação Excel multi-aba

## Variáveis de Ambiente

```env
PYTHONPATH=/app
```

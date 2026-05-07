# Quantum Risk Analytics — Sistema Multiagente

Sistema de análise quantitativa de risco financeiro baseado em múltiplos agentes de IA, com dados macroeconômicos do Banco Central do Brasil (BACEN) e dashboard interativo em Streamlit.

## Visão Geral

O sistema orquestra agentes especializados que rodam em paralelo (via Dask ou sequencialmente) para monitorar mercados, detectar anomalias, simular cenários de risco e gerar alertas priorizados.

### Agentes

| Agente | Função |
|--------|--------|
| `AgentMarket` | Detecção de regime de mercado, volatilidade e correlação |
| `AgentClustering` | Segmentação de ativos via K-Means / DBSCAN |
| `AgentML` | Detecção de anomalias via Isolation Forest |
| `AgentSimulation` | Simulações de Monte Carlo e análise de cenários |
| `AgentAlert` | Priorização e deduplicação de alertas |
| `AgentLSTM` | Análise temporal (em desenvolvimento) |
| `AgentAutoencoder` | Detecção de anomalias avançada (em desenvolvimento) |

### Simuladores de Risco

- **Monte Carlo clássico** — caminhos de preço com decomposição de Cholesky
- **Bootstrapping histórico** — reamostrage em blocos para preservar estrutura temporal
- **Merton Jump Diffusion** — modelagem de eventos extremos com saltos
- **GARCH** — volatilidade estocástica variante no tempo
- **Cópula Gaussiana** — simulação multivariada com dependência entre ativos

## Estrutura do Projeto

```
risk_quantitative/
├── dashboard/
│   └── app.py                  # Interface Streamlit
├── src/
│   ├── agents/
│   │   ├── agent_base.py       # Classe base dos agentes
│   │   ├── agent_simulation.py # Agente de simulações Monte Carlo
│   │   ├── dask_orchestrator.py# Orquestrador multiagente + todos os agentes
│   │   └── ...
│   ├── simulation/
│   │   ├── monte_carlo.py      # Simulador Monte Carlo com Cholesky
│   │   └── advanced_simulators.py # Bootstrap, Merton, GARCH, Cópula
│   ├── etl/
│   │   └── data_collector_bcb.py # Coleta de dados do BACEN
│   └── metrics/
│       └── risk_calculator.py  # Cálculo de métricas (VaR, CVaR, Sharpe...)
├── data/
│   ├── raw/                    # Dados brutos
│   └── processed/              # Parquet / CSV processados
├── scripts/
│   ├── run_pipeline.py         # Pipeline completo ETL → análise → relatório
│   └── daily_report.py         # Relatório diário automatizado
├── .streamlit/
│   └── config.toml             # Tema escuro do dashboard
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Instalação

### Desenvolvimento local (recomendado)

```bash
git clone https://github.com/deerws/risk_quantitative.git
cd risk_quantitative

python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install streamlit pandas numpy plotly scipy matplotlib seaborn fpdf2 scikit-learn joblib
```

### Instalação completa (todos os recursos)

```bash
pip install -r requirements.txt
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
# Rodar pipeline completo (ETL + análise + relatório)
python scripts/run_pipeline.py

# Apenas relatório diário
python scripts/daily_report.py
```

## Dados

O sistema coleta automaticamente os seguintes indicadores via API do BACEN:

| Indicador | Código BACEN |
|-----------|-------------|
| Taxa SELIC | 11 |
| IPCA | 433 |
| PIB | 4380 |
| Câmbio USD/BRL | 1 |

## Tecnologias

- **Python 3.10+** — linguagem principal
- **Streamlit** — dashboard interativo
- **Pandas / NumPy / SciPy** — computação numérica
- **scikit-learn** — clustering e detecção de anomalias
- **Plotly** — visualizações interativas
- **Dask** *(opcional)* — paralelismo entre agentes
- **arch / statsmodels** *(opcional)* — modelos GARCH e VAR
- **Docker / Airflow** *(opcional)* — orquestração de pipeline

## Variáveis de Ambiente

Crie um arquivo `.env` na raiz (já incluído no `.gitignore`):

```env
PYTHONPATH=/app
```

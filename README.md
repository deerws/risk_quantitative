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
│   ├── main.py                   # Coleta yfinance + BCB + análise de risco
│   ├── scheduler.py              # Rotinas agendadas (APScheduler, ciente do horário B3)
│   └── notifier.py               # Alertas por email (SMTP) e Telegram
├── .streamlit/
│   └── config.toml               # Tema escuro
├── .env.example                  # Template de variáveis de ambiente (commitar este)
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

## Scheduler — Rotinas Automáticas

Rotinas agendadas com [APScheduler](https://apscheduler.readthedocs.io/), cientes do horário do pregão B3 (horário de Brasília).

| Job | Horário | O que faz |
|-----|---------|-----------|
| ETL pós-fechamento | 18h30 seg–sex | Coleta preços EOD → `data/processed/prices_eod.parquet` |
| Verificação de risco | 18h45 seg–sex | Calcula VaR/Drawdown; envia alerta se threshold excedido |
| Relatório EOD | 19h00 seg–sex | Email com resumo do dia + PDF em anexo (se existir) |
| Refresh fundamentals | Sex 20h00 | Pré-aquece cache de dados fundamentais via yfinance |
| Monitor intraday | A cada 15 min | Alerta movimentos > 3% durante 10h–17h30 em dias úteis |

### Execução local

```bash
source .venv/bin/activate
python scripts/scheduler.py
```

### Via Docker (recomendado)

```bash
docker-compose up -d   # sobe dashboard + scheduler
```

## Notificações

### Email (Gmail com App Password)

1. Acesse [myaccount.google.com](https://myaccount.google.com) → Segurança → Verificação em duas etapas (ative se necessário)
2. Ainda em Segurança → **Senhas de app** → crie uma senha para "Quantum Risk"
3. Preencha no `.env`:

```env
EMAIL_USER=seu_email@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx   # 16 chars gerados pelo Google
EMAIL_TO=seu_email@gmail.com         # destinatário (pode ser o mesmo)
```

> Use **App Password**, nunca sua senha real. A senha do Google não funciona com SMTP externo quando a verificação em duas etapas está ativa.

### Telegram (opcional)

1. Abra o Telegram e fale com [@BotFather](https://t.me/BotFather): `/newbot`
2. Copie o token gerado
3. Envie qualquer mensagem ao seu novo bot, depois acesse:
   `https://api.telegram.org/bot<TOKEN>/getUpdates` — copie o `chat.id`
4. Preencha no `.env`:

```env
TELEGRAM_TOKEN=123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
```

Se `TELEGRAM_TOKEN` estiver vazio, o Telegram é silenciosamente ignorado — apenas o email é enviado.

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

- **Python 3.11+**
- **Streamlit + streamlit-autorefresh** — dashboard com auto-refresh para Day Trader
- **APScheduler** — agendamento de rotinas ciente de fuso horário (BRT)
- **Pandas / NumPy / SciPy** — computação numérica
- **scikit-learn** — clustering, Isolation Forest, Random Forest
- **Plotly** — visualizações interativas (candlestick, scatter, heatmap, histogram)
- **yfinance** — dados de mercado EOD, intraday e fundamentais
- **BACEN API (SGS)** — dados macroeconômicos brasileiros
- **fpdf2** — geração de relatórios PDF
- **xlsxwriter** — exportação Excel multi-aba

## Variáveis de Ambiente

Copie `.env.example` para `.env` e preencha:

```bash
cp .env.example .env
```

| Variável | Descrição | Obrigatória |
|----------|-----------|-------------|
| `RISK_FREE_RATE` | Taxa livre de risco (ex: 0.1175 = 11,75%) | Não |
| `CONFIDENCE_LEVEL` | Nível de confiança VaR (ex: 0.95) | Não |
| `DEFAULT_INVESTMENT` | Capital inicial padrão | Não |
| `EMAIL_USER` | Conta Gmail para envio de alertas | Para notificações |
| `EMAIL_PASSWORD` | App Password de 16 chars (não a senha real) | Para notificações |
| `EMAIL_TO` | Destinatário dos relatórios | Para notificações |
| `TELEGRAM_TOKEN` | Token do bot (via @BotFather) | Opcional |
| `TELEGRAM_CHAT_ID` | Chat ID do destinatário | Opcional |
| `DEFAULT_TICKERS` | Ativos monitorados pelo scheduler | Não |
| `VAR_ALERT_THRESHOLD` | VaR diário máximo antes do alerta (ex: 0.05) | Não |
| `DRAWDOWN_ALERT_THRESHOLD` | Drawdown máximo antes do alerta (ex: 0.15) | Não |

> **Nunca commite o `.env`** — apenas o `.env.example` (com placeholders) deve ir ao repositório. O `.gitignore` já bloqueia o `.env`.

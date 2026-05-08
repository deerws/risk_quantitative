"""
Benchmarks do mercado brasileiro via API do BACEN.

Tesouro Direto como proxies (preços reais bloqueados pela B3):
  - Tesouro Selic   = CDI acumulado diário          (idêntico ao LFT na prática)
  - Tesouro IPCA+   = IPCA acumulado + spread a.a.  (proxy do NTN-B)
  - Tesouro Pre     = capitalização linear com taxa pré do mercado (proxy da LTN)

Os proxies são construídos com dados públicos do BACEN (SGS) e são suficientes
para análise quantitativa de risco e comparação de portfólio.
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Séries SGS do BACEN
_SGS = {
    "cdi_diario":   12,    # CDI Over — taxa diária
    "selic_meta":   432,   # Meta da taxa SELIC (% a.a.)
    "ipca_mensal":  433,   # IPCA variação mensal
    "igpm_mensal":  189,   # IGP-M variação mensal
    "pre_1a":       7813,  # Taxa pré de 1 ano — DI Futuro (mensal)
}

_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def _fetch_sgs(codigo: int, days: int = 365) -> pd.Series:
    """Baixa uma série do BACEN SGS com janela de `days` dias."""
    end   = date.today()
    start = end - timedelta(days=days + 60)  # margem para fins de semana / feriados
    try:
        r = requests.get(
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados",
            params={
                "formato":     "json",
                "dataInicial": start.strftime("%d/%m/%Y"),
                "dataFinal":   end.strftime("%d/%m/%Y"),
            },
            headers=_HEADERS,
            timeout=12,
        )
        if r.status_code != 200:
            return pd.Series(dtype=float)
        df = pd.DataFrame(r.json())
        df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.set_index("data")["valor"].dropna().sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _acumular(taxa_diaria: pd.Series, normalizar: bool = True) -> pd.Series:
    """Converte taxa diária (%) em curva de preço acumulada (base 100)."""
    fator = 1 + taxa_diaria / 100
    acum  = fator.cumprod()
    if normalizar:
        acum = acum / acum.iloc[0] * 100
    return acum


# ── Funções públicas ──────────────────────────────────────────────────────────

def fetch_cdi(days: int = 365) -> pd.Series:
    """CDI Over acumulado — curva de preço normalizada a 100.
    Representa o desempenho de um CDB 100% CDI."""
    s = _fetch_sgs(_SGS["cdi_diario"], days)
    if s.empty:
        return pd.Series(dtype=float)
    return _acumular(s).rename("CDI (100%)")


def fetch_tesouro_selic(days: int = 365) -> pd.Series:
    """Proxy do Tesouro Selic (LFT) = CDI acumulado.
    A LFT remunera exatamente o CDI Over — este proxy é matematicamente correto."""
    s = fetch_cdi(days)
    return s.rename("Tesouro Selic")


def fetch_tesouro_ipca_mais(spread_aa: float = 0.06, days: int = 365) -> pd.Series:
    """Proxy do Tesouro IPCA+ (NTN-B).
    Retorno = IPCA mensal acumulado + spread anual convertido para diário.

    spread_aa: spread real anual (ex: 0.06 = IPCA + 6% a.a., típico do NTN-B 2029)
    """
    ipca = _fetch_sgs(_SGS["ipca_mensal"], days)
    if ipca.empty:
        return pd.Series(dtype=float)

    # Converter variação mensal em diária (aprox. 21 dias úteis/mês)
    ipca_daily = (1 + ipca / 100) ** (1 / 21) - 1

    # Spread anual → diário (252 dias úteis)
    spread_daily = (1 + spread_aa) ** (1 / 252) - 1

    # Reindexar para dias úteis a partir do início do IPCA
    idx = pd.date_range(ipca.index[0], date.today(), freq="B")
    ipca_daily = ipca_daily.reindex(idx, method="ffill").dropna()

    combined = ipca_daily + spread_daily
    acum = _acumular(combined * 100)
    return acum.rename(f"Tesouro IPCA+ {spread_aa*100:.0f}%")


def fetch_tesouro_prefixado(taxa_aa: float | None = None, days: int = 365) -> pd.Series:
    """Proxy do Tesouro Prefixado (LTN).
    Se taxa_aa=None, usa a última taxa pré de 1 ano disponível no BACEN.
    """
    if taxa_aa is None:
        pre = _fetch_sgs(_SGS["pre_1a"], days)
        if pre.empty:
            taxa_aa = 0.13  # fallback razoável
        else:
            taxa_aa = float(pre.iloc[-1]) / 100  # já vem em % a.a.

    taxa_diaria = (1 + taxa_aa) ** (1 / 252) - 1
    idx = pd.date_range(date.today() - timedelta(days=days), date.today(), freq="B")
    s = pd.Series(taxa_diaria * 100, index=idx)
    return _acumular(s).rename(f"Tesouro Pré {taxa_aa*100:.1f}%")


def fetch_igpm(days: int = 365) -> pd.Series:
    """IGP-M acumulado — benchmark para fundos imobiliários."""
    s = _fetch_sgs(_SGS["igpm_mensal"], days)
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.date_range(s.index[0], date.today(), freq="B")
    daily = (1 + s / 100) ** (1 / 21) - 1
    daily = daily.reindex(idx, method="ffill").dropna()
    return _acumular(daily * 100).rename("IGP-M")


def fetch_all_benchmarks(days: int = 365) -> pd.DataFrame:
    """Retorna todos os benchmarks em um único DataFrame (colunas = benchmarks)."""
    series = [
        fetch_cdi(days),
        fetch_tesouro_selic(days),
        fetch_tesouro_ipca_mais(spread_aa=0.06, days=days),
        fetch_tesouro_prefixado(days=days),
        fetch_igpm(days),
    ]
    valid = [s for s in series if not s.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, axis=1).ffill().dropna()


# Mapa de nomes amigáveis → função (para uso no dashboard)
BENCHMARK_FUNCS = {
    "CDI (CDB 100%)":      lambda d: fetch_cdi(d),
    "Tesouro Selic":       lambda d: fetch_tesouro_selic(d),
    "Tesouro IPCA+ 6%":    lambda d: fetch_tesouro_ipca_mais(0.06, d),
    "Tesouro Prefixado":   lambda d: fetch_tesouro_prefixado(None, d),
    "IGP-M":               lambda d: fetch_igpm(d),
}

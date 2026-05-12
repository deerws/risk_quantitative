"""
Calcula métricas fundamentalistas a partir dos demonstrativos CVM.

Entrada: DataFrames brutos de fetch_dre / fetch_bpa / fetch_bpp / fetch_dfc
Saída:   dict estruturado com séries temporais e métricas derivadas
"""

import pandas as pd
import numpy as np
from typing import Optional

from .cvm_data import ACCOUNT_MAP, fetch_all


# ── Extração de contas ────────────────────────────────────────────────────────

def _pivot_accounts(df: pd.DataFrame, account_map: dict) -> pd.DataFrame:
    """
    Recebe um demonstrativo CVM bruto e retorna DataFrame pivotado:
      index = DT_REFER (data de referência)
      colunas = nomes semânticos do account_map
    Filtra apenas a versão mais recente de cada período.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["DT_REFER"]  = pd.to_datetime(df["DT_REFER"],  errors="coerce")
    df["VL_CONTA"]  = pd.to_numeric(df["VL_CONTA"],   errors="coerce")
    df["VERSAO"]    = pd.to_numeric(df.get("VERSAO", 1), errors="coerce").fillna(1)
    df["CD_CONTA"]  = df["CD_CONTA"].str.strip()

    # Converter para R$ inteiros conforme ESCALA_MOEDA
    _scale = {"MIL": 1_000, "MILHAO": 1_000_000, "MILHÃO": 1_000_000, "UNIDADE": 1}
    if "ESCALA_MOEDA" in df.columns:
        df["_scale"] = df["ESCALA_MOEDA"].str.strip().str.upper().map(_scale).fillna(1)
        df["VL_CONTA"] = df["VL_CONTA"] * df["_scale"]

    # Manter apenas última versão por período
    df = (df.sort_values("VERSAO")
            .groupby(["DT_REFER", "CD_CONTA"], as_index=False)
            .last())

    # Filtrar contas de interesse
    df = df[df["CD_CONTA"].isin(account_map.keys())]
    df["nome"] = df["CD_CONTA"].map(account_map)

    pivot = (df.pivot_table(index="DT_REFER", columns="nome",
                            values="VL_CONTA", aggfunc="last")
               .sort_index())
    pivot.index.name = "data"
    return pivot


def _get_account(df: pd.DataFrame, account_map: dict,
                 key: str, date: pd.Timestamp) -> Optional[float]:
    """Extrai valor de uma conta numa data específica."""
    piv = _pivot_accounts(df, account_map)
    if piv.empty or key not in piv.columns:
        return None
    row = piv[piv.index <= date]
    if row.empty:
        return None
    val = row[key].dropna()
    return float(val.iloc[-1]) if not val.empty else None


# ── Calculadora principal ─────────────────────────────────────────────────────

def compute_metrics(data: dict, market_cap: Optional[float] = None,
                    shares: Optional[float] = None) -> dict:
    """
    Recebe o dict retornado por cvm_data.fetch_all() e retorna métricas
    estruturadas por período anual.

    market_cap e shares (em unidades, não milhares) são passados do yfinance
    para calcular múltiplos de mercado (P/L, P/VP, EV/EBITDA).
    """
    if not data or data.get("dre", pd.DataFrame()).empty:
        return {}

    dre = data["dre"]
    bpa = data["bpa"]
    bpp = data["bpp"]
    dfc = data["dfc"]

    dre_piv = _pivot_accounts(dre, ACCOUNT_MAP)
    bpa_piv = _pivot_accounts(bpa, ACCOUNT_MAP)
    bpp_piv = _pivot_accounts(bpp, ACCOUNT_MAP)
    dfc_piv = _pivot_accounts(dfc, ACCOUNT_MAP)

    if dre_piv.empty:
        return {}

    records = []

    for dt in dre_piv.index:
        rec = {"data": dt.strftime("%Y-%m-%d")}

        def g(piv, key):
            if piv.empty or key not in piv.columns:
                return None
            row = piv[piv.index <= dt]
            if row.empty:
                return None
            v = row[key].dropna()
            return float(v.iloc[-1]) if not v.empty else None

        # ── DRE ──────────────────────────────────────────────────────────────
        rec["receita_liquida"]   = g(dre_piv, "receita_liquida")
        rec["custo_bens"]        = g(dre_piv, "custo_bens_servicos")
        rec["resultado_bruto"]   = g(dre_piv, "resultado_bruto")
        rec["despesas_op"]       = g(dre_piv, "despesas_operacionais")
        rec["ebit"]              = g(dre_piv, "ebit")
        rec["resultado_fin"]     = g(dre_piv, "resultado_financeiro")
        rec["lucro_liquido"]     = g(dre_piv, "lucro_liquido")

        # Margens
        rl = rec["receita_liquida"]
        if rl and rl != 0:
            rec["margem_bruta"]   = (rec["resultado_bruto"] or 0) / rl
            rec["margem_ebit"]    = (rec["ebit"] or 0) / rl
            rec["margem_liquida"] = (rec["lucro_liquido"] or 0) / rl
        else:
            rec["margem_bruta"] = rec["margem_ebit"] = rec["margem_liquida"] = None

        # EBITDA proxy (EBIT + D&A) — D&A estimado via FCF quando disponível
        ebit = rec["ebit"]
        fco  = g(dfc_piv, "fcf_operacional")
        da_proxy = abs(fco - ebit) if (fco and ebit and fco > ebit) else None
        rec["ebitda"] = (ebit + da_proxy) if (ebit is not None and da_proxy) else ebit
        if rl and rl != 0 and rec["ebitda"]:
            rec["margem_ebitda"] = rec["ebitda"] / rl

        # ── Balanço ───────────────────────────────────────────────────────────
        rec["caixa"]            = g(bpa_piv, "caixa_equivalentes")
        rec["ativo_total"]      = g(bpa_piv, "ativo_total")
        rec["ativo_circ"]       = g(bpa_piv, "ativo_circulante")
        rec["passivo_circ"]     = g(bpp_piv, "passivo_circulante")
        rec["divida_cp"]        = g(bpp_piv, "emprestimos_cp")
        rec["divida_lp"]        = g(bpp_piv, "emprestimos_lp")
        rec["pl"]               = g(bpp_piv, "patrimonio_liquido")

        # Dívida bruta e líquida
        divida_bruta = (rec["divida_cp"] or 0) + (rec["divida_lp"] or 0)
        rec["divida_bruta"]   = divida_bruta if divida_bruta else None
        rec["divida_liquida"] = (divida_bruta - (rec["caixa"] or 0)) if divida_bruta else None

        # ── Fluxo de Caixa ────────────────────────────────────────────────────
        rec["fcf_operacional"]   = g(dfc_piv, "fcf_operacional")
        rec["fcf_investimento"]  = g(dfc_piv, "fcf_investimento")
        rec["fcf_financiamento"] = g(dfc_piv, "fcf_financiamento")
        # FCF Livre ≈ FCO + capex (investimento geralmente negativo)
        if rec["fcf_operacional"] and rec["fcf_investimento"]:
            rec["fcf_livre"] = rec["fcf_operacional"] + rec["fcf_investimento"]

        # ── Indicadores de crédito ────────────────────────────────────────────
        ebitda = rec.get("ebitda")
        dl     = rec.get("divida_liquida")
        ebit_v = rec.get("ebit")
        rf_v   = rec.get("resultado_fin")

        if ebitda and ebitda != 0:
            if dl is not None:
                rec["alavancagem"]       = dl / ebitda          # Dívida Líq/EBITDA
            if ebit_v and rf_v and rf_v != 0:
                rec["cobertura_juros"]   = ebit_v / abs(rf_v)   # EBIT / Desp. Fin.

        if rec["ativo_circ"] and rec["passivo_circ"] and rec["passivo_circ"] != 0:
            rec["liquidez_corrente"]     = rec["ativo_circ"] / rec["passivo_circ"]

        if rec["pl"] and rec["ativo_total"] and rec["ativo_total"] != 0:
            rec["roa"]   = (rec["lucro_liquido"] or 0) / rec["ativo_total"]
            rec["roe"]   = (rec["lucro_liquido"] or 0) / rec["pl"] if rec["pl"] != 0 else None

        # ── Múltiplos de mercado (requer market_cap) ──────────────────────────
        if market_cap:
            ev = market_cap + (dl or 0)
            rec["market_cap"]   = market_cap
            rec["ev"]           = ev
            if rec["lucro_liquido"] and rec["lucro_liquido"] > 0:
                rec["p_l"]      = market_cap / rec["lucro_liquido"]
            if rec["pl"] and rec["pl"] > 0:
                rec["p_vp"]     = market_cap / rec["pl"]
            if ebitda and ebitda > 0:
                rec["ev_ebitda"] = ev / ebitda
            if rl and rl > 0:
                rec["ev_receita"] = ev / rl

        records.append(rec)

    df_metrics = pd.DataFrame(records).set_index("data")
    return {
        "metrics_df":  df_metrics,
        "latest":      df_metrics.iloc[-1].to_dict() if not df_metrics.empty else {},
        "periods":     df_metrics.index.tolist(),
        "meta":        data.get("meta", {}),
    }


def get_fundamental_summary(ticker: str, years: int = 3,
                             market_cap: Optional[float] = None) -> dict:
    """
    Função de alto nível: recebe ticker (.SA) e retorna métricas completas.

    market_cap em R$ (ou USD para empresas internacionais) — se None, tenta
    buscar via yfinance automaticamente.
    """
    # Buscar dados CVM
    data = fetch_all(ticker, years=years)
    if not data:
        return {"error": f"Ticker '{ticker}' não encontrado no cadastro CVM."}

    # market_cap via yfinance se não fornecido
    if market_cap is None:
        try:
            import yfinance as yf
            info       = yf.Ticker(ticker).info
            market_cap = info.get("marketCap")
            shares     = info.get("sharesOutstanding")
        except Exception:
            shares = None
    else:
        shares = None

    metrics = compute_metrics(data, market_cap=market_cap, shares=shares)
    return metrics

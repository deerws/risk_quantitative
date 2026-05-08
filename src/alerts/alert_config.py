"""
Configuração e avaliação de alertas de risco.

Persiste limites configuráveis em JSON (data/alert_config.json).
Avalia alertas contra dados de retorno em tempo real.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "alert_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "portfolio": {
        "var_threshold":         0.05,   # VaR 95% diário > 5%
        "volatility_threshold":  0.30,   # Volatilidade anual > 30%
        "drawdown_threshold":    0.15,   # Drawdown atual > 15%
        "correlation_threshold": 0.80,   # Correlação média > 0.80
    },
    "assets": {},  # overrides por ticker: {"PETR4.SA": {"var_threshold": 0.07}}
}

_SEVERITY_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def load_config() -> dict:
    """Carrega config do JSON; retorna DEFAULT_CONFIG se arquivo não existir."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            cfg = {
                "portfolio": {**DEFAULT_CONFIG["portfolio"], **saved.get("portfolio", {})},
                "assets":    saved.get("assets", {}),
            }
            return cfg
        except Exception:
            pass
    return {"portfolio": dict(DEFAULT_CONFIG["portfolio"]), "assets": {}}


def save_config(config: dict) -> None:
    """Salva config no JSON (cria diretório se necessário)."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def evaluate_alerts(returns: pd.DataFrame, config: dict) -> list[dict]:
    """
    Avalia os limites configurados contra os retornos do portfólio e por ativo.

    Retorna lista de dicts com: type, severity, asset, message, value, threshold.
    Ordenada por severidade descendente.
    """
    alerts: list[dict] = []
    p = config.get("portfolio", DEFAULT_CONFIG["portfolio"])

    if returns.empty:
        return alerts

    port_ret = returns.mean(axis=1).dropna()
    if port_ret.empty:
        return alerts

    # ── Portfólio ────────────────────────────────────────────────────────────

    # VaR 95% (percentil 5 dos retornos diários)
    var_val = float(abs(np.percentile(port_ret, 5)))
    var_lim = float(p.get("var_threshold", 0.05))
    if var_val > var_lim:
        alerts.append({
            "type": "var", "asset": "Portfólio",
            "severity": "high" if var_val > var_lim * 1.5 else "medium",
            "message": f"VaR 95% do portfólio = {var_val:.2%} (limite: {var_lim:.2%})",
            "value": var_val, "threshold": var_lim,
        })

    # Volatilidade anualizada
    vol_val = float(port_ret.std() * np.sqrt(252))
    vol_lim = float(p.get("volatility_threshold", 0.30))
    if vol_val > vol_lim:
        alerts.append({
            "type": "volatility", "asset": "Portfólio",
            "severity": "high" if vol_val > vol_lim * 1.5 else "medium",
            "message": f"Volatilidade anual do portfólio = {vol_val:.2%} (limite: {vol_lim:.2%})",
            "value": vol_val, "threshold": vol_lim,
        })

    # Drawdown atual
    cum = (1 + port_ret).cumprod()
    rolling_max = cum.expanding().max()
    dd_val = float(abs((cum.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]))
    dd_lim = float(p.get("drawdown_threshold", 0.15))
    if dd_val > dd_lim:
        alerts.append({
            "type": "drawdown", "asset": "Portfólio",
            "severity": "critical" if dd_val > dd_lim * 1.5 else "high",
            "message": f"Drawdown atual do portfólio = {dd_val:.2%} (limite: {dd_lim:.2%})",
            "value": dd_val, "threshold": dd_lim,
        })

    # Correlação média (diversificação)
    if len(returns.columns) > 1:
        corr = returns.corr().values
        upper = corr[np.triu_indices(len(corr), k=1)]
        corr_val = float(upper.mean())
        corr_lim = float(p.get("correlation_threshold", 0.80))
        if corr_val > corr_lim:
            alerts.append({
                "type": "correlation", "asset": "Portfólio",
                "severity": "medium",
                "message": f"Correlação média = {corr_val:.2f} — diversificação comprometida (limite: {corr_lim:.2f})",
                "value": corr_val, "threshold": corr_lim,
            })

    # ── Por ativo ─────────────────────────────────────────────────────────────
    asset_overrides = config.get("assets", {})
    for asset in returns.columns:
        r = returns[asset].dropna()
        if r.empty:
            continue
        cfg_a = {**p, **asset_overrides.get(asset, {})}

        var_a = float(abs(np.percentile(r, 5)))
        var_lim_a = float(cfg_a.get("var_threshold", var_lim))
        if var_a > var_lim_a:
            alerts.append({
                "type": "var", "asset": asset,
                "severity": "medium",
                "message": f"VaR 95% = {var_a:.2%} (limite: {var_lim_a:.2%})",
                "value": var_a, "threshold": var_lim_a,
            })

        vol_a = float(r.std() * np.sqrt(252))
        vol_lim_a = float(cfg_a.get("volatility_threshold", vol_lim))
        if vol_a > vol_lim_a:
            alerts.append({
                "type": "volatility", "asset": asset,
                "severity": "medium",
                "message": f"Volatilidade anual = {vol_a:.2%} (limite: {vol_lim_a:.2%})",
                "value": vol_a, "threshold": vol_lim_a,
            })

    alerts.sort(key=lambda a: _SEVERITY_LEVELS.get(a["severity"], 0), reverse=True)
    return alerts


def alert_summary(alerts: list[dict]) -> dict:
    """Conta alertas por severidade."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for a in alerts:
        counts[a.get("severity", "low")] = counts.get(a.get("severity", "low"), 0) + 1
    return counts

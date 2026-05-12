"""
Scheduler de rotinas — APScheduler com consciência de horário B3.

Jobs agendados (horário de Brasília):
  - ETL diário           18:30  seg–sex  (após fechamento B3)
  - Relatório EOD        19:00  seg–sex  (após ETL)
  - Cache fundamentals   20:00  sex      (semanal, dados pesados)
  - Monitor intraday     a cada 15 min   10:00–17:30  seg–sex

Execução:
  python scripts/scheduler.py

Docker:
  command: python scripts/scheduler.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from dotenv import load_dotenv

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

BRT = ZoneInfo("America/Sao_Paulo")

# ── tickers padrão ───────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    t.strip()
    for t in os.getenv("DEFAULT_TICKERS", "PETR4.SA,VALE3.SA,ITUB4.SA,BBDC4.SA,WEGE3.SA").split(",")
    if t.strip()
]

VAR_THRESHOLD = float(os.getenv("VAR_ALERT_THRESHOLD", "0.05"))
DD_THRESHOLD = float(os.getenv("DRAWDOWN_ALERT_THRESHOLD", "0.15"))


# ── jobs ─────────────────────────────────────────────────────────────────────

def job_etl_daily():
    """Coleta preços EOD e persiste em data/processed/."""
    logger.info("▶ ETL diário iniciado")
    try:
        import yfinance as yf
        import pandas as pd

        end = datetime.now(BRT)
        prices = {}
        for ticker in DEFAULT_TICKERS:
            try:
                df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
                if not df.empty:
                    prices[ticker] = df["Close"]
            except Exception as e:
                logger.warning(f"  {ticker}: {e}")

        if prices:
            combined = pd.DataFrame(prices)
            out = ROOT / "data" / "processed" / "prices_eod.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(out)
            logger.info(f"✓ ETL concluído — {len(prices)} ativos → {out}")
        else:
            logger.warning("ETL: nenhum dado coletado")
    except Exception as e:
        logger.error(f"ETL falhou: {e}", exc_info=True)
        raise


def job_risk_check():
    """Calcula VaR/drawdown e emite alertas se thresholds forem excedidos."""
    logger.info("▶ Verificação de risco iniciada")
    try:
        import pandas as pd
        import numpy as np
        from scripts.notifier import Notifier

        prices_path = ROOT / "data" / "processed" / "prices_eod.parquet"
        if not prices_path.exists():
            logger.warning("risk_check: prices_eod.parquet não encontrado — rode ETL primeiro")
            return

        prices = pd.read_parquet(prices_path)
        returns = prices.pct_change().dropna()
        notifier = Notifier()

        for ticker in returns.columns:
            ret = returns[ticker].dropna()
            if len(ret) < 20:
                continue

            var_95 = float(np.percentile(ret, 5))
            cum = (1 + ret).cumprod()
            drawdown = float((cum / cum.cummax() - 1).min())

            if abs(var_95) > VAR_THRESHOLD:
                detail = f"VaR 95% diário = {var_95:.2%} (limite: {VAR_THRESHOLD:.2%})"
                logger.warning(f"ALERTA VaR — {ticker}: {detail}")
                notifier.notify_alert(ticker, "VaR excedido", detail)

            if abs(drawdown) > DD_THRESHOLD:
                detail = f"Max Drawdown = {drawdown:.2%} (limite: {DD_THRESHOLD:.2%})"
                logger.warning(f"ALERTA Drawdown — {ticker}: {detail}")
                notifier.notify_alert(ticker, "Drawdown elevado", detail)

        logger.info("✓ Verificação de risco concluída")
    except Exception as e:
        logger.error(f"risk_check falhou: {e}", exc_info=True)
        raise


def job_daily_report():
    """Envia relatório EOD por email com PDF em anexo (se existir)."""
    logger.info("▶ Relatório EOD sendo enviado")
    try:
        from scripts.notifier import Notifier

        report_candidates = sorted(
            (ROOT / "reports").glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        report_path = str(report_candidates[0]) if report_candidates else None

        Notifier().notify_daily_report(report_path)
        logger.info("✓ Relatório EOD enviado")
    except Exception as e:
        logger.error(f"Relatório EOD falhou: {e}", exc_info=True)


def job_refresh_fundamentals():
    """Pré-aquece cache de fundamentals (pesado — roda só na sexta)."""
    logger.info("▶ Refresh de fundamentals iniciado")
    try:
        import yfinance as yf

        for ticker in DEFAULT_TICKERS:
            try:
                t = yf.Ticker(ticker)
                _ = t.info
                _ = t.financials
                _ = t.balance_sheet
                logger.info(f"  {ticker}: cache atualizado")
            except Exception as e:
                logger.warning(f"  {ticker}: {e}")

        logger.info("✓ Fundamentals atualizados")
    except Exception as e:
        logger.error(f"refresh_fundamentals falhou: {e}", exc_info=True)


def job_intraday_monitor():
    """Monitora preços intraday durante pregão e alerta movimentos > 3%."""
    now = datetime.now(BRT)
    # Só executa em dias úteis entre 10h e 17h30
    if now.weekday() >= 5:
        return
    if not (10 <= now.hour < 17 or (now.hour == 17 and now.minute <= 30)):
        return

    logger.info("▶ Monitor intraday")
    try:
        import yfinance as yf
        from scripts.notifier import Notifier

        notifier = Notifier()
        for ticker in DEFAULT_TICKERS:
            try:
                df = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=True)
                if df.empty or len(df) < 2:
                    continue
                open_price = float(df["Close"].iloc[0])
                last_price = float(df["Close"].iloc[-1])
                change = (last_price - open_price) / open_price
                if abs(change) >= 0.03:
                    direction = "↑" if change > 0 else "↓"
                    detail = f"Variação intraday {direction} {change:+.2%} (abertura: {open_price:.2f} → atual: {last_price:.2f})"
                    logger.info(f"INTRADAY {ticker}: {detail}")
                    notifier.notify_alert(ticker, f"Movimento intraday {direction}", detail)
            except Exception as e:
                logger.debug(f"  {ticker} intraday: {e}")
    except Exception as e:
        logger.error(f"intraday_monitor falhou: {e}", exc_info=True)


# ── event listener ───────────────────────────────────────────────────────────

def _on_job_event(event):
    if event.exception:
        logger.error(f"Job {event.job_id} falhou: {event.exception}")
    else:
        logger.debug(f"Job {event.job_id} concluído")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    scheduler = BlockingScheduler(timezone=BRT)
    scheduler.add_listener(_on_job_event, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    # ETL pós-fechamento B3 (18:30 BRT, seg–sex)
    scheduler.add_job(
        job_etl_daily,
        "cron",
        hour=18, minute=30,
        day_of_week="mon-fri",
        id="etl_daily",
        name="ETL Diário (pós-fechamento B3)",
        misfire_grace_time=600,
        coalesce=True,
    )

    # Verificação de risco logo após ETL (18:45 BRT)
    scheduler.add_job(
        job_risk_check,
        "cron",
        hour=18, minute=45,
        day_of_week="mon-fri",
        id="risk_check",
        name="Verificação VaR/Drawdown",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Relatório EOD por email (19:00 BRT, seg–sex)
    scheduler.add_job(
        job_daily_report,
        "cron",
        hour=19, minute=0,
        day_of_week="mon-fri",
        id="daily_report",
        name="Relatório EOD por Email",
        misfire_grace_time=600,
        coalesce=True,
    )

    # Cache de fundamentals — toda sexta às 20h
    scheduler.add_job(
        job_refresh_fundamentals,
        "cron",
        hour=20, minute=0,
        day_of_week="fri",
        id="fundamentals_refresh",
        name="Refresh Semanal de Fundamentals",
        misfire_grace_time=1800,
        coalesce=True,
    )

    # Monitor intraday — a cada 15 min (filtra horário internamente)
    scheduler.add_job(
        job_intraday_monitor,
        "cron",
        minute="*/15",
        id="intraday_monitor",
        name="Monitor Intraday B3",
        misfire_grace_time=60,
        coalesce=True,
    )

    logger.info("Scheduler iniciado. Jobs registrados:")
    for job in scheduler.get_jobs():
        logger.info(f"  [{job.id}] {job.name}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler encerrado.")


if __name__ == "__main__":
    main()

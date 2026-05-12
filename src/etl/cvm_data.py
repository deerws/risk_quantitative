"""
ETL para dados fundamentais de empresas B3 via CVM Dados Abertos.

Fonte oficial: https://dados.cvm.gov.br/dados/CIA_ABERTA/
Cobertura: 100% das empresas de capital aberto listadas na B3.
Atualização: trimestral (ITR) e anual (DFP).

Fluxo:
  ticker (.SA) → nome via yfinance → fuzzy match no cadastro CVM → CD_CVM
  CD_CVM → DRE / BPA / BPP / DFC filtrados localmente (arquivos cacheados)
"""

import io
import re
import zipfile
import difflib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import numpy as np

# ── Constantes ────────────────────────────────────────────────────────────────

CVM_BASE  = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
CAD_URL   = f"{CVM_BASE}/CAD/DADOS/cad_cia_aberta.csv"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cvm_cache"
MAP_FILE  = CACHE_DIR / "ticker_cdcvm_map.json"   # cache de resoluções ticker→CD_CVM

_HEADERS  = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT  = 30   # segundos

# Conta corrente nos arquivos CVM (código → descrição semântica)
ACCOUNT_MAP = {
    # DRE
    "3.01":  "receita_liquida",
    "3.02":  "custo_bens_servicos",
    "3.03":  "resultado_bruto",
    "3.04":  "despesas_operacionais",
    "3.05":  "ebit",                        # EBIT = Resultado antes do fin. e tributos
    "3.06":  "resultado_financeiro",
    "3.07":  "resultado_antes_tributos",
    "3.08":  "imposto_renda",
    "3.11":  "lucro_liquido",
    # BPA — Ativo
    "1":     "ativo_total",
    "1.01":  "ativo_circulante",
    "1.01.01": "caixa_equivalentes",
    "1.02":  "ativo_nao_circulante",
    # BPP — Passivo + PL
    "2":     "passivo_total",
    "2.01":  "passivo_circulante",
    "2.01.04": "emprestimos_cp",            # dívida de curto prazo
    "2.02":  "passivo_nao_circulante",
    "2.02.01": "emprestimos_lp",            # dívida de longo prazo
    "2.03":  "patrimonio_liquido",
    # DFC (método indireto)
    "6.01":  "fcf_operacional",
    "6.02":  "fcf_investimento",
    "6.03":  "fcf_financiamento",
}


# ── Helpers internos ──────────────────────────────────────────────────────────

def _ensure_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_fresh(path: Path, max_days: int) -> bool:
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    return age < max_days


def _download(url: str) -> bytes:
    r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.content


def _read_cvm_csv(content: bytes, csv_name: str = "") -> pd.DataFrame:
    """
    Lê CSV da CVM (sep=';', encoding ISO-8859-1).
    Se content for ZIP, extrai csv_name (ou o primeiro arquivo se csv_name='').
    """
    if content[:2] == b"PK":
        zf = zipfile.ZipFile(io.BytesIO(content))
        names = zf.namelist()
        target = csv_name if csv_name and csv_name in names else names[0]
        with zf.open(target) as f:
            raw = f.read()
    else:
        raw = content
    return pd.read_csv(io.BytesIO(raw), sep=";", encoding="iso-8859-1",
                       dtype=str, low_memory=False)


def _normalize_name(name: str) -> str:
    """Remove sufixos societários e normaliza para comparação."""
    name = name.upper()
    for suf in [" S.A.", " SA ", " S/A", " S.A", " LTDA", " INC.", " CORP.",
                " GROUP", " HOLDING", " PARTICIPACOES", " PARTICIPAÇÕES"]:
        name = name.replace(suf, " ")
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


# ── Cadastro de empresas ──────────────────────────────────────────────────────

def fetch_cadastro(use_cache: bool = True) -> pd.DataFrame:
    """
    Baixa o cadastro de todas as empresas abertas da CVM.
    Cache de 7 dias em data/cvm_cache/cad_cia_aberta.parquet.

    Colunas relevantes: CD_CVM, CNPJ_CIA, DENOM_SOCIAL, DENOM_COMERC, SIT
    """
    _ensure_cache()
    cache = CACHE_DIR / "cad_cia_aberta.parquet"

    if use_cache and _cache_fresh(cache, max_days=7):
        return pd.read_parquet(cache)

    try:
        raw = _download(CAD_URL)
        df  = _read_cvm_csv(raw)
        df.to_parquet(cache)
        return df
    except Exception as e:
        if cache.exists():
            return pd.read_parquet(cache)
        raise RuntimeError(f"Falha ao baixar cadastro CVM: {e}") from e


def resolve_ticker(ticker: str) -> Optional[dict]:
    """
    Resolve um ticker yfinance (ex: 'PETR4.SA') para CD_CVM, CNPJ e nome oficial.

    Estratégia:
      1. Consulta cache local (ticker_cdcvm_map.json)
      2. Busca nome da empresa via yfinance
      3. Fuzzy match contra DENOM_SOCIAL / DENOM_COMERC do cadastro CVM
      4. Persiste resultado no cache

    Retorna dict com: cd_cvm, cnpj, nome_cvm, ticker | None se não encontrado.
    """
    _ensure_cache()

    # 1. Cache local de mapeamentos já resolvidos
    resolved: dict = {}
    if MAP_FILE.exists():
        resolved = json.loads(MAP_FILE.read_text())
    if ticker in resolved:
        return resolved[ticker]

    # 2. Nome da empresa via yfinance
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        yf_name = info.get("longName") or info.get("shortName") or ""
    except Exception:
        yf_name = ""

    if not yf_name:
        return None

    # 3. Cadastro CVM e fuzzy match
    try:
        cad = fetch_cadastro()
    except Exception:
        return None

    # Manter apenas empresas ativas
    if "SIT" in cad.columns:
        cad = cad[cad["SIT"].str.strip().isin(["ATIVO", "FASE PRÉ-OPERACIONAL"])]

    # Coluna de nome: preferir DENOM_COMERC, fallback DENOM_SOCIAL
    name_col = "DENOM_COMERC" if "DENOM_COMERC" in cad.columns else "DENOM_SOCIAL"
    cad      = cad.dropna(subset=[name_col, "CD_CVM"])

    yf_norm  = _normalize_name(yf_name)
    cvm_norms = cad[name_col].apply(_normalize_name).tolist()

    matches = difflib.get_close_matches(yf_norm, cvm_norms, n=1, cutoff=0.55)
    if not matches:
        # Segunda tentativa com DENOM_SOCIAL
        if name_col != "DENOM_SOCIAL" and "DENOM_SOCIAL" in cad.columns:
            soc_norms = cad["DENOM_SOCIAL"].apply(_normalize_name).tolist()
            matches   = difflib.get_close_matches(yf_norm, soc_norms, n=1, cutoff=0.55)
            if matches:
                idx = soc_norms.index(matches[0])
            else:
                return None
        else:
            return None
    else:
        idx = cvm_norms.index(matches[0])

    row    = cad.iloc[idx]
    result = {
        "ticker":   ticker,
        "cd_cvm":   int(str(row["CD_CVM"]).strip()),
        "cnpj":     str(row.get("CNPJ_CIA", "")).strip(),
        "nome_cvm": str(row.get("DENOM_SOCIAL", "")).strip(),
        "yf_name":  yf_name,
    }

    # 4. Persistir no cache
    resolved[ticker] = result
    MAP_FILE.write_text(json.dumps(resolved, ensure_ascii=False, indent=2))
    return result


# ── Download de demonstrativos financeiros ────────────────────────────────────

def _fetch_dfp_doc(doc_type: str, years: list[int]) -> pd.DataFrame:
    """
    Baixa e concatena demonstrativos DFP (anuais) para os anos solicitados.
    doc_type: 'DRE', 'BPA', 'BPP', 'DFC_MI'

    URL correta: dfp_cia_aberta_{year}.zip (um ZIP por ano com todos os CSVs dentro).
    CSV alvo dentro do ZIP: dfp_cia_aberta_{doc_type}_con_{year}.csv
    Cache por tipo+ano em parquet; anos fechados nunca expiram.
    """
    _ensure_cache()
    frames = []
    current_year = date.today().year

    for year in years:
        cache_file = CACHE_DIR / f"dfp_{doc_type}_{year}.parquet"
        is_closed  = year < current_year

        if is_closed and cache_file.exists():
            frames.append(pd.read_parquet(cache_file))
            continue

        if not is_closed and _cache_fresh(cache_file, max_days=30):
            frames.append(pd.read_parquet(cache_file))
            continue

        url      = f"{CVM_BASE}/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
        csv_name = f"dfp_cia_aberta_{doc_type}_con_{year}.csv"
        try:
            raw = _download(url)
            df  = _read_cvm_csv(raw, csv_name)
            if df.empty:
                continue
            frames.append(df)
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
        except Exception:
            if cache_file.exists():
                frames.append(pd.read_parquet(cache_file))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_itr_doc(doc_type: str, year: int) -> pd.DataFrame:
    """
    Baixa demonstrativo ITR (trimestral) de um ano.
    doc_type: 'DRE', 'BPA', 'BPP', 'DFC_MI'

    URL correta: itr_cia_aberta_{year}.zip (um ZIP por ano).
    CSV alvo dentro do ZIP: itr_cia_aberta_{doc_type}_con_{year}.csv
    """
    _ensure_cache()
    cache_file = CACHE_DIR / f"itr_{doc_type}_{year}.parquet"

    if _cache_fresh(cache_file, max_days=15):
        return pd.read_parquet(cache_file)

    url      = f"{CVM_BASE}/DOC/ITR/DADOS/itr_cia_aberta_{year}.zip"
    csv_name = f"itr_cia_aberta_{doc_type}_con_{year}.csv"
    try:
        raw = _download(url)
        df  = _read_cvm_csv(raw, csv_name)
        if not df.empty:
            df.to_parquet(cache_file)
        return df
    except Exception:
        return pd.read_parquet(cache_file) if cache_file.exists() else pd.DataFrame()


def _filter_company(df: pd.DataFrame, cd_cvm: int) -> pd.DataFrame:
    if df.empty or "CD_CVM" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["CD_CVM"] = pd.to_numeric(df["CD_CVM"], errors="coerce")
    return df[df["CD_CVM"] == cd_cvm].copy()


# ── API pública ───────────────────────────────────────────────────────────────

def fetch_dre(cd_cvm: int, years: int = 3) -> pd.DataFrame:
    """DRE consolidada dos últimos `years` anos (DFP anual)."""
    year_list = list(range(date.today().year - years, date.today().year + 1))
    raw = _fetch_dfp_doc("DRE", year_list)
    return _filter_company(raw, cd_cvm)


def fetch_bpa(cd_cvm: int, years: int = 3) -> pd.DataFrame:
    """Balanço Patrimonial — Ativo (DFP anual)."""
    year_list = list(range(date.today().year - years, date.today().year + 1))
    raw = _fetch_dfp_doc("BPA", year_list)
    return _filter_company(raw, cd_cvm)


def fetch_bpp(cd_cvm: int, years: int = 3) -> pd.DataFrame:
    """Balanço Patrimonial — Passivo + PL (DFP anual)."""
    year_list = list(range(date.today().year - years, date.today().year + 1))
    raw = _fetch_dfp_doc("BPP", year_list)
    return _filter_company(raw, cd_cvm)


def fetch_dfc(cd_cvm: int, years: int = 3) -> pd.DataFrame:
    """Demonstração de Fluxo de Caixa — método indireto (DFP anual)."""
    year_list = list(range(date.today().year - years, date.today().year + 1))
    raw = _fetch_dfp_doc("DFC_MI", year_list)
    return _filter_company(raw, cd_cvm)


def fetch_dre_trimestral(cd_cvm: int) -> pd.DataFrame:
    """DRE do ITR do ano corrente (últimos 4 trimestres disponíveis)."""
    current = date.today().year
    frames  = []
    for y in [current - 1, current]:
        df = _fetch_itr_doc("DRE", y)
        fc = _filter_company(df, cd_cvm)
        if not fc.empty:
            frames.append(fc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_all(ticker: str, years: int = 3) -> dict:
    """
    Entry point principal: dado um ticker (.SA), retorna todos os
    demonstrativos disponíveis + metadados da empresa.

    Retorna dict com chaves: meta, dre, bpa, bpp, dfc, dre_trimestral
    Retorna None se o ticker não for resolvido.
    """
    meta = resolve_ticker(ticker)
    if meta is None:
        return {}

    cd = meta["cd_cvm"]
    return {
        "meta":           meta,
        "dre":            fetch_dre(cd, years),
        "bpa":            fetch_bpa(cd, years),
        "bpp":            fetch_bpp(cd, years),
        "dfc":            fetch_dfc(cd, years),
        "dre_trimestral": fetch_dre_trimestral(cd),
    }

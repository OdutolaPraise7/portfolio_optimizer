import re
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
PRICE_FILE = PROJECT_ROOT / "PRICE_LIST.csv"
SIGNAL_FILE = PROJECT_ROOT / "signal_store.csv"
STALE_SIGNAL_HOURS = 24 * 7
ENFORCE_SIGNAL_FRESHNESS = os.getenv("ENFORCE_SIGNAL_FRESHNESS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

PRICE_SNAPSHOT_LOCK = Lock()
SIGNAL_SNAPSHOT_LOCK = Lock()

NON_EQUITY_KEYWORDS = [
    "FGS", "FG1", "FG2", "FG6", "FG9", "FGB", "LAB", "DIF", "IAO", "UCAP2",
    "ZAM", "FCM2", "FBQ", "EPF", "CSF", "LASUK", "FGSUK", "FHSUK", "FID2",
    "TAJ", "LAFARGEWAPCO", "CHAPELHILL", "IBTCINFRA", "ETF", "LOTUS",
    "NEWGOLD", "NGX30", "NGX50", "STANBICETF", "SIAMLETF", "GREENWETF",
    "VETGRIF", "VETINDETF", "VSPBOND", "NGXAFR", "LOTUSHAL", "REIT",
    "UHOMREIT", "SFSREIT", "UPDCREIT", "UHOREIT", "NIGFUND", "IBTCNEF",
    "NESF", "NIDF", "MERGROWTH", "MERVALUE", "NGXMERI", "NGXPENSION",
]
NON_EQUITY_PATTERN = re.compile("|".join(re.escape(keyword) for keyword in NON_EQUITY_KEYWORDS))


class SnapshotError(Exception):
    """Raised when a lightweight API snapshot cannot be loaded."""


def _file_signature(path: Path) -> Tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sector_for_symbol(symbol: str) -> str:
    banking = {"ACCESSCORP", "ETI", "FCMB", "FBNH", "FIDELITYBK", "GTCO", "JAIZBANK", "STANBIC", "UBA", "WEMABANK", "ZENITHBANK"}
    consumer = {"BUAFOODS", "CADBURY", "DANGSUGAR", "FLOURMILL", "GUINNESS", "NASCON", "NB", "NESTLE", "PZ", "UNILEVER"}
    industrial = {"BERGER", "BETAGLAS", "BUACEMENT", "CAP", "CUTIX", "DANGCEM", "WAPCO"}
    oil_gas = {"ARDOVA", "CONOIL", "ETERNA", "MOBIL", "MRS", "OANDO", "SEPLAT", "TOTAL"}
    insurance = {"AIICO", "CUSTODIAN", "LASACO", "MANSARD", "NEM", "PRESTIGE", "WAPIC"}
    telecoms = {"AIRTELAFRI", "MTNN"}
    symbol = str(symbol).upper()
    if symbol in banking:
        return "Banking"
    if symbol in consumer:
        return "Consumer Goods"
    if symbol in industrial:
        return "Industrial Goods"
    if symbol in oil_gas:
        return "Oil and Gas"
    if symbol in insurance:
        return "Insurance"
    if symbol in telecoms:
        return "Telecommunications"
    return "Other"


@lru_cache(maxsize=4)
def _load_market_snapshot_cached(resolved_path: str, mtime_ns: int, size: int) -> Dict[str, object]:
    import pandas as pd

    df = pd.read_csv(
        resolved_path,
        usecols=["SYMBOL", "TRANS_DATE", "CLOSE_PRICE"],
        low_memory=False,
    )
    df.columns = [column.replace("\ufeff", "") for column in df.columns]
    missing = {"SYMBOL", "TRANS_DATE", "CLOSE_PRICE"} - set(df.columns)
    if missing:
        raise SnapshotError(f"Price file missing required columns: {sorted(missing)}")

    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df = df[~df["SYMBOL"].str.contains(NON_EQUITY_PATTERN, regex=True, na=False)].copy()
    df["CLOSE_PRICE"] = pd.to_numeric(df["CLOSE_PRICE"], errors="coerce")
    df = df.dropna(subset=["TRANS_DATE", "CLOSE_PRICE"])
    df = df[df["CLOSE_PRICE"] > 0]

    counts = df.groupby("SYMBOL").size().sort_values(ascending=False)
    latest_indexes = df.groupby("SYMBOL")["TRANS_DATE"].idxmax()
    latest = df.loc[latest_indexes].set_index("SYMBOL")["CLOSE_PRICE"].astype(float)
    latest_date = str(df["TRANS_DATE"].max()).replace(" ", "T")

    return {
        "symbols": counts.index.tolist(),
        "prices": {symbol: round(float(price), 4) for symbol, price in latest.items()},
        "updated_at": latest_date if latest_date and latest_date.lower() != "nan" else None,
    }


def get_market_snapshot(price_file: str | Path = PRICE_FILE) -> Dict[str, object]:
    path = Path(price_file)
    if not path.exists():
        raise SnapshotError(f"Price file not found: {price_file}")

    signature = _file_signature(path)
    with PRICE_SNAPSHOT_LOCK:
        snapshot = _load_market_snapshot_cached(*signature)

    return {
        "symbols": list(snapshot["symbols"]),
        "prices": dict(snapshot["prices"]),
        "updated_at": snapshot["updated_at"],
    }


def get_supported_symbols(price_file: str | Path = PRICE_FILE) -> List[str]:
    return list(get_market_snapshot(price_file)["symbols"])


def get_latest_price_snapshot(price_file: str | Path = PRICE_FILE) -> Dict[str, object]:
    snapshot = get_market_snapshot(price_file)
    return {
        "prices": snapshot["prices"],
        "updated_at": snapshot["updated_at"],
    }


def _load_signal_rows(signal_file: str | Path, stale_after_hours: int) -> Tuple[List[Dict[str, str]], str, bool, float]:
    path = Path(signal_file)
    if not path.exists():
        raise SnapshotError(f"Signal store not found at '{signal_file}'. Run merge_signals.py first.")

    modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    age_hours = (datetime.now() - modified_at).total_seconds() / 3600
    is_stale = age_hours > stale_after_hours
    if ENFORCE_SIGNAL_FRESHNESS and is_stale:
        raise SnapshotError(
            f"Signal store is stale ({age_hours:.1f}h old). Refresh it by running merge_signals.py."
        )

    signature = _file_signature(path)
    with SIGNAL_SNAPSHOT_LOCK:
        rows = _load_signal_rows_cached(*signature)

    return [dict(row) for row in rows], modified_at.isoformat(), is_stale, age_hours


@lru_cache(maxsize=4)
def _load_signal_rows_cached(resolved_path: str, mtime_ns: int, size: int) -> Tuple[Dict[str, str], ...]:
    import csv

    with Path(resolved_path).open(newline="", encoding="utf-8-sig") as file:
        return tuple(csv.DictReader(file))


def _avg_return(row: Dict[str, str]) -> float:
    values = []
    for prefix in ["XGB", "RF", "LSTM"]:
        quality_value = str(row.get(f"{prefix}_Quality_Pass", "")).strip().lower()
        if quality_value and quality_value not in {"true", "1", "yes"}:
            continue
        if row.get(f"{prefix}_Return (%)") not in {None, ""}:
            values.append(_coerce_float(row.get(f"{prefix}_Return (%)")))
    if not values and row.get("Predicted_Return (%)") not in {None, ""}:
        values.append(_coerce_float(row.get("Predicted_Return (%)")))
    return sum(values) / len(values) if values else 0.0


def _signal_score(row: Dict[str, str]) -> float:
    avg_confidence = _coerce_float(row.get("Avg_Confidence"))
    avg_r2 = max(_coerce_float(row.get("Avg_Quality_R2") or row.get("Avg_R2")), 0.0)
    avg_return = max(_avg_return(row), 0.0)
    tier = min(max(_coerce_int(row.get("Consensus_Tier"), 3), 1), 3)
    return avg_confidence * 0.45 + avg_r2 * 0.20 + (avg_return / 100.0) * 0.25 + ((4 - tier) / 10.0) * 0.10


def get_signal_summary(
    signal_file: str | Path = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
) -> Dict[str, object]:
    rows, generated_at, is_stale, age_hours = _load_signal_rows(signal_file, stale_after_hours)
    signals = [str(row.get("Consensus_Signal", "")).upper() for row in rows]
    confidences = [_coerce_float(row.get("Avg_Confidence")) for row in rows]
    r2_values = [_coerce_float(row.get("Avg_R2")) for row in rows]
    return {
        "path": str(Path(signal_file).resolve()),
        "row_count": len(rows),
        "generated_at": generated_at,
        "buy_count": signals.count("BUY"),
        "sell_count": signals.count("SELL"),
        "conflict_count": signals.count("CONFLICT"),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "avg_r2": round(sum(r2_values) / len(r2_values), 4) if r2_values else 0.0,
        "is_stale": is_stale,
        "age_hours": round(age_hours, 1),
    }


def get_signal_watchlist(
    signal_file: str | Path = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
    limit: int = 5,
) -> Dict[str, object]:
    rows, _, _, _ = _load_signal_rows(signal_file, stale_after_hours)

    def serialize(row: Dict[str, str]) -> Dict[str, object]:
        symbol = str(row.get("Symbol", "")).upper()
        signal = str(row.get("Consensus_Signal", "UNKNOWN")).upper()
        avg_return = _avg_return(row)
        avg_confidence = _coerce_float(row.get("Avg_Confidence"))
        return {
            "symbol": symbol,
            "signal": signal,
            "sector": _sector_for_symbol(symbol),
            "avg_return": round(avg_return, 4),
            "avg_confidence": round(avg_confidence, 4),
            "avg_r2": round(_coerce_float(row.get("Avg_R2")), 4),
            "signal_score": round(_signal_score(row), 6),
            "reason": (
                f"{signal} consensus with {avg_return:.2f}% expected return and "
                f"{avg_confidence * 100:.0f}% average confidence."
            ),
        }

    scored = [(row, _signal_score(row), _avg_return(row)) for row in rows]
    top_buys = sorted(
        [item for item in scored if str(item[0].get("Consensus_Signal", "")).upper() == "BUY"],
        key=lambda item: (item[1], item[2], _coerce_float(item[0].get("Avg_Confidence"))),
        reverse=True,
    )[:limit]
    top_sells = sorted(
        [item for item in scored if str(item[0].get("Consensus_Signal", "")).upper() == "SELL"],
        key=lambda item: (item[2], -_coerce_float(item[0].get("Avg_Confidence"))),
    )[:limit]
    return {
        "top_buys": [serialize(row) for row, _, _ in top_buys],
        "top_sells": [serialize(row) for row, _, _ in top_sells],
    }

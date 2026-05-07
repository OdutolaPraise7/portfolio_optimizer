import math
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PRICE_FILE = str(PROJECT_ROOT / "PRICE_LIST.csv")
SIGNAL_FILE = str(PROJECT_ROOT / "signal_store.csv")
STALE_SIGNAL_HOURS = 24 * 7
PRICE_CACHE_LOCK = Lock()
SIGNAL_CACHE_LOCK = Lock()
MIN_SIGNAL_CONFIDENCE = 0.02
MIN_HISTORY_DAYS = 90
TRADING_DAYS = 252
RANDOM_PORTFOLIO_SAMPLES = 5000
OPTIMIZER_SEED = 42
BACKTEST_LOOKBACK_DAYS = 252
RISK_FREE_RATE = 0.02

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

SECTOR_MAP = {
    "ABBEYBDS": "Financial Services",
    "ACCESSCORP": "Banking",
    "AFRIPRUD": "Financial Services",
    "AIICO": "Insurance",
    "AIRTELAFRI": "Telecommunications",
    "ARDOVA": "Oil and Gas",
    "BERGER": "Industrial Goods",
    "BETAGLAS": "Industrial Goods",
    "BOCGAS": "Industrial Goods",
    "BUACEMENT": "Industrial Goods",
    "BUAFOODS": "Consumer Goods",
    "CADBURY": "Consumer Goods",
    "CAP": "Industrial Goods",
    "CAVERTON": "Services",
    "CHAMPION": "Consumer Goods",
    "CHAMS": "ICT",
    "CONOIL": "Oil and Gas",
    "COURTVILLE": "ICT",
    "CUSTODIAN": "Insurance",
    "CUTIX": "Industrial Goods",
    "CWG": "ICT",
    "DANGCEM": "Industrial Goods",
    "DANGSUGAR": "Consumer Goods",
    "ETERNA": "Oil and Gas",
    "ETI": "Banking",
    "FCMB": "Banking",
    "FBNH": "Banking",
    "FIDELITYBK": "Banking",
    "FIDSON": "Healthcare",
    "FLOURMILL": "Consumer Goods",
    "FTNCOCOA": "Agriculture",
    "GLAXOSMITH": "Healthcare",
    "GTCO": "Banking",
    "GUINNESS": "Consumer Goods",
    "IKEJAHOTEL": "Services",
    "INTBREW": "Consumer Goods",
    "INTENEGINS": "Industrial Goods",
    "JAIZBANK": "Banking",
    "JAPAULGOLD": "Mining",
    "JAPAULOIL": "Oil and Gas",
    "JBERGER": "Construction",
    "JOHNHOLT": "Conglomerates",
    "LASACO": "Insurance",
    "LAWUNION": "Insurance",
    "LEARNAFRCA": "Services",
    "LINKASSURE": "Insurance",
    "LIVESTOCK": "Agriculture",
    "MANSARD": "Insurance",
    "MAYBAKER": "Healthcare",
    "MBENEFIT": "Insurance",
    "MOBIL": "Oil and Gas",
    "MORISON": "Healthcare",
    "MRS": "Oil and Gas",
    "MTNN": "Telecommunications",
    "MULTIVERSE": "Industrial Goods",
    "NAHCO": "Services",
    "NASCON": "Consumer Goods",
    "NB": "Consumer Goods",
    "NEM": "Insurance",
    "NESTLE": "Consumer Goods",
    "NGXGROUP": "Financial Services",
    "NIGERINS": "Insurance",
    "NNFM": "Consumer Goods",
    "NSLTECH": "ICT",
    "OANDO": "Oil and Gas",
    "OMATEK": "ICT",
    "PHARMDEKO": "Healthcare",
    "PRESCO": "Agriculture",
    "PRESTIGE": "Insurance",
    "PZ": "Consumer Goods",
    "REDSTAREX": "Services",
    "REGALINS": "Insurance",
    "ROYALEX": "Insurance",
    "RTBRISCOE": "Services",
    "SCOA": "Conglomerates",
    "SEPLAT": "Oil and Gas",
    "SKYAVN": "Services",
    "SOVRENINS": "Insurance",
    "STANBIC": "Banking",
    "STDINSURE": "Insurance",
    "STERLINGNG": "Banking",
    "STERLNBANK": "Banking",
    "SUNUASSUR": "Insurance",
    "TOTAL": "Oil and Gas",
    "TOURIST": "Services",
    "TRANSCORP": "Conglomerates",
    "TRANSCOHOT": "Services",
    "TRANSEXPR": "Services",
    "UACN": "Conglomerates",
    "UBA": "Banking",
    "UBN": "Banking",
    "UCAP": "Financial Services",
    "UNILEVER": "Consumer Goods",
    "UNITYBNK": "Banking",
    "UPDC": "Real Estate",
    "UPL": "Industrial Goods",
    "VFDGROUP": "Financial Services",
    "VITAFOAM": "Consumer Goods",
    "WAPCO": "Industrial Goods",
    "WAPIC": "Insurance",
    "WEMABANK": "Banking",
    "ZENITHBANK": "Banking",
}

REBALANCE_STEP_MAP = {
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
}


class PortfolioOptimiserError(Exception):
    """Base optimizer error."""


class ValidationError(PortfolioOptimiserError):
    """User input or data contract error."""


class SignalStoreError(PortfolioOptimiserError):
    """Signal store missing or stale."""


@dataclass(frozen=True)
class RiskProfileConfig:
    max_weight: float
    max_sector_weight: float
    signal_strength: float
    allow_exit_threshold: float
    min_weight_floor: float
    new_stock_budget: float
    turnover_penalty: float
    transaction_cost_rate: float
    no_trade_band: float
    cvar_penalty: float
    benchmark_tilt: float


@dataclass(frozen=True)
class MandateProfileConfig:
    label: str
    objective: str
    benchmark: str
    max_stock_weight: float | None = None
    max_sector_weight: float | None = None
    min_liquidity_score: float = 0.35
    max_turnover: float = 0.45
    max_portfolio_volatility: float | None = None
    max_new_stock_budget: float | None = None
    min_buy_confidence: float = MIN_SIGNAL_CONFIDENCE


RISK_PROFILE_CONFIG: Dict[str, RiskProfileConfig] = {
    "conservative": RiskProfileConfig(
        max_weight=0.08,
        max_sector_weight=0.35,
        signal_strength=0.65,
        allow_exit_threshold=0.015,
        min_weight_floor=0.02,
        new_stock_budget=0.20,
        turnover_penalty=0.45,
        transaction_cost_rate=0.0025,
        no_trade_band=0.02,
        cvar_penalty=0.60,
        benchmark_tilt=0.20,
    ),
    "balanced": RiskProfileConfig(
        max_weight=0.10,
        max_sector_weight=0.40,
        signal_strength=0.80,
        allow_exit_threshold=0.010,
        min_weight_floor=0.015,
        new_stock_budget=0.30,
        turnover_penalty=0.30,
        transaction_cost_rate=0.0020,
        no_trade_band=0.015,
        cvar_penalty=0.40,
        benchmark_tilt=0.30,
    ),
    "aggressive": RiskProfileConfig(
        max_weight=0.15,
        max_sector_weight=0.48,
        signal_strength=0.95,
        allow_exit_threshold=0.005,
        min_weight_floor=0.01,
        new_stock_budget=0.40,
        turnover_penalty=0.15,
        transaction_cost_rate=0.0015,
        no_trade_band=0.010,
        cvar_penalty=0.25,
        benchmark_tilt=0.45,
    ),
}

MANDATE_PROFILE_CONFIG: Dict[str, MandateProfileConfig] = {
    "balanced_equity": MandateProfileConfig(
        label="Balanced Equity Fund",
        objective="Construct a diversified NGX equity portfolio with a balanced risk-return profile.",
        benchmark="NGX All-Share / liquidity-weighted equity universe proxy",
        max_turnover=0.40,
    ),
    "growth_equity": MandateProfileConfig(
        label="Growth Equity Fund",
        objective="Tilt toward higher-conviction ML buy signals while retaining diversification controls.",
        benchmark="NGX 30 / high-liquidity equity proxy",
        max_stock_weight=0.15,
        max_sector_weight=0.48,
        min_liquidity_score=0.30,
        max_turnover=0.55,
        max_new_stock_budget=0.45,
    ),
    "income_equity": MandateProfileConfig(
        label="Income / Defensive Equity Fund",
        objective="Prefer steadier, liquid equities with tighter concentration and turnover limits.",
        benchmark="NGX dividend and large-cap equity proxy",
        max_stock_weight=0.08,
        max_sector_weight=0.35,
        min_liquidity_score=0.45,
        max_turnover=0.30,
        max_portfolio_volatility=0.55,
        max_new_stock_budget=0.20,
        min_buy_confidence=0.04,
    ),
    "pension_equity": MandateProfileConfig(
        label="Pension-Style Equity Sleeve",
        objective="Model the equity component of a pension-style mandate with strict concentration controls.",
        benchmark="Pension equity sleeve / broad NGX equity proxy",
        max_stock_weight=0.07,
        max_sector_weight=0.30,
        min_liquidity_score=0.50,
        max_turnover=0.25,
        max_portfolio_volatility=0.45,
        max_new_stock_budget=0.15,
        min_buy_confidence=0.05,
    ),
}


def is_equity(symbol: str) -> bool:
    return not bool(NON_EQUITY_PATTERN.search(str(symbol).upper()))


def _equity_symbol_mask(symbols: pd.Series) -> pd.Series:
    return ~symbols.astype(str).str.upper().str.contains(NON_EQUITY_PATTERN, regex=True, na=False)


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise ValidationError("Symbol cannot be blank.")
    return value


def _normalize_rebalance_frequency(value: str) -> str:
    frequency = str(value or "monthly").strip().lower()
    if frequency not in REBALANCE_STEP_MAP:
        raise ValidationError("rebalance_frequency must be one of: weekly, monthly, quarterly.")
    return frequency


def _sector_for_symbol(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Other")


def _file_signature(path: Path) -> Tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=8)
def _load_price_data_cached(resolved_path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    df = pd.read_csv(resolved_path, low_memory=False)
    df.columns = [col.replace("\ufeff", "") for col in df.columns]
    required = {"SYMBOL", "TRANS_DATE", "CLOSE_PRICE"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(f"Price file missing required columns: {sorted(missing)}")

    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df = df[_equity_symbol_mask(df["SYMBOL"])].copy()
    df["TRANS_DATE"] = pd.to_datetime(df["TRANS_DATE"], errors="coerce")

    numeric_columns = [
        "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE",
        "CHANGE", "TRADES", "VOLUME", "TRADE_VALUE",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["TRANS_DATE", "CLOSE_PRICE"])
    df = df[df["CLOSE_PRICE"] > 0].copy()
    df = df.sort_values(["SYMBOL", "TRANS_DATE"]).reset_index(drop=True)
    return df


def _load_price_data(price_file: str = PRICE_FILE) -> pd.DataFrame:
    path = Path(price_file)
    if not path.exists():
        raise ValidationError(f"Price file not found: {price_file}")

    signature = _file_signature(path)
    with PRICE_CACHE_LOCK:
        return _load_price_data_cached(*signature).copy()


def get_supported_symbols(price_file: str = PRICE_FILE) -> List[str]:
    df = _load_price_data(price_file)
    counts = df.groupby("SYMBOL").size().sort_values(ascending=False)
    return counts.index.tolist()


def _latest_prices(price_df: pd.DataFrame) -> pd.Series:
    latest = (
        price_df.sort_values("TRANS_DATE")
        .groupby("SYMBOL")
        .tail(1)
        .set_index("SYMBOL")["CLOSE_PRICE"]
        .astype(float)
    )
    return latest


def get_latest_price_snapshot(price_file: str = PRICE_FILE) -> Dict[str, object]:
    df = _load_price_data(price_file)
    latest_date = df["TRANS_DATE"].max()
    return {
        "prices": {symbol: round(float(price), 4) for symbol, price in _latest_prices(df).items()},
        "updated_at": latest_date.isoformat() if pd.notna(latest_date) else None,
    }


def load_signal_store(
    signal_file: str = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
) -> pd.DataFrame:
    path = Path(signal_file)
    if not path.exists():
        raise SignalStoreError(
            f"Signal store not found at '{signal_file}'. Run merge_signals.py first."
        )

    age_hours = (
        pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    ).total_seconds() / 3600
    if age_hours > stale_after_hours:
        raise SignalStoreError(
            f"Signal store is stale ({age_hours:.1f}h old). Refresh it by running merge_signals.py."
        )

    signature = _file_signature(path)
    with SIGNAL_CACHE_LOCK:
        return _load_signal_store_cached(*signature).copy()


@lru_cache(maxsize=8)
def _load_signal_store_cached(resolved_path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    df = pd.read_csv(resolved_path)
    if "Symbol" not in df.columns:
        raise SignalStoreError("signal_store.csv is missing the 'Symbol' column.")

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    numeric_columns = [
        "Avg_Confidence", "Avg_R2", "Consensus_Tier",
        "Avg_Quality_R2", "Qualified_Models",
        "XGB_Return (%)", "RF_Return (%)", "LSTM_Return (%)",
        "XGB_Confidence", "RF_Confidence", "LSTM_Confidence",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Consensus_Signal" in df.columns:
        df["Consensus_Signal"] = df["Consensus_Signal"].astype(str).str.upper()
    for column in ["XGB_Quality_Pass", "RF_Quality_Pass", "LSTM_Quality_Pass"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.lower().isin(["true", "1", "yes"])

    model_return_columns = []
    for prefix in ["XGB", "RF", "LSTM"]:
        return_col = f"{prefix}_Return (%)"
        quality_col = f"{prefix}_Quality_Pass"
        if return_col not in df.columns:
            continue
        if quality_col in df.columns:
            filtered_col = f"{prefix}_Qualified_Return"
            df[filtered_col] = df[return_col].where(df[quality_col])
            model_return_columns.append(filtered_col)
        else:
            model_return_columns.append(return_col)

    if model_return_columns:
        df["Avg_Return"] = df[model_return_columns].mean(axis=1, skipna=True)
    elif "Predicted_Return (%)" in df.columns:
        df["Avg_Return"] = pd.to_numeric(df["Predicted_Return (%)"], errors="coerce")
    else:
        df["Avg_Return"] = 0.0
    df["Avg_Return"] = df["Avg_Return"].fillna(0.0)

    df["Avg_Confidence"] = df.get("Avg_Confidence", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Avg_R2"] = df.get("Avg_R2", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Avg_Quality_R2"] = df.get("Avg_Quality_R2", df["Avg_R2"]).fillna(0.0)
    df["Qualified_Models"] = df.get("Qualified_Models", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Consensus_Tier"] = df.get("Consensus_Tier", pd.Series(3.0, index=df.index)).fillna(3.0)
    return df


def get_signal_summary(
    signal_file: str = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
) -> Dict[str, object]:
    path = Path(signal_file)
    df = load_signal_store(signal_file, stale_after_hours)
    modified_at = pd.Timestamp(path.stat().st_mtime, unit="s")
    return {
        "path": str(path.resolve()),
        "row_count": int(len(df)),
        "generated_at": modified_at.isoformat(),
        "buy_count": int((df.get("Consensus_Signal") == "BUY").sum()),
        "sell_count": int((df.get("Consensus_Signal") == "SELL").sum()),
        "conflict_count": int((df.get("Consensus_Signal") == "CONFLICT").sum()),
        "avg_confidence": round(float(df["Avg_Confidence"].mean()), 4) if not df.empty else 0.0,
        "avg_r2": round(float(df["Avg_R2"].mean()), 4) if not df.empty else 0.0,
    }


def _signal_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["Signal_Score"] = (
        ranked["Avg_Confidence"] * 0.45
        + ranked.get("Avg_Quality_R2", ranked["Avg_R2"]).clip(lower=0.0) * 0.20
        + (ranked["Avg_Return"].clip(lower=0.0) / 100.0) * 0.25
        + ((4 - ranked["Consensus_Tier"].clip(lower=1, upper=3)) / 10.0) * 0.10
    )
    return ranked


def get_signal_watchlist(
    signal_file: str = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
    limit: int = 5,
) -> Dict[str, object]:
    df = _signal_score_frame(load_signal_store(signal_file, stale_after_hours))
    if "Consensus_Signal" not in df.columns:
        raise SignalStoreError("signal_store.csv is missing the 'Consensus_Signal' column.")

    model_signal_columns = [col for col in ["XGB_Signal", "RF_Signal", "LSTM_Signal"] if col in df.columns]
    model_return_columns = [col for col in ["XGB_Return (%)", "RF_Return (%)", "LSTM_Return (%)"] if col in df.columns]
    if model_signal_columns:
        df["Buy_Votes"] = (df[model_signal_columns] == "BUY").sum(axis=1)
    else:
        df["Buy_Votes"] = 0
    if model_return_columns:
        df["Best_Model_Return"] = df[model_return_columns].max(axis=1, skipna=True)
    else:
        df["Best_Model_Return"] = df["Avg_Return"]

    strict_buys = df[df["Consensus_Signal"] == "BUY"].sort_values(
        ["Signal_Score", "Avg_Return", "Avg_Confidence"],
        ascending=False,
    )
    opportunistic_buys = df[
        (df["Consensus_Signal"] != "SELL")
        & (
            (df["Buy_Votes"] > 0)
            | (df["Best_Model_Return"].fillna(0.0) > 0)
            | (df["Avg_Return"].fillna(0.0) > 0)
        )
    ].sort_values(
        ["Buy_Votes", "Best_Model_Return", "Signal_Score", "Avg_Confidence"],
        ascending=False,
    )
    top_buys = (
        pd.concat([strict_buys, opportunistic_buys], ignore_index=True)
        .drop_duplicates(subset=["Symbol"])
        .head(limit)
    )
    top_sells = df[df["Consensus_Signal"] == "SELL"].sort_values(
        ["Avg_Return", "Avg_Confidence"],
        ascending=[True, False],
    ).head(limit)

    def _serialize(row: pd.Series, side: str) -> Dict[str, object]:
        avg_return = float(row.get("Avg_Return", 0.0))
        confidence = float(row.get("Avg_Confidence", 0.0))
        signal = str(row.get("Consensus_Signal", "UNKNOWN"))
        if side == "buy":
            reason = (
                f"{signal} consensus with {avg_return:.2f}% expected upside and "
                f"{confidence * 100:.0f}% average confidence."
            )
        else:
            reason = (
                f"{signal} consensus with {avg_return:.2f}% expected return and "
                f"{confidence * 100:.0f}% confidence."
            )
        return {
            "symbol": row["Symbol"],
            "signal": signal,
            "sector": _sector_for_symbol(row["Symbol"]),
            "avg_return": round(avg_return, 4),
            "avg_confidence": round(confidence, 4),
            "avg_r2": round(float(row.get("Avg_R2", 0.0)), 4),
            "signal_score": round(float(row.get("Signal_Score", 0.0)), 6),
            "buy_votes": int(row.get("Buy_Votes", 0)),
            "reason": reason,
        }

    return {
        "top_buys": [_serialize(row, "buy") for _, row in top_buys.iterrows()],
        "top_sells": [_serialize(row, "sell") for _, row in top_sells.iterrows()],
    }


def _validate_and_normalize_holdings(
    holdings: Sequence[Dict[str, object]],
    supported_symbols: Sequence[str],
) -> pd.DataFrame:
    if not holdings:
        raise ValidationError("Portfolio must contain at least one holding.")

    supported_set = set(supported_symbols)
    rows = []
    for item in holdings:
        symbol = _normalize_symbol(item.get("symbol"))
        amount = float(item.get("amount_naira", 0))
        if amount <= 0:
            raise ValidationError(f"Amount for {symbol} must be greater than 0.")
        if symbol not in supported_set:
            raise ValidationError(f"Unsupported symbol: {symbol}")
        rows.append({"symbol": symbol, "amount_naira": amount, "sector": _sector_for_symbol(symbol)})

    df = pd.DataFrame(rows)
    df = df.groupby(["symbol", "sector"], as_index=False)["amount_naira"].sum()
    df["current_weight"] = df["amount_naira"] / df["amount_naira"].sum()
    return df.sort_values("symbol").reset_index(drop=True)


def _prepare_signal_candidates(
    signals_df: pd.DataFrame,
    min_confidence: float = MIN_SIGNAL_CONFIDENCE,
) -> pd.DataFrame:
    df = _signal_score_frame(signals_df)
    if "Consensus_Signal" in df.columns:
        df = df[df["Consensus_Signal"] == "BUY"].copy()
    df = df[df["Avg_Confidence"].fillna(0) >= min_confidence].copy()
    df = df[df["Consensus_Tier"].fillna(99) <= 2].copy()
    df = df[df["Avg_Return"].fillna(0) > 0].copy()
    df["sector"] = df["Symbol"].map(_sector_for_symbol)
    return df


def _full_signal_frame(signals_df: pd.DataFrame) -> pd.DataFrame:
    df = _signal_score_frame(signals_df)
    if "Consensus_Signal" in df.columns:
        df["Consensus_Signal"] = df["Consensus_Signal"].astype(str).str.upper()
    else:
        df["Consensus_Signal"] = "UNKNOWN"
    return df


def _build_asset_metadata(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.sort_values(["SYMBOL", "TRANS_DATE"]).copy()
    df["return"] = df.groupby("SYMBOL")["CLOSE_PRICE"].pct_change()
    df["volume"] = df.get("VOLUME", pd.Series(np.nan, index=df.index)).fillna(0.0)
    df["trade_value"] = df.get("TRADE_VALUE", pd.Series(np.nan, index=df.index)).fillna(0.0)

    trailing = (
        df.groupby("SYMBOL")
        .tail(20)
        .groupby("SYMBOL")
        .agg(
            latest_price=("CLOSE_PRICE", "last"),
            avg_volume_20d=("volume", "mean"),
            avg_trade_value_20d=("trade_value", "mean"),
            volatility_20d=("return", "std"),
        )
        .reset_index()
    )
    trailing["sector"] = trailing["SYMBOL"].map(_sector_for_symbol)
    trailing["liquidity_score"] = (
        trailing["avg_trade_value_20d"].fillna(0.0).rank(pct=True)
    )
    trailing["volatility_20d"] = trailing["volatility_20d"].fillna(0.0)
    return trailing.rename(columns={"SYMBOL": "symbol"})


def _build_candidate_universe(
    holdings_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    asset_metadata: pd.DataFrame,
    allow_new_stocks: bool,
    max_new_stocks: int,
    min_liquidity_score: float = 0.35,
    min_buy_confidence: float = MIN_SIGNAL_CONFIDENCE,
) -> Tuple[List[str], int]:
    current_symbols = holdings_df["symbol"].tolist()
    if not allow_new_stocks or max_new_stocks <= 0:
        return current_symbols, 0

    candidates = _prepare_signal_candidates(signals_df, min_confidence=min_buy_confidence)
    candidates = candidates[~candidates["Symbol"].isin(current_symbols)].copy()
    if candidates.empty:
        return current_symbols, 0

    liquidity = asset_metadata.set_index("symbol")
    candidates["liquidity_score"] = candidates["Symbol"].map(
        liquidity.get("liquidity_score", pd.Series(dtype=float))
    ).fillna(0.0)
    candidates["avg_trade_value_20d"] = candidates["Symbol"].map(
        liquidity.get("avg_trade_value_20d", pd.Series(dtype=float))
    ).fillna(0.0)
    candidates = candidates[candidates["liquidity_score"] >= min_liquidity_score].copy()
    candidates = candidates.sort_values(
        ["Signal_Score", "Avg_Return", "Avg_Confidence", "liquidity_score"],
        ascending=False,
    )
    additions = candidates.head(max_new_stocks)["Symbol"].tolist()
    return current_symbols + additions, int(len(candidates))


def _build_returns_matrix(price_df: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    subset = price_df[price_df["SYMBOL"].isin(symbols)].copy()
    pivot = (
        subset.pivot_table(index="TRANS_DATE", columns="SYMBOL", values="CLOSE_PRICE", aggfunc="last")
        .sort_index()
        .ffill()
    )
    pivot = pivot.dropna(axis=1, thresh=MIN_HISTORY_DAYS)
    returns = pivot.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna(axis=0, how="any")
    if returns.empty or returns.shape[1] == 0:
        raise ValidationError("Not enough historical data to optimize this portfolio.")
    return returns


def _risk_profile_config(risk_profile: str) -> RiskProfileConfig:
    profile = str(risk_profile).strip().lower()
    if profile not in RISK_PROFILE_CONFIG:
        raise ValidationError("risk_profile must be one of: conservative, balanced, aggressive.")
    return RISK_PROFILE_CONFIG[profile]


def _mandate_profile_config(mandate_profile: str) -> Tuple[str, MandateProfileConfig]:
    profile = str(mandate_profile or "balanced_equity").strip().lower()
    if profile not in MANDATE_PROFILE_CONFIG:
        options = ", ".join(sorted(MANDATE_PROFILE_CONFIG))
        raise ValidationError(f"mandate_profile must be one of: {options}.")
    return profile, MANDATE_PROFILE_CONFIG[profile]


def _apply_mandate_to_risk_config(
    risk_config: RiskProfileConfig,
    mandate_config: MandateProfileConfig,
) -> RiskProfileConfig:
    updates = {}
    if mandate_config.max_stock_weight is not None:
        updates["max_weight"] = min(risk_config.max_weight, mandate_config.max_stock_weight)
    if mandate_config.max_sector_weight is not None:
        updates["max_sector_weight"] = min(
            risk_config.max_sector_weight,
            mandate_config.max_sector_weight,
        )
    if mandate_config.max_new_stock_budget is not None:
        updates["new_stock_budget"] = min(
            risk_config.new_stock_budget,
            mandate_config.max_new_stock_budget,
        )
    return replace(risk_config, **updates) if updates else risk_config


def _expected_returns(
    returns_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: RiskProfileConfig,
) -> pd.Series:
    hist_mu = returns_df.mean() * TRADING_DAYS
    signal_df = _full_signal_frame(signals_df).set_index("Symbol")
    ml_forecast = pd.Series(index=returns_df.columns, dtype="float64")

    for symbol in returns_df.columns:
        if symbol in signal_df.index:
            row = signal_df.loc[symbol]
            confidence = float(row.get("Avg_Confidence", 0.0))
            signal_return = float(row.get("Avg_Return", 0.0)) / 100.0
            consensus = str(row.get("Consensus_Signal", "UNKNOWN")).upper()
            if consensus == "BUY" and signal_return > 0:
                blend = min(max(confidence * config.signal_strength, 0.10), 0.90)
                ml_forecast.loc[symbol] = blend * signal_return + (1 - blend) * hist_mu.loc[symbol]
            elif consensus == "SELL" and signal_return < 0:
                blend = min(max(confidence * config.signal_strength, 0.10), 0.90)
                ml_forecast.loc[symbol] = blend * signal_return + (1 - blend) * min(hist_mu.loc[symbol], 0.0)
            elif consensus == "CONFLICT":
                ml_forecast.loc[symbol] = min(hist_mu.loc[symbol], 0.0) * 0.5
            else:
                ml_forecast.loc[symbol] = hist_mu.loc[symbol]
        else:
            ml_forecast.loc[symbol] = hist_mu.loc[symbol]

    return ml_forecast.clip(lower=-0.60, upper=1.20)


def _covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    cov = returns_df.cov() * TRADING_DAYS
    diagonal = np.diag(np.diag(cov.values))
    shrunk = 0.80 * cov.values + 0.20 * diagonal
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def _build_benchmark_weights(symbols: Sequence[str], asset_metadata: pd.DataFrame) -> pd.Series:
    if not symbols:
        return pd.Series(dtype="float64")
    benchmark = asset_metadata.set_index("symbol").reindex(symbols)["avg_trade_value_20d"].fillna(0.0)
    if benchmark.sum() <= 0:
        benchmark[:] = 1.0
    return benchmark / benchmark.sum()


def _portfolio_return_series(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns_df @ pd.Series(weights, index=returns_df.columns)


def _max_drawdown(series: pd.Series) -> float:
    wealth = (1 + series.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _cvar_95(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cutoff = float(series.quantile(0.05))
    tail = series[series <= cutoff]
    return float(tail.mean()) if not tail.empty else cutoff


def _tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    active = portfolio_returns - benchmark_returns
    return float(active.std(ddof=0) * math.sqrt(TRADING_DAYS)) if len(active) else 0.0


def _sortino_ratio(series: pd.Series, annual_return: float) -> float:
    downside = series[series < 0]
    downside_dev = float(downside.std(ddof=0) * math.sqrt(TRADING_DAYS)) if len(downside) else 0.0
    excess_return = annual_return - RISK_FREE_RATE
    return excess_return / downside_dev if downside_dev > 0 else 0.0


def _portfolio_metrics(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    expected_return = float(np.dot(weights, mu.values))
    variance = float(weights.T @ cov.values @ weights)
    volatility = math.sqrt(max(variance, 0.0))
    sharpe = (expected_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0.0

    portfolio_returns = _portfolio_return_series(returns_df, weights)
    realized_return = float(portfolio_returns.mean() * TRADING_DAYS) if len(portfolio_returns) else 0.0
    sortino = _sortino_ratio(portfolio_returns, realized_return)
    cvar_95 = _cvar_95(portfolio_returns)
    max_drawdown = _max_drawdown(portfolio_returns)
    tracking_error = _tracking_error(portfolio_returns, benchmark_returns)
    active_return = realized_return - float(benchmark_returns.mean() * TRADING_DAYS)
    info_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown,
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
        "annualized_realized_return": realized_return,
    }


def _normalize_weights(weights: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    normalized = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, None)
    total = normalized.sum()
    if total > 0:
        return normalized / total

    if fallback is not None:
        fallback_weights = np.nan_to_num(np.asarray(fallback, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        fallback_weights = np.clip(fallback_weights, 0.0, None)
        fallback_total = fallback_weights.sum()
        if fallback_total > 0:
            return fallback_weights / fallback_total

    if len(normalized) == 0:
        return normalized
    return np.ones_like(normalized, dtype=float) / len(normalized)


def _apply_weight_constraints(weights: np.ndarray, config: RiskProfileConfig, floor: float = 0.0) -> np.ndarray:
    clipped = _normalize_weights(np.clip(weights, 0.0, config.max_weight))

    if floor > 0:
        clipped[clipped < floor] = 0.0
        clipped = _normalize_weights(clipped)

    for _ in range(10):
        over = clipped > config.max_weight
        if not over.any():
            break
        excess = clipped[over] - config.max_weight
        clipped[over] = config.max_weight
        under = ~over
        if under.any() and excess.sum() > 0:
            under_total = clipped[under].sum()
            if under_total > 0:
                clipped[under] += excess.sum() * (clipped[under] / under_total)
            else:
                clipped[under] += excess.sum() / under.sum()
        clipped = np.clip(clipped, 0.0, config.max_weight)
        clipped = _normalize_weights(clipped)

    return clipped


def _apply_sector_constraints(
    weights: np.ndarray,
    symbols: Sequence[str],
    sectors: Dict[str, str],
    config: RiskProfileConfig,
) -> np.ndarray:
    adjusted = weights.copy()
    for _ in range(8):
        changed = False
        sector_groups: Dict[str, List[int]] = {}
        for index, symbol in enumerate(symbols):
            sector_groups.setdefault(sectors.get(symbol, "Other"), []).append(index)

        for indexes in sector_groups.values():
            sector_weight = adjusted[indexes].sum()
            if sector_weight <= config.max_sector_weight:
                continue
            changed = True
            excess = sector_weight - config.max_sector_weight
            adjusted[indexes] *= config.max_sector_weight / sector_weight
            outside = np.ones(len(adjusted), dtype=bool)
            outside[indexes] = False
            if outside.any():
                outside_total = adjusted[outside].sum()
                if outside_total > 0:
                    adjusted[outside] += excess * (adjusted[outside] / outside_total)
                else:
                    adjusted[outside] += excess / outside.sum()

        adjusted = np.clip(adjusted, 0.0, config.max_weight)
        if adjusted.sum() <= 0:
            adjusted = np.ones_like(adjusted)
        adjusted = adjusted / adjusted.sum()
        if not changed:
            break
    return adjusted


def _apply_trade_bands(
    candidate: np.ndarray,
    current: np.ndarray,
    config: RiskProfileConfig,
) -> np.ndarray:
    adjusted = candidate.copy()
    within_band = np.abs(adjusted - current) < config.no_trade_band
    adjusted[within_band] = current[within_band]
    if adjusted.sum() <= 0:
        return current if current.sum() > 0 else np.ones_like(adjusted) / len(adjusted)
    return adjusted / adjusted.sum()


def _score_portfolio(
    weights: np.ndarray,
    current: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    config: RiskProfileConfig,
) -> Tuple[float, Dict[str, float]]:
    metrics = _portfolio_metrics(weights, mu, cov, returns_df, benchmark_returns)
    turnover = float(np.abs(weights - current).sum() / 2)
    transaction_cost = turnover * config.transaction_cost_rate
    score = (
        metrics["sharpe"]
        + config.benchmark_tilt * metrics["information_ratio"]
        + 0.10 * metrics["sortino"]
        - config.cvar_penalty * abs(metrics["cvar_95"])
        - 0.20 * abs(metrics["max_drawdown"])
        - config.turnover_penalty * turnover
        - transaction_cost
    )
    metrics["turnover"] = turnover
    metrics["transaction_cost_rate"] = transaction_cost
    metrics["objective_score"] = score
    return score, metrics


def _fast_score_portfolio(
    weights: np.ndarray,
    current: np.ndarray,
    mu_values: np.ndarray,
    cov_values: np.ndarray,
    benchmark: np.ndarray,
    benchmark_expected_return: float,
    config: RiskProfileConfig,
) -> float:
    expected_return = float(np.dot(weights, mu_values))
    variance = float(weights.T @ cov_values @ weights)
    volatility = math.sqrt(max(variance, 0.0))
    sharpe = (expected_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0.0

    active = weights - benchmark
    tracking_variance = float(active.T @ cov_values @ active)
    tracking_error = math.sqrt(max(tracking_variance, 0.0))
    information_ratio = (
        (expected_return - benchmark_expected_return) / tracking_error
        if tracking_error > 0
        else 0.0
    )
    turnover = float(np.abs(weights - current).sum() / 2)
    transaction_cost = turnover * config.transaction_cost_rate
    return (
        sharpe
        + config.benchmark_tilt * information_ratio
        - config.turnover_penalty * turnover
        - transaction_cost
    )


def _optimize_weights(
    returns_df: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    current_weights: pd.Series,
    benchmark_weights: pd.Series,
    sectors: Dict[str, str],
    config: RiskProfileConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(OPTIMIZER_SEED)
    symbols = list(returns_df.columns)
    n_assets = len(symbols)

    current = current_weights.reindex(symbols).fillna(0.0).values
    if current.sum() <= 0:
        current = np.ones(n_assets) / n_assets
    else:
        current = current / current.sum()

    benchmark = benchmark_weights.reindex(symbols).fillna(0.0).values
    if benchmark.sum() <= 0:
        benchmark = np.ones(n_assets) / n_assets
    else:
        benchmark = benchmark / benchmark.sum()
    benchmark_returns = _portfolio_return_series(returns_df, benchmark)
    mu_values = mu.reindex(symbols).fillna(0.0).values
    cov_values = cov.reindex(index=symbols, columns=symbols).fillna(0.0).values
    benchmark_expected_return = float(np.dot(benchmark, mu_values))

    def _constrain(weights: np.ndarray, floor: float) -> np.ndarray:
        adjusted = _apply_weight_constraints(weights, config, floor=floor)
        adjusted = _apply_sector_constraints(adjusted, symbols, sectors, config)
        adjusted = _apply_trade_bands(adjusted, current, config)
        adjusted = _apply_weight_constraints(adjusted, config, floor=0.0)
        adjusted = _apply_sector_constraints(adjusted, symbols, sectors, config)
        return _normalize_weights(adjusted, fallback=current)

    signal_alpha = np.maximum(mu.values - mu.values.min() + 0.05, 0.01)
    seeds = [
        _constrain(current.copy(), floor=0.0),
        _constrain(benchmark.copy(), floor=0.0),
        _constrain(np.ones(n_assets) / n_assets, floor=0.0),
    ]

    best_weights = seeds[0]
    best_score = _fast_score_portfolio(
        best_weights, current, mu_values, cov_values, benchmark, benchmark_expected_return, config
    )

    for seed_weights in seeds[1:]:
        score = _fast_score_portfolio(
            seed_weights, current, mu_values, cov_values, benchmark, benchmark_expected_return, config
        )
        if score > best_score:
            best_weights = seed_weights
            best_score = score

    sample_count = min(RANDOM_PORTFOLIO_SAMPLES, max(600, n_assets * 250))
    for _ in range(sample_count):
        sampled = rng.dirichlet(signal_alpha)
        blend_to_current = rng.uniform(0.20, 0.60)
        blend_to_benchmark = rng.uniform(0.10, 0.35)
        candidate = (
            sampled * (1 - blend_to_current - blend_to_benchmark)
            + current * blend_to_current
            + benchmark * blend_to_benchmark
        )
        candidate = _constrain(candidate, floor=config.min_weight_floor)
        score = _fast_score_portfolio(
            candidate, current, mu_values, cov_values, benchmark, benchmark_expected_return, config
        )
        if score > best_score:
            best_weights = candidate
            best_score = score

    _, best_metrics = _score_portfolio(
        best_weights, current, mu, cov, returns_df, benchmark_returns, config
    )
    return best_weights, best_metrics


def _ensure_new_stock_exposure(
    weights: np.ndarray,
    symbols: Sequence[str],
    current_symbols: Sequence[str],
    config: RiskProfileConfig,
) -> np.ndarray:
    new_indexes = [index for index, symbol in enumerate(symbols) if symbol not in current_symbols]
    if not new_indexes:
        return weights

    adjusted = weights.copy()
    target_each = min(config.min_weight_floor, config.new_stock_budget / max(len(new_indexes), 1))
    for index in new_indexes:
        shortfall = max(0.0, target_each - adjusted[index])
        if shortfall <= 0:
            continue
        donors = [i for i in range(len(adjusted)) if i != index]
        donor_total = adjusted[donors].sum()
        if donor_total > 0:
            adjusted[donors] -= shortfall * (adjusted[donors] / donor_total)
        adjusted[index] += shortfall
        adjusted = np.clip(adjusted, 0.0, None)
        adjusted = _normalize_weights(adjusted, fallback=weights)
    return _normalize_weights(adjusted, fallback=weights)


def _selective_exit_indexes(
    symbols: Sequence[str],
    current_weights: pd.Series,
    signals_df: pd.DataFrame,
    config: RiskProfileConfig,
) -> List[int]:
    signal_frame = _full_signal_frame(signals_df).set_index("Symbol")
    exit_indexes: List[int] = []
    for index, symbol in enumerate(symbols):
        current_weight = float(current_weights.get(symbol, 0.0))
        if current_weight <= config.allow_exit_threshold or symbol not in signal_frame.index:
            continue

        row = signal_frame.loc[symbol]
        consensus = str(row.get("Consensus_Signal", "")).upper()
        qualified = float(row.get("Qualified_Models", 0.0) or 0.0)
        confidence = float(row.get("Avg_Confidence", 0.0) or 0.0)
        avg_return = float(row.get("Avg_Return", 0.0) or 0.0)

        clear_sell = consensus == "SELL" and qualified >= 2 and avg_return < 0
        untrusted_conflict = (
            consensus == "CONFLICT"
            and qualified <= 0
            and confidence <= MIN_SIGNAL_CONFIDENCE
            and avg_return <= 0
        )
        if clear_sell or untrusted_conflict:
            exit_indexes.append(index)
    return exit_indexes


def _apply_selective_exits(
    weights: np.ndarray,
    symbols: Sequence[str],
    current_weights: pd.Series,
    signals_df: pd.DataFrame,
    config: RiskProfileConfig,
) -> np.ndarray:
    exit_indexes = _selective_exit_indexes(symbols, current_weights, signals_df, config)
    if not exit_indexes:
        return _normalize_weights(weights)

    adjusted = _normalize_weights(weights)
    current_positive = [
        index for index, symbol in enumerate(symbols)
        if float(current_weights.get(symbol, 0.0)) > config.allow_exit_threshold
    ]
    if len(exit_indexes) >= len(current_positive):
        return adjusted

    exit_weight = float(adjusted[exit_indexes].sum())
    if exit_weight <= 0:
        return adjusted

    receiver_mask = np.ones(len(adjusted), dtype=bool)
    receiver_mask[exit_indexes] = False
    if receiver_mask.sum() * config.max_weight < 1.0:
        return adjusted

    adjusted[exit_indexes] = 0.0

    signal_frame = _full_signal_frame(signals_df).set_index("Symbol")
    receiver_scores = np.zeros(len(adjusted), dtype=float)
    for index, symbol in enumerate(symbols):
        if not receiver_mask[index]:
            continue
        if symbol in signal_frame.index:
            row = signal_frame.loc[symbol]
            consensus = str(row.get("Consensus_Signal", "")).upper()
            if consensus == "BUY":
                receiver_scores[index] = max(float(row.get("Signal_Score", 0.0) or 0.0), 0.0)

    if receiver_scores.sum() > 0:
        adjusted += exit_weight * (receiver_scores / receiver_scores.sum())
    else:
        receiver_total = adjusted[receiver_mask].sum()
        if receiver_total > 0:
            adjusted[receiver_mask] += exit_weight * (adjusted[receiver_mask] / receiver_total)
        else:
            adjusted[receiver_mask] += exit_weight / receiver_mask.sum()

    return _normalize_weights(adjusted, fallback=weights)


def _build_actions(result_df: pd.DataFrame, config: RiskProfileConfig) -> pd.DataFrame:
    actions = []
    for row in result_df.itertuples():
        current = row.current_weight
        optimized = row.optimized_weight
        delta = row.weight_delta

        if current <= config.allow_exit_threshold and optimized > config.allow_exit_threshold:
            action = "add"
        elif optimized <= config.allow_exit_threshold and current > config.allow_exit_threshold:
            action = "exit"
        elif delta > config.no_trade_band:
            action = "increase"
        elif delta < -config.no_trade_band:
            action = "reduce"
        else:
            action = "keep"
        actions.append(action)

    result_df["action"] = actions
    return result_df


def _serialize_sector_allocations(result_df: pd.DataFrame) -> List[Dict[str, object]]:
    grouped = (
        result_df.groupby("sector", as_index=False)[["current_weight", "optimized_weight"]]
        .sum()
        .sort_values("optimized_weight", ascending=False)
    )
    return [
        {
            "sector": row["sector"],
            "current_weight": round(float(row["current_weight"]), 6),
            "optimized_weight": round(float(row["optimized_weight"]), 6),
        }
        for _, row in grouped.iterrows()
    ]


def _serialize_model_votes(symbol: str, signal_metadata: pd.DataFrame) -> List[Dict[str, object]]:
    if symbol not in signal_metadata.index:
        return []

    row = signal_metadata.loc[symbol]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    votes = []
    for model in ["XGB", "RF", "LSTM"]:
        signal_col = f"{model}_Signal"
        return_col = f"{model}_Return (%)"
        confidence_col = f"{model}_Confidence"
        r2_col = f"{model}_R2"
        quality_col = f"{model}_Quality_Pass"
        if signal_col not in row.index:
            continue
        votes.append(
            {
                "model": model,
                "signal": str(row.get(signal_col, "UNKNOWN")),
                "expected_return": round(float(row.get(return_col, 0.0) or 0.0) / 100.0, 6),
                "confidence": round(float(row.get(confidence_col, 0.0) or 0.0), 4),
                "r2": round(float(row.get(r2_col, 0.0) or 0.0), 4),
                "quality_pass": bool(row.get(quality_col, False)),
            }
        )
    return votes


def _build_prediction_engine_summary(signals_df: pd.DataFrame) -> Dict[str, object]:
    model_columns = [col for col in ["XGB_Signal", "RF_Signal", "LSTM_Signal"] if col in signals_df.columns]
    qualified = pd.to_numeric(
        signals_df.get("Qualified_Models", pd.Series(0.0, index=signals_df.index)),
        errors="coerce",
    ).fillna(0.0)
    confidence = pd.to_numeric(
        signals_df.get("Avg_Confidence", pd.Series(0.0, index=signals_df.index)),
        errors="coerce",
    ).fillna(0.0)
    r2 = pd.to_numeric(
        signals_df.get("Avg_R2", pd.Series(0.0, index=signals_df.index)),
        errors="coerce",
    ).fillna(0.0)
    consensus = signals_df.get("Consensus_Signal", pd.Series("", index=signals_df.index)).astype(str)

    return {
        "scope": "Nigerian listed equities",
        "models": [col.replace("_Signal", "") for col in model_columns],
        "symbols_scored": int(len(signals_df)),
        "buy_count": int((consensus == "BUY").sum()),
        "sell_count": int((consensus == "SELL").sum()),
        "conflict_count": int((consensus == "CONFLICT").sum()),
        "average_confidence": round(float(confidence.mean()), 4) if len(confidence) else 0.0,
        "average_r2": round(float(r2.mean()), 4) if len(r2) else 0.0,
        "qualified_model_coverage": round(float((qualified > 0).mean()), 4) if len(qualified) else 0.0,
    }


def _compliance_item(
    rule: str,
    observed: float | int | str,
    limit: float | int | str,
    passed: bool,
    message: str,
    severity: str = "breach",
) -> Dict[str, object]:
    return {
        "rule": rule,
        "status": "pass" if passed else severity,
        "observed": observed,
        "limit": limit,
        "message": message,
    }


def _build_compliance_report(
    result_df: pd.DataFrame,
    sector_allocations: List[Dict[str, object]],
    turnover: float,
    config: RiskProfileConfig,
    mandate_profile: str,
    mandate_config: MandateProfileConfig,
    max_new_stocks: int,
    added_symbols: Sequence[str],
    optimized_metrics: Dict[str, float],
) -> Dict[str, object]:
    active_positions = result_df[result_df["optimized_weight"] > config.allow_exit_threshold].copy()
    max_stock_weight = float(result_df["optimized_weight"].max()) if not result_df.empty else 0.0
    max_sector_weight = max((float(item["optimized_weight"]) for item in sector_allocations), default=0.0)
    weakest_liquidity = (
        float(active_positions["liquidity_score"].min())
        if not active_positions.empty
        else 0.0
    )
    optimized_volatility = float(optimized_metrics.get("volatility", 0.0))

    items = [
        _compliance_item(
            "Single-stock concentration",
            round(max_stock_weight, 6),
            round(config.max_weight, 6),
            max_stock_weight <= config.max_weight + 1e-6,
            "Largest optimized equity position must stay within the mandate cap.",
        ),
        _compliance_item(
            "Sector concentration",
            round(max_sector_weight, 6),
            round(config.max_sector_weight, 6),
            max_sector_weight <= config.max_sector_weight + 1e-6,
            "Largest optimized sector exposure must stay within the mandate cap.",
        ),
        _compliance_item(
            "Turnover control",
            round(turnover, 6),
            round(mandate_config.max_turnover, 6),
            turnover <= mandate_config.max_turnover + 1e-6,
            "Recommended trades should not create excessive rebalancing for the selected mandate.",
            severity="warn",
        ),
        _compliance_item(
            "New-stock limit",
            len(added_symbols),
            int(max_new_stocks),
            len(added_symbols) <= int(max_new_stocks),
            "Optimizer cannot introduce more new equities than allowed by the fund manager.",
        ),
        _compliance_item(
            "Liquidity screen",
            round(weakest_liquidity, 6),
            round(mandate_config.min_liquidity_score, 6),
            weakest_liquidity >= mandate_config.min_liquidity_score - 1e-6,
            "Active optimized positions should pass the mandate liquidity threshold.",
            severity="warn",
        ),
    ]

    if mandate_config.max_portfolio_volatility is not None:
        items.append(
            _compliance_item(
                "Portfolio volatility",
                round(optimized_volatility, 6),
                round(mandate_config.max_portfolio_volatility, 6),
                optimized_volatility <= mandate_config.max_portfolio_volatility + 1e-6,
                "Annualized optimized volatility should remain within the mandate tolerance.",
                severity="warn",
            )
        )

    if any(item["status"] == "breach" for item in items):
        overall_status = "breach"
    elif any(item["status"] == "warn" for item in items):
        overall_status = "review"
    else:
        overall_status = "pass"

    return {
        "overall_status": overall_status,
        "mandate_profile": mandate_profile,
        "mandate_label": mandate_config.label,
        "checked_at": pd.Timestamp.now("UTC").isoformat(),
        "items": items,
    }


def _build_fund_manager_report(
    mandate_profile: str,
    mandate_config: MandateProfileConfig,
    compliance_report: Dict[str, object],
    current_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    added_symbols: Sequence[str],
    removed_symbols: Sequence[str],
) -> Dict[str, object]:
    if compliance_report["overall_status"] == "pass":
        recommendation = "Proceed with rebalance subject to investment committee approval."
    elif compliance_report["overall_status"] == "review":
        recommendation = "Review mandate warnings before execution."
    else:
        recommendation = "Do not execute until compliance breaches are resolved."

    return {
        "title": "Equity Portfolio Construction Report",
        "market": "Nigeria / NGX listed equities",
        "mandate_profile": mandate_profile,
        "mandate_label": mandate_config.label,
        "objective": mandate_config.objective,
        "benchmark": mandate_config.benchmark,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "recommendation": recommendation,
        "summary": {
            "current_expected_return": round(float(current_metrics["expected_return"]), 6),
            "optimized_expected_return": round(float(optimized_metrics["expected_return"]), 6),
            "current_sharpe": round(float(current_metrics["sharpe"]), 6),
            "optimized_sharpe": round(float(optimized_metrics["sharpe"]), 6),
            "added_symbols": list(added_symbols),
            "removed_symbols": list(removed_symbols),
            "compliance_status": compliance_report["overall_status"],
        },
    }


def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _simulate_rebalanced_portfolio(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    step: int,
) -> pd.Series:
    subset = returns_df.copy()
    if subset.empty:
        return pd.Series(dtype="float64")

    target = weights.reindex(subset.columns).fillna(0.0)
    if target.sum() <= 0:
        return pd.Series(0.0, index=subset.index)
    target = target / target.sum()

    current_weights = target.copy()
    realized = []
    for i, (_, row) in enumerate(subset.iterrows()):
        portfolio_return = float(np.dot(current_weights.values, row.values))
        realized.append(portfolio_return)
        drifted = current_weights.values * (1 + row.values)
        total = drifted.sum()
        if total > 0:
            current_weights = pd.Series(drifted / total, index=subset.columns)
        if (i + 1) % step == 0:
            current_weights = target.copy()
    return pd.Series(realized, index=subset.index)


def _summarize_backtest_series(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    cumulative = float((1 + series).prod() - 1)
    annualized_return = float(series.mean() * TRADING_DAYS)
    annualized_volatility = float(series.std(ddof=0) * math.sqrt(TRADING_DAYS))
    sharpe = (
        (annualized_return - RISK_FREE_RATE) / annualized_volatility
        if annualized_volatility > 0
        else 0.0
    )
    return {
        "cumulative_return": round(cumulative, 6),
        "annualized_return": round(annualized_return, 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(_max_drawdown(series), 6),
    }


def _build_backtest_summary(
    returns_df: pd.DataFrame,
    current_weights: pd.Series,
    optimized_weights: pd.Series,
    benchmark_weights: pd.Series,
    rebalance_frequency: str,
) -> Dict[str, object]:
    recent = returns_df.tail(min(BACKTEST_LOOKBACK_DAYS, len(returns_df)))
    step = REBALANCE_STEP_MAP[rebalance_frequency]
    equal_weights = pd.Series(1 / len(recent.columns), index=recent.columns)

    strategies = {
        "current_portfolio": _simulate_rebalanced_portfolio(recent, current_weights, step),
        "optimized_portfolio": _simulate_rebalanced_portfolio(recent, optimized_weights, step),
        "equal_weight": _simulate_rebalanced_portfolio(recent, equal_weights, step),
        "benchmark": _simulate_rebalanced_portfolio(recent, benchmark_weights, step),
    }
    summaries = {name: _summarize_backtest_series(series) for name, series in strategies.items()}
    winner = max(summaries.items(), key=lambda item: item[1]["cumulative_return"])[0]
    return {
        "window_days": int(len(recent)),
        "rebalance_frequency": rebalance_frequency,
        "winner": winner,
        "strategies": summaries,
    }


def optimize_portfolio(
    holdings: Sequence[Dict[str, object]],
    risk_profile: str = "balanced",
    allow_new_stocks: bool = True,
    max_new_stocks: int = 5,
    price_file: str = PRICE_FILE,
    signal_file: str = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
    rebalance_frequency: str = "monthly",
    holding_period_days: int = 20,
    mandate_profile: str = "balanced_equity",
    construction_amount_naira: float | None = None,
) -> Dict[str, object]:
    is_construction = construction_amount_naira is not None and len(holdings) == 0
    construction_amount = float(construction_amount_naira or 0.0)
    if is_construction and construction_amount <= 0:
        raise ValidationError("initial_cash_naira must be greater than 0 for portfolio construction.")

    supported_symbols = get_supported_symbols(price_file)
    if is_construction:
        holdings_df = pd.DataFrame(columns=["symbol", "amount_naira", "sector", "current_weight"])
    else:
        holdings_df = _validate_and_normalize_holdings(holdings, supported_symbols)
    signals_df = load_signal_store(signal_file, stale_after_hours)
    mandate_profile, mandate_config = _mandate_profile_config(mandate_profile)
    config = _apply_mandate_to_risk_config(
        _risk_profile_config(risk_profile),
        mandate_config,
    )
    rebalance_frequency = _normalize_rebalance_frequency(rebalance_frequency)
    max_new_stocks = int(max_new_stocks)
    if max_new_stocks < 0 or max_new_stocks > 20:
        raise ValidationError("max_new_stocks must be between 0 and 20.")
    holding_period_days = int(holding_period_days)
    if holding_period_days <= 0 or holding_period_days > 252:
        raise ValidationError("holding_period_days must be between 1 and 252.")

    effective_allow_new_stocks = True if is_construction else allow_new_stocks
    effective_max_new_stocks = max(1, max_new_stocks) if is_construction else max_new_stocks
    price_df = _load_price_data(price_file)
    asset_metadata = _build_asset_metadata(price_df)

    candidate_symbols, liquidity_screened_count = _build_candidate_universe(
        holdings_df=holdings_df,
        signals_df=signals_df,
        asset_metadata=asset_metadata,
        allow_new_stocks=effective_allow_new_stocks,
        max_new_stocks=effective_max_new_stocks,
        min_liquidity_score=mandate_config.min_liquidity_score,
        min_buy_confidence=mandate_config.min_buy_confidence,
    )

    returns_df = _build_returns_matrix(price_df, candidate_symbols)
    candidate_symbols = list(returns_df.columns)
    if not is_construction and not set(holdings_df["symbol"]).issubset(candidate_symbols):
        missing = sorted(set(holdings_df["symbol"]) - set(candidate_symbols))
        raise ValidationError(f"Not enough price history for holdings: {', '.join(missing)}")

    if is_construction:
        current_weights = pd.Series(0.0, index=candidate_symbols)
    else:
        current_weights = holdings_df.set_index("symbol")["current_weight"].reindex(candidate_symbols).fillna(0.0)
        current_weights = current_weights / current_weights.sum()
    sectors = {symbol: _sector_for_symbol(symbol) for symbol in candidate_symbols}

    metadata = asset_metadata.set_index("symbol").reindex(candidate_symbols)
    latest_prices = metadata["latest_price"]
    benchmark_weights = _build_benchmark_weights(candidate_symbols, asset_metadata)
    benchmark_returns = _portfolio_return_series(returns_df, benchmark_weights.values)
    mu = _expected_returns(returns_df, signals_df, config).reindex(candidate_symbols).fillna(0.0)
    cov = _covariance_matrix(returns_df).reindex(index=candidate_symbols, columns=candidate_symbols)

    optimized_weights, optimization_metrics = _optimize_weights(
        returns_df=returns_df,
        mu=mu,
        cov=cov,
        current_weights=current_weights,
        benchmark_weights=benchmark_weights,
        sectors=sectors,
        config=config,
    )
    optimized_weights = _ensure_new_stock_exposure(
        optimized_weights, candidate_symbols, holdings_df["symbol"].tolist(), config
    )
    optimized_weights = _normalize_weights(_apply_weight_constraints(optimized_weights, config, floor=0.0))
    optimized_weights = _apply_sector_constraints(optimized_weights, candidate_symbols, sectors, config)
    optimized_weights = _normalize_weights(optimized_weights)
    optimized_weights = _apply_selective_exits(
        optimized_weights,
        candidate_symbols,
        current_weights,
        signals_df,
        config,
    )
    optimized_weights = _normalize_weights(_apply_weight_constraints(optimized_weights, config, floor=0.0))
    optimized_weights = _apply_sector_constraints(optimized_weights, candidate_symbols, sectors, config)
    optimized_weights = _normalize_weights(optimized_weights)

    optimized_weights_series = pd.Series(optimized_weights, index=candidate_symbols)
    optimized_metrics = _portfolio_metrics(
        optimized_weights, mu, cov, returns_df, benchmark_returns
    )
    current_metrics = _portfolio_metrics(
        current_weights.values, mu, cov, returns_df, benchmark_returns
    )
    benchmark_metrics = _portfolio_metrics(
        benchmark_weights.values, mu, cov, returns_df, benchmark_returns
    )

    signal_metadata = _full_signal_frame(signals_df).set_index("Symbol")
    result_df = pd.DataFrame(
        {
            "symbol": candidate_symbols,
            "sector": [sectors[symbol] for symbol in candidate_symbols],
            "current_weight": current_weights.values,
            "optimized_weight": optimized_weights,
            "latest_price": latest_prices.values,
            "expected_return": mu.values,
            "avg_volume_20d": metadata["avg_volume_20d"].fillna(0.0).values,
            "avg_trade_value_20d": metadata["avg_trade_value_20d"].fillna(0.0).values,
            "volatility_20d": metadata["volatility_20d"].fillna(0.0).values,
            "liquidity_score": metadata["liquidity_score"].fillna(0.0).values,
        }
    )
    result_df["signal_status"] = result_df["symbol"].map(signal_metadata.get("Consensus_Signal", pd.Series(dtype=object)))
    result_df["consensus_tier"] = result_df["symbol"].map(signal_metadata.get("Consensus_Tier", pd.Series(dtype=float)))
    result_df["avg_confidence"] = result_df["symbol"].map(signal_metadata.get("Avg_Confidence", pd.Series(dtype=float)))
    result_df["avg_r2"] = result_df["symbol"].map(signal_metadata.get("Avg_R2", pd.Series(dtype=float)))
    result_df["signal_score"] = result_df["symbol"].map(
        signal_metadata.get("Signal_Score", pd.Series(dtype=float))
    ).fillna(0.0)
    result_df["weight_delta"] = result_df["optimized_weight"] - result_df["current_weight"]
    result_df = _build_actions(result_df, config)
    result_df["selected_new_stock"] = (
        (result_df["current_weight"] <= config.allow_exit_threshold)
        & (result_df["optimized_weight"] > config.allow_exit_threshold)
    )
    result_df["removed_stock"] = (
        (result_df["current_weight"] > config.allow_exit_threshold)
        & (result_df["optimized_weight"] <= config.allow_exit_threshold)
    )
    result_df = result_df.sort_values("optimized_weight", ascending=False).reset_index(drop=True)

    added_symbols = result_df[result_df["selected_new_stock"]]["symbol"].tolist()
    removed_symbols = result_df[result_df["removed_stock"]]["symbol"].tolist()
    sector_allocations = _serialize_sector_allocations(result_df)
    backtest_summary = _build_backtest_summary(
        returns_df,
        current_weights,
        optimized_weights_series,
        benchmark_weights,
        rebalance_frequency,
    )

    portfolio_value = construction_amount if is_construction else float(holdings_df["amount_naira"].sum())
    turnover = float(np.abs(optimized_weights_series - current_weights).sum() / 2)
    transaction_cost_naira = turnover * config.transaction_cost_rate * portfolio_value
    max_sector_weight_after = 0.0
    if sector_allocations:
        max_sector_weight_after = max(item["optimized_weight"] for item in sector_allocations)

    compliance_report = _build_compliance_report(
        result_df=result_df,
        sector_allocations=sector_allocations,
        turnover=turnover,
        config=config,
        mandate_profile=mandate_profile,
        mandate_config=mandate_config,
        max_new_stocks=max_new_stocks,
        added_symbols=added_symbols,
        optimized_metrics=optimized_metrics,
    )
    fund_manager_report = _build_fund_manager_report(
        mandate_profile=mandate_profile,
        mandate_config=mandate_config,
        compliance_report=compliance_report,
        current_metrics=current_metrics,
        optimized_metrics=optimized_metrics,
        added_symbols=added_symbols,
        removed_symbols=removed_symbols,
    )

    def _serialize(row: pd.Series) -> Dict[str, object]:
        return {
            "symbol": row["symbol"],
            "sector": row["sector"],
            "current_weight": round(float(row["current_weight"]), 6),
            "optimized_weight": round(float(row["optimized_weight"]), 6),
            "weight_delta": round(float(row["weight_delta"]), 6),
            "action": row["action"],
            "latest_price": round(float(row["latest_price"]), 4) if pd.notna(row["latest_price"]) else None,
            "expected_return": round(float(row["expected_return"]), 6),
            "signal_status": row["signal_status"] if pd.notna(row["signal_status"]) else "HISTORICAL",
            "consensus_tier": int(row["consensus_tier"]) if pd.notna(row["consensus_tier"]) else None,
            "avg_confidence": round(float(row["avg_confidence"]), 4) if pd.notna(row["avg_confidence"]) else None,
            "avg_r2": round(float(row["avg_r2"]), 4) if pd.notna(row["avg_r2"]) else None,
            "signal_score": round(float(row["signal_score"]), 6),
            "avg_volume_20d": round(float(row["avg_volume_20d"]), 2),
            "avg_trade_value_20d": round(float(row["avg_trade_value_20d"]), 2),
            "volatility_20d": round(float(row["volatility_20d"]), 6),
            "liquidity_score": round(float(row["liquidity_score"]), 6),
            "model_votes": _serialize_model_votes(row["symbol"], signal_metadata),
        }

    response = {
        "portfolio_mode": "construction" if is_construction else "optimization",
        "risk_profile": risk_profile.lower(),
        "mandate_profile": mandate_profile,
        "mandate_summary": {
            "label": mandate_config.label,
            "objective": mandate_config.objective,
            "benchmark": mandate_config.benchmark,
            "max_stock_weight": round(float(config.max_weight), 6),
            "max_sector_weight": round(float(config.max_sector_weight), 6),
            "min_liquidity_score": round(float(mandate_config.min_liquidity_score), 6),
            "max_turnover": round(float(mandate_config.max_turnover), 6),
            "max_portfolio_volatility": (
                round(float(mandate_config.max_portfolio_volatility), 6)
                if mandate_config.max_portfolio_volatility is not None
                else None
            ),
        },
        "prediction_engine": _build_prediction_engine_summary(signals_df),
        "allow_new_stocks": bool(effective_allow_new_stocks),
        "max_new_stocks": int(effective_max_new_stocks),
        "rebalance_frequency": rebalance_frequency,
        "holding_period_days": holding_period_days,
        "current_portfolio_value": round(portfolio_value, 2),
        "initial_cash_naira": round(construction_amount, 2) if is_construction else None,
        "current_weights": [
            {
                "symbol": row.symbol,
                "amount_naira": round(float(row.amount_naira), 2),
                "weight": round(float(row.current_weight), 6),
                "sector": row.sector,
            }
            for row in holdings_df.itertuples()
        ],
        "optimized_allocations": [_serialize(row) for _, row in result_df.iterrows()],
        "added_symbols": added_symbols,
        "removed_symbols": removed_symbols,
        "sector_allocations": sector_allocations,
        "constraint_summary": {
            "max_stock_weight": round(float(result_df["optimized_weight"].max()), 6),
            "max_sector_weight": round(float(max_sector_weight_after), 6),
            "turnover": round(turnover, 6),
            "transaction_cost_rate": round(config.transaction_cost_rate, 6),
            "estimated_transaction_cost_naira": round(transaction_cost_naira, 2),
            "liquidity_screened_candidates": liquidity_screened_count,
            "no_trade_band": round(config.no_trade_band, 6),
        },
        "compliance_report": compliance_report,
        "benchmark_metrics": _round_metrics(benchmark_metrics),
        "backtest_summary": backtest_summary,
        "fund_manager_report": fund_manager_report,
        "summary_metrics": {
            "candidate_count": int(len(candidate_symbols)),
            "current_expected_return": round(current_metrics["expected_return"], 6),
            "current_volatility": round(current_metrics["volatility"], 6),
            "current_sharpe": round(current_metrics["sharpe"], 6),
            "current_sortino": round(current_metrics["sortino"], 6),
            "current_cvar_95": round(current_metrics["cvar_95"], 6),
            "current_max_drawdown": round(current_metrics["max_drawdown"], 6),
            "optimized_expected_return": round(optimized_metrics["expected_return"], 6),
            "optimized_volatility": round(optimized_metrics["volatility"], 6),
            "optimized_sharpe": round(optimized_metrics["sharpe"], 6),
            "optimized_sortino": round(optimized_metrics["sortino"], 6),
            "optimized_cvar_95": round(optimized_metrics["cvar_95"], 6),
            "optimized_max_drawdown": round(optimized_metrics["max_drawdown"], 6),
            "optimized_tracking_error": round(optimized_metrics["tracking_error"], 6),
            "optimized_information_ratio": round(optimized_metrics["information_ratio"], 6),
            "optimization_objective_score": round(float(optimization_metrics.get("objective_score", 0.0)), 6),
        },
    }
    return _json_safe(response)


def construct_portfolio(
    initial_cash_naira: float,
    risk_profile: str = "balanced",
    max_stocks: int = 8,
    price_file: str = PRICE_FILE,
    signal_file: str = SIGNAL_FILE,
    stale_after_hours: int = STALE_SIGNAL_HOURS,
    rebalance_frequency: str = "monthly",
    holding_period_days: int = 20,
    mandate_profile: str = "balanced_equity",
) -> Dict[str, object]:
    return optimize_portfolio(
        holdings=[],
        risk_profile=risk_profile,
        mandate_profile=mandate_profile,
        allow_new_stocks=True,
        max_new_stocks=max_stocks,
        price_file=price_file,
        signal_file=signal_file,
        stale_after_hours=stale_after_hours,
        rebalance_frequency=rebalance_frequency,
        holding_period_days=holding_period_days,
        construction_amount_naira=initial_cash_naira,
    )

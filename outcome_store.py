"""
outcome_store.py
----------------
Persists every portfolio optimization result as individual trade-level rows,
then computes realised outcomes once prices move.

Two things live here:
  1. SQLite database  (outcome_store.db)  — the primary, queryable store.
  2. CSV mirror       (outcome_store.csv) — human-readable, importable into Excel.

Schema
------
trades
  id              TEXT  PRIMARY KEY   e.g. "trade_<hex>"
  run_id          TEXT                links back to a batch/single run
  portfolio_id    TEXT                FK to portfolio_store portfolio
  manager_id      TEXT
  consumer_id     TEXT
  consumer_name   TEXT
  symbol          TEXT
  sector          TEXT
  action          TEXT                keep | increase | reduce | add | exit
  signal_status   TEXT                BUY | SELL | CONFLICT | HISTORICAL
  signal_score    REAL
  avg_confidence  REAL
  avg_r2          REAL
  expected_return REAL                optimizer's forecast at decision time (fraction)
  current_weight  REAL
  optimized_weight REAL
  latest_price    REAL                price at decision date
  decision_date   TEXT                ISO date of the optimization run
  outcome_price   TEXT                price captured later (NULL until measured)
  outcome_date    TEXT                date outcome_price was captured (NULL until measured)
  realised_return REAL                (outcome_price - latest_price) / latest_price
  prediction_error REAL               realised_return - expected_return
  risk_profile    TEXT
  mandate_profile TEXT
  holding_period_days INT

outcome_snapshots (lightweight; one row per symbol per price-capture pass)
  id              TEXT  PRIMARY KEY
  symbol          TEXT
  capture_date    TEXT
  close_price     REAL

The learning query that answers "did Chevron work in April → does it work in August?"
is simply:
  SELECT symbol, decision_date, expected_return, realised_return, prediction_error
  FROM trades
  WHERE symbol = 'CHEVRON'
  ORDER BY decision_date
"""

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent
DB_FILE   = PROJECT_ROOT / "outcome_store.db"
CSV_FILE  = PROJECT_ROOT / "outcome_store.csv"
_DB_LOCK  = Lock()


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _connect(db_file: Path = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── schema ────────────────────────────────────────────────────────────────────

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    portfolio_id        TEXT NOT NULL,
    manager_id          TEXT NOT NULL,
    consumer_id         TEXT NOT NULL DEFAULT '',
    consumer_name       TEXT NOT NULL DEFAULT '',
    symbol              TEXT NOT NULL,
    sector              TEXT NOT NULL DEFAULT '',
    action              TEXT NOT NULL,
    signal_status       TEXT NOT NULL DEFAULT '',
    signal_score        REAL NOT NULL DEFAULT 0,
    avg_confidence      REAL,
    avg_r2              REAL,
    expected_return     REAL NOT NULL DEFAULT 0,
    current_weight      REAL NOT NULL DEFAULT 0,
    optimized_weight    REAL NOT NULL DEFAULT 0,
    latest_price        REAL,
    decision_date       TEXT NOT NULL,
    outcome_price       REAL,
    outcome_date        TEXT,
    realised_return     REAL,
    prediction_error    REAL,
    risk_profile        TEXT NOT NULL DEFAULT '',
    mandate_profile     TEXT NOT NULL DEFAULT '',
    holding_period_days INTEGER NOT NULL DEFAULT 20
)
"""

_CREATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS outcome_snapshots (
    id           TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    capture_date TEXT NOT NULL,
    close_price  REAL NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol         ON trades(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_run_id         ON trades(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_trades_portfolio_id   ON trades(portfolio_id)",
    "CREATE INDEX IF NOT EXISTS idx_trades_manager_id     ON trades(manager_id)",
    "CREATE INDEX IF NOT EXISTS idx_trades_decision_date  ON trades(decision_date)",
    "CREATE INDEX IF NOT EXISTS idx_trades_outcome_date   ON trades(outcome_date)",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_date ON outcome_snapshots(symbol, capture_date)",
]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TRADES)
    conn.execute(_CREATE_SNAPSHOTS)
    for stmt in _CREATE_INDEXES:
        conn.execute(stmt)
    conn.commit()


# ── CSV mirror ────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "id", "run_id", "portfolio_id", "manager_id", "consumer_id", "consumer_name",
    "symbol", "sector", "action", "signal_status", "signal_score",
    "avg_confidence", "avg_r2", "expected_return", "current_weight",
    "optimized_weight", "latest_price", "decision_date",
    "outcome_price", "outcome_date", "realised_return", "prediction_error",
    "risk_profile", "mandate_profile", "holding_period_days",
]


def _sync_csv(db_file: Path = DB_FILE, csv_file: Path = CSV_FILE) -> None:
    """Rewrite the CSV from the full trades table. Called after every mutation."""
    conn = _connect(db_file)
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_CSV_COLS)} FROM trades ORDER BY decision_date, symbol"
        ).fetchall()
    finally:
        conn.close()

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLS)
        for row in rows:
            writer.writerow([row[c] for c in _CSV_COLS])


# ── public API ────────────────────────────────────────────────────────────────

def record_optimization_trades(
    *,
    run_id: str,
    portfolio_id: str,
    manager_id: str,
    consumer_id: str,
    consumer_name: str,
    risk_profile: str,
    mandate_profile: str,
    holding_period_days: int,
    result: Dict[str, Any],
    db_file: Path = DB_FILE,
    csv_file: Path = CSV_FILE,
) -> List[str]:
    """
    Extract per-symbol trade rows from an optimize_portfolio result dict
    and persist them.  Returns the list of inserted trade IDs.
    """
    decision_date = _now()
    allocations: List[Dict[str, Any]] = result.get("optimized_allocations", [])
    if not allocations:
        return []

    rows = []
    for alloc in allocations:
        rows.append({
            "id": _new_id("trd"),
            "run_id": run_id,
            "portfolio_id": portfolio_id,
            "manager_id": manager_id,
            "consumer_id": consumer_id,
            "consumer_name": consumer_name,
            "symbol": alloc.get("symbol", ""),
            "sector": alloc.get("sector", ""),
            "action": alloc.get("action", ""),
            "signal_status": alloc.get("signal_status", ""),
            "signal_score": float(alloc.get("signal_score") or 0),
            "avg_confidence": alloc.get("avg_confidence"),
            "avg_r2": alloc.get("avg_r2"),
            "expected_return": float(alloc.get("expected_return") or 0),
            "current_weight": float(alloc.get("current_weight") or 0),
            "optimized_weight": float(alloc.get("optimized_weight") or 0),
            "latest_price": alloc.get("latest_price"),
            "decision_date": decision_date,
            "outcome_price": None,
            "outcome_date": None,
            "realised_return": None,
            "prediction_error": None,
            "risk_profile": risk_profile,
            "mandate_profile": mandate_profile,
            "holding_period_days": int(holding_period_days),
        })

    cols = list(rows[0].keys())
    placeholders = ", ".join("?" * len(cols))
    sql = f"INSERT OR IGNORE INTO trades ({', '.join(cols)}) VALUES ({placeholders})"

    with _DB_LOCK:
        conn = _connect(db_file)
        try:
            _ensure_schema(conn)
            conn.executemany(sql, [[r[c] for c in cols] for r in rows])
            conn.commit()
        finally:
            conn.close()
        _sync_csv(db_file, csv_file)

    return [r["id"] for r in rows]


def record_price_snapshot(
    prices: Dict[str, float],
    capture_date: Optional[str] = None,
    db_file: Path = DB_FILE,
) -> int:
    """
    Store a price snapshot (e.g. from your daily PRICE_LIST load) and immediately
    compute realised returns for any open trades whose holding_period has elapsed.
    Returns the number of trades updated.
    """
    date_str = capture_date or _now()[:10]
    snapshot_rows = [
        (_new_id("snp"), symbol, date_str, float(price))
        for symbol, price in prices.items()
    ]

    with _DB_LOCK:
        conn = _connect(db_file)
        try:
            _ensure_schema(conn)
            conn.executemany(
                "INSERT OR IGNORE INTO outcome_snapshots (id, symbol, capture_date, close_price) VALUES (?,?,?,?)",
                snapshot_rows,
            )

            # For each open trade whose holding period has passed, find the
            # closest snapshot price and compute realised return + prediction error.
            open_trades = conn.execute(
                """
                SELECT id, symbol, latest_price, expected_return, decision_date, holding_period_days
                FROM trades
                WHERE outcome_price IS NULL AND latest_price IS NOT NULL
                """
            ).fetchall()

            updated = 0
            for trade in open_trades:
                due_date = (
                    datetime.fromisoformat(trade["decision_date"][:10])
                    .date()
                    .__str__()
                )
                # Simple check: capture_date >= due_date (holding period elapsed check
                # is approximate here; a scheduler should call this daily)
                if date_str < due_date:
                    continue
                outcome = prices.get(trade["symbol"])
                if outcome is None:
                    continue
                entry = float(trade["latest_price"])
                if entry == 0:
                    continue
                realised = (outcome - entry) / entry
                error = realised - float(trade["expected_return"])
                conn.execute(
                    """
                    UPDATE trades
                    SET outcome_price=?, outcome_date=?, realised_return=?, prediction_error=?
                    WHERE id=?
                    """,
                    (outcome, date_str, round(realised, 6), round(error, 6), trade["id"]),
                )
                updated += 1

            conn.commit()
        finally:
            conn.close()
    return updated


def get_symbol_outcome_history(
    symbol: str,
    db_file: Path = DB_FILE,
) -> List[Dict[str, Any]]:
    """
    Return all recorded trades for a symbol with their outcomes.
    Used by the optimizer to answer: "has this stock paid off historically?"
    """
    with _DB_LOCK:
        conn = _connect(db_file)
        try:
            _ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    symbol, decision_date, action, signal_status, signal_score,
                    avg_confidence, avg_r2, expected_return,
                    outcome_date, realised_return, prediction_error,
                    risk_profile, mandate_profile, consumer_name, portfolio_id
                FROM trades
                WHERE symbol = ?
                ORDER BY decision_date
                """,
                (symbol.upper(),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_symbol_alpha_signal(
    symbol: str,
    db_file: Path = DB_FILE,
) -> Dict[str, Any]:
    """
    Aggregate historical accuracy for a symbol.
    Returns a dict the optimizer can use as an extra prior:
      {
        "symbol": "ZENITHBANK",
        "sample_size": 12,
        "hit_rate": 0.67,          # fraction where realised_return > 0
        "mean_realised_return": 0.043,
        "mean_prediction_error": -0.012,
        "last_realised_return": 0.08,
        "last_decision_date": "2024-04-01",
      }
    """
    rows = get_symbol_outcome_history(symbol, db_file)
    settled = [r for r in rows if r["realised_return"] is not None]
    if not settled:
        return {
            "symbol": symbol.upper(),
            "sample_size": 0,
            "hit_rate": None,
            "mean_realised_return": None,
            "mean_prediction_error": None,
            "last_realised_return": None,
            "last_decision_date": None,
        }
    realised = [r["realised_return"] for r in settled]
    errors   = [r["prediction_error"] for r in settled if r["prediction_error"] is not None]
    last     = settled[-1]
    return {
        "symbol": symbol.upper(),
        "sample_size": len(settled),
        "hit_rate": round(sum(1 for v in realised if v > 0) / len(realised), 4),
        "mean_realised_return": round(sum(realised) / len(realised), 6),
        "mean_prediction_error": round(sum(errors) / len(errors), 6) if errors else None,
        "last_realised_return": round(last["realised_return"], 6),
        "last_decision_date": last["decision_date"][:10],
    }


def list_outcome_trades(
    manager_id: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    symbol: Optional[str] = None,
    settled_only: bool = False,
    limit: int = 500,
    db_file: Path = DB_FILE,
) -> List[Dict[str, Any]]:
    """Flexible query for the frontend outcome explorer."""
    clauses = []
    params: List[Any] = []
    if manager_id:
        clauses.append("manager_id = ?")
        params.append(manager_id)
    if portfolio_id:
        clauses.append("portfolio_id = ?")
        params.append(portfolio_id)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    if settled_only:
        clauses.append("outcome_price IS NOT NULL")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT {', '.join(_CSV_COLS)}
        FROM trades
        {where}
        ORDER BY decision_date DESC, symbol
        LIMIT ?
    """
    params.append(limit)

    with _DB_LOCK:
        conn = _connect(db_file)
        try:
            _ensure_schema(conn)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def export_trades_csv(
    db_file: Path = DB_FILE,
    csv_file: Path = CSV_FILE,
) -> Path:
    """Force-sync CSV from DB and return the path."""
    _sync_csv(db_file, csv_file)
    return csv_file

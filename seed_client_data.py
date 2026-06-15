"""
seed_client_data.py
-------------------
Populates portfolio_store.json with fabricated fund manager, client and
portfolio data so that the Clients / Workspace tab can be tested without
running a live optimisation.

IMPORTANT
---------
- This script does NOT touch PRICE_LIST.csv or signal_store.csv.
- All portfolio holdings and run results are derived from symbols and
  prices that already exist in your signal_store.csv so the data is
  internally consistent with the real signals the system has produced.
- Running this script more than once will APPEND a new workspace each
  time.  Delete portfolio_store.json first if you want a clean slate.

Usage
-----
    python seed_client_data.py                     # reads signal_store.csv from cwd
    python seed_client_data.py --store my_dir/     # write store to a different folder
"""

import argparse
import csv
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────────

def _now_iso(offset_days: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(days=offset_days)
    return dt.isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _load_store(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data.setdefault("managers", [])
    data.setdefault("consumers", [])
    data.setdefault("portfolios", [])
    data.setdefault("runs", [])
    return data


def _save_store(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── read real signal store ─────────────────────────────────────────────────────

def load_signal_symbols(signal_file: Path) -> dict:
    """
    Returns a dict keyed by symbol with the fields the optimiser cares about.
    Only includes symbols that have a price (Last_Close) so holdings are realistic.
    """
    symbols = {}
    if not signal_file.exists():
        raise FileNotFoundError(
            f"signal_store.csv not found at {signal_file}.\n"
            "Run merge_signals.py first, then re-run this script."
        )
    with signal_file.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = row.get("Symbol", "").strip()
            price_str = row.get("Last_Close (₦)", "").strip()
            signal = row.get("Consensus_Signal", "").strip().upper()
            if not sym or not price_str:
                continue
            try:
                price = float(price_str)
            except ValueError:
                continue
            if price <= 0:
                continue
            symbols[sym] = {
                "symbol": sym,
                "price": price,
                "signal": signal,
                "confidence": float(row.get("Avg_Confidence") or 0),
                "avg_return": float(row.get("Avg_Return") or 0),
                "tier": int(row.get("Consensus_Tier") or 3),
            }
    return symbols


# ── portfolio builder helpers ──────────────────────────────────────────────────

def _pick_holdings(symbols: dict, signal_filter: str, n: int, rng: random.Random) -> list[dict]:
    """Pick n symbols matching signal_filter and return as holdings (symbol + amount_naira)."""
    pool = [v for v in symbols.values() if v["signal"] == signal_filter and v["price"] > 0]
    if len(pool) < n:
        # fall back to any signal if not enough matching ones
        pool = list(symbols.values())
    chosen = rng.sample(pool, min(n, len(pool)))
    total = rng.uniform(5_000_000, 50_000_000)
    weights = [rng.uniform(0.08, 0.25) for _ in chosen]
    wsum = sum(weights)
    return [
        {"symbol": c["symbol"], "amount_naira": round(total * w / wsum, 2)}
        for c, w in zip(chosen, weights)
    ]


def _make_run_summary(
    holdings: list[dict],
    symbols: dict,
    added: list[str],
    removed: list[str],
    offset_days: int,
    rng: random.Random,
) -> dict:
    """Build a minimal but schema-correct optimisation result that portfolio_store understands."""
    portfolio_value = sum(h["amount_naira"] for h in holdings)
    opt_value = portfolio_value * rng.uniform(1.01, 1.12)
    exp_return = rng.uniform(0.04, 0.18)
    sharpe = rng.uniform(0.45, 1.10)
    generated_at = _now_iso(offset_days)

    # Build optimised_allocations so the dashboard tab has something to render
    allocations = []
    all_syms = [h["symbol"] for h in holdings] + added
    weight_raw = [rng.uniform(0.05, 0.20) for _ in all_syms]
    wsum = sum(weight_raw)
    for sym, w in zip(all_syms, weight_raw):
        info = symbols.get(sym, {})
        action = "add" if sym in added else rng.choice(["keep", "increase", "reduce"])
        allocations.append({
            "symbol": sym,
            "sector": _sector(sym),
            "current_weight": round((w / wsum) * rng.uniform(0.7, 1.0), 4),
            "optimized_weight": round(w / wsum, 4),
            "weight_delta": round(rng.uniform(-0.04, 0.06), 4),
            "action": action,
            "latest_price": info.get("price"),
            "expected_return": round(info.get("avg_return", rng.uniform(2, 15)) / 100, 4),
            "signal_status": info.get("signal", "BUY"),
            "consensus_tier": info.get("tier", 2),
            "avg_confidence": round(info.get("confidence", 0.12), 4),
            "avg_r2": round(rng.uniform(0.04, 0.18), 4),
            "signal_score": round(rng.uniform(0.05, 0.35), 6),
            "avg_volume_20d": round(rng.uniform(500_000, 5_000_000)),
            "avg_trade_value_20d": round(rng.uniform(10_000_000, 200_000_000)),
            "volatility_20d": round(rng.uniform(0.008, 0.035), 4),
            "liquidity_score": round(rng.uniform(0.3, 0.9), 4),
            "model_votes": [],
        })

    compliance_status = rng.choice(["pass", "pass", "pass", "review"])

    return {
        "portfolio_mode": "optimization",
        "risk_profile": "balanced",
        "mandate_profile": "balanced_equity",
        "current_portfolio_value": round(portfolio_value, 2),
        "optimized_portfolio_value": round(opt_value, 2),
        "added_symbols": added,
        "removed_symbols": removed,
        "optimized_allocations": allocations,
        "sector_allocations": [],
        "compliance_report": {
            "overall_status": compliance_status,
            "mandate_profile": "balanced_equity",
            "mandate_label": "Balanced Equity",
            "checked_at": generated_at,
            "items": [],
        },
        "summary_metrics": {
            "optimized_expected_return": round(exp_return, 4),
            "optimized_sharpe": round(sharpe, 4),
            "current_expected_return": round(exp_return * rng.uniform(0.5, 0.85), 4),
            "current_sharpe": round(sharpe * rng.uniform(0.4, 0.8), 4),
            "optimized_volatility": round(rng.uniform(0.08, 0.18), 4),
            "current_volatility": round(rng.uniform(0.10, 0.22), 4),
            "optimized_sortino": round(rng.uniform(0.6, 1.5), 4),
            "optimized_cvar_95": round(rng.uniform(-0.05, -0.02), 4),
            "optimized_max_drawdown": round(rng.uniform(-0.15, -0.05), 4),
            "optimized_tracking_error": round(rng.uniform(0.02, 0.08), 4),
            "optimized_information_ratio": round(rng.uniform(0.2, 0.9), 4),
            "candidate_count": len(all_syms),
            "optimization_objective_score": round(rng.uniform(0.4, 0.9), 4),
        },
        "fund_manager_report": {
            "title": "Portfolio Optimisation Report",
            "market": "Nigerian Exchange Group (NGX)",
            "mandate_profile": "balanced_equity",
            "mandate_label": "Balanced Equity",
            "objective": "Maximise risk-adjusted return with diversified NGX equity exposure.",
            "benchmark": "Liquidity-weighted NGX benchmark",
            "generated_at": generated_at,
            "recommendation": (
                f"The optimised portfolio targets an expected return of "
                f"{round(exp_return * 100, 1)}% with a Sharpe ratio of {round(sharpe, 2)}. "
                f"{'Symbols ' + ', '.join(added) + ' were added based on BUY consensus signals.' if added else ''} "
                f"{'Symbols ' + ', '.join(removed) + ' were exited due to SELL signals.' if removed else ''}"
            ).strip(),
            "summary": {
                "current_expected_return": round(exp_return * rng.uniform(0.5, 0.85), 4),
                "optimized_expected_return": round(exp_return, 4),
                "current_sharpe": round(sharpe * rng.uniform(0.4, 0.8), 4),
                "optimized_sharpe": round(sharpe, 4),
                "added_symbols": added,
                "removed_symbols": removed,
                "compliance_status": compliance_status,
            },
        },
        "backtest_summary": {
            "window_days": 252,
            "rebalance_frequency": "monthly",
            "winner": "optimized_portfolio",
            "strategies": {
                "current_portfolio": {
                    "cumulative_return": round(rng.uniform(0.02, 0.08), 4),
                    "annualized_return": round(rng.uniform(0.02, 0.08), 4),
                    "annualized_volatility": round(rng.uniform(0.10, 0.20), 4),
                    "sharpe": round(sharpe * rng.uniform(0.4, 0.7), 4),
                    "max_drawdown": round(rng.uniform(-0.18, -0.06), 4),
                },
                "optimized_portfolio": {
                    "cumulative_return": round(rng.uniform(0.06, 0.15), 4),
                    "annualized_return": round(exp_return, 4),
                    "annualized_volatility": round(rng.uniform(0.08, 0.16), 4),
                    "sharpe": round(sharpe, 4),
                    "max_drawdown": round(rng.uniform(-0.12, -0.04), 4),
                },
                "equal_weight": {
                    "cumulative_return": round(rng.uniform(0.03, 0.09), 4),
                    "annualized_return": round(rng.uniform(0.03, 0.09), 4),
                    "annualized_volatility": round(rng.uniform(0.09, 0.17), 4),
                    "sharpe": round(rng.uniform(0.30, 0.55), 4),
                    "max_drawdown": round(rng.uniform(-0.16, -0.07), 4),
                },
                "benchmark": {
                    "cumulative_return": round(rng.uniform(0.02, 0.07), 4),
                    "annualized_return": round(rng.uniform(0.02, 0.07), 4),
                    "annualized_volatility": round(rng.uniform(0.08, 0.15), 4),
                    "sharpe": round(rng.uniform(0.25, 0.50), 4),
                    "max_drawdown": round(rng.uniform(-0.14, -0.06), 4),
                },
            },
        },
        "efficient_frontier": {"points": [], "optimized": {}, "current": {}, "benchmark": {}},
        "correlation_matrix": {"symbols": [], "values": []},
        "risk_contributions": [],
        "diversification_score": {
            "score": round(rng.uniform(0.5, 0.85), 4),
            "effective_positions": len(all_syms),
            "active_positions": len(all_syms),
            "effective_sectors": rng.randint(3, 6),
            "sector_count": rng.randint(3, 6),
            "largest_weight": round(max(w / wsum for w in weight_raw), 4),
            "largest_sector_weight": round(rng.uniform(0.20, 0.38), 4),
            "message": "Portfolio is adequately diversified.",
        },
        "benchmark_metrics": {
            "expected_return": round(rng.uniform(0.03, 0.08), 4),
            "volatility": round(rng.uniform(0.08, 0.14), 4),
            "sharpe": round(rng.uniform(0.25, 0.50), 4),
            "sortino": round(rng.uniform(0.30, 0.70), 4),
            "cvar_95": round(rng.uniform(-0.05, -0.02), 4),
            "max_drawdown": round(rng.uniform(-0.14, -0.05), 4),
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "annualized_realized_return": round(rng.uniform(0.02, 0.07), 4),
        },
        "constraint_summary": {
            "max_stock_weight": 0.10,
            "max_sector_weight": 0.40,
            "turnover": round(rng.uniform(0.10, 0.35), 4),
            "transaction_cost_rate": 0.005,
            "estimated_transaction_cost_naira": round(portfolio_value * rng.uniform(0.001, 0.005)),
            "liquidity_screened_candidates": len(all_syms),
            "no_trade_band": 0.02,
        },
        "mandate_summary": {
            "label": "Balanced Equity",
            "objective": "Diversified NGX equity exposure with moderate turnover.",
            "benchmark": "Liquidity-weighted NGX benchmark",
            "max_stock_weight": 0.10,
            "max_sector_weight": 0.40,
            "min_liquidity_score": 0.30,
            "max_turnover": 0.40,
            "max_portfolio_volatility": None,
        },
        "prediction_engine": {
            "scope": "Nigerian Exchange equities",
            "models": ["XGBoost", "Random Forest", "LSTM"],
            "symbols_scored": 129,
            "buy_count": 71,
            "sell_count": 21,
            "conflict_count": 7,
            "average_confidence": 0.1134,
            "average_r2": 0.0512,
            "qualified_model_coverage": round(rng.uniform(0.25, 0.45), 4),
        },
        "allow_new_stocks": True,
        "max_new_stocks": 3,
        "rebalance_frequency": "monthly",
        "holding_period_days": 20,
        "current_weights": [
            {
                "symbol": h["symbol"],
                "amount_naira": h["amount_naira"],
                "weight": round(h["amount_naira"] / portfolio_value, 4),
                "sector": _sector(h["symbol"]),
            }
            for h in holdings
        ],
    }


def _sector(symbol: str) -> str:
    banking = {"ACCESSCORP", "ETI", "FCMB", "FBNH", "FIDELITYBK", "GTCO",
               "JAIZBANK", "STANBIC", "UBA", "WEMABANK", "ZENITHBANK", "UNITYBNK"}
    consumer = {"BUAFOODS", "CADBURY", "DANGSUGAR", "FLOURMILL", "GUINNESS",
                "NASCON", "NB", "NESTLE", "PZ", "UNILEVER", "NNFM", "VITAFOAM"}
    industrial = {"BERGER", "BETAGLAS", "BUACEMENT", "CAP", "CUTIX",
                  "DANGCEM", "WAPCO", "JBERGER"}
    oil_gas = {"ARDOVA", "CONOIL", "ETERNA", "FO", "MRS", "OANDO",
               "SEPLAT", "TOTAL", "ARADEL"}
    insurance = {"AIICO", "CUSTODIAN", "LASACO", "MANSARD", "NEM",
                 "PRESTIGE", "WAPIC", "SUNUASSUR"}
    telecoms = {"AIRTELAFRI", "MTNN"}
    sym = symbol.upper()
    if sym in banking:   return "Banking"
    if sym in consumer:  return "Consumer Goods"
    if sym in industrial: return "Industrial Goods"
    if sym in oil_gas:   return "Oil and Gas"
    if sym in insurance: return "Insurance"
    if sym in telecoms:  return "Telecommunications"
    return "Other"


# ── summary helper (mirrors portfolio_store._summary_from_result) ──────────────

def _make_summary(result: dict) -> dict:
    return {
        "generated_at": result["fund_manager_report"]["generated_at"],
        "portfolio_value": result["current_portfolio_value"],
        "optimized_portfolio_value": result["optimized_portfolio_value"],
        "compliance_status": result["compliance_report"]["overall_status"],
        "optimized_expected_return": result["summary_metrics"]["optimized_expected_return"],
        "optimized_sharpe": result["summary_metrics"]["optimized_sharpe"],
        "added_symbols": result["added_symbols"],
        "removed_symbols": result["removed_symbols"],
    }


# ── main seeding function ──────────────────────────────────────────────────────

def seed(store_path: Path, signal_path: Path) -> None:
    rng = random.Random(42)  # fixed seed → reproducible output

    print(f"Reading signals from: {signal_path}")
    symbols = load_signal_symbols(signal_path)
    buy_syms  = [v for v in symbols.values() if v["signal"] == "BUY"]
    sell_syms = [v for v in symbols.values() if v["signal"] == "SELL"]
    print(f"  {len(buy_syms)} BUY  |  {len(sell_syms)} SELL  |  {len(symbols)} total")

    data = _load_store(store_path)

    # ── 1 fund manager ─────────────────────────────────────────────────────────
    mgr_id = _new_id("mgr")
    manager = {
        "id":         mgr_id,
        "name":       "Adewale Okonkwo",
        "firm":       "Lagos Capital Asset Management",
        "email":      "adewale.okonkwo@lagoscapital.ng",
        "created_at": _now_iso(-60),
        "updated_at": _now_iso(-1),
    }
    data["managers"].append(manager)
    print(f"\nManager: {manager['name']} ({mgr_id})")

    # ── 4 clients ──────────────────────────────────────────────────────────────
    client_defs = [
        ("Chinedu Okafor",   "chinedu.okafor@gmail.com",    True,  "balanced",     "balanced_equity"),
        ("Ngozi Adeleke",    "ngozi.adeleke@outlook.com",   True,  "conservative", "income_equity"),
        ("Emeka Nwosu",      "emeka.nwosu@nwosugroup.com",  True,  "aggressive",   "growth_equity"),
        ("Fatima Al-Hassan", "fatima@alhassaninvest.ng",    False, "balanced",     "pension_equity"),
    ]

    consumer_ids = []
    for name, email, has_portfolio, risk, mandate in client_defs:
        cons_id = _new_id("cons")
        consumer_ids.append(cons_id)
        consumer = {
            "id":                    cons_id,
            "manager_id":            mgr_id,
            "name":                  name,
            "email":                 email,
            "consumer_has_portfolio": has_portfolio,
            "created_at":            _now_iso(-50),
            "updated_at":            _now_iso(-2),
        }
        data["consumers"].append(consumer)
        print(f"  Client: {name} ({cons_id})  risk={risk}  mandate={mandate}")

    # ── portfolios & run histories ─────────────────────────────────────────────
    # Each client gets one portfolio with 3 historical runs spaced ~2 weeks apart.
    portfolio_configs = [
        # (client index, portfolio name, n_holdings, n_add, n_remove, rebalance_freq)
        (0, "Chinedu — Balanced Growth Sleeve",   5, 2, 1, "monthly"),
        (1, "Ngozi — Income Equity Sleeve",        4, 1, 1, "quarterly"),
        (2, "Emeka — High Conviction Growth",      6, 3, 1, "monthly"),
        (3, "Fatima — Pension Equity Sleeve",      4, 1, 0, "quarterly"),
    ]

    for client_idx, pf_name, n_hold, n_add, n_remove, freq in portfolio_configs:
        cons_id = consumer_ids[client_idx]
        cons_name, cons_email, has_portfolio, risk, mandate = client_defs[client_idx]

        pf_id = _new_id("pf")

        # Build initial holdings using BUY signals from the real signal store
        holdings = _pick_holdings(symbols, "BUY", n_hold, rng)

        # Symbols to add in the latest run (also BUY-signal symbols)
        add_pool = [
            s["symbol"] for s in buy_syms
            if s["symbol"] not in {h["symbol"] for h in holdings}
        ]
        added = rng.sample(add_pool, min(n_add, len(add_pool)))

        # Symbols to remove (SELL-signal symbols that happened to be in holdings, or random)
        remove_pool = [h["symbol"] for h in holdings]
        removed = rng.sample(remove_pool, min(n_remove, len(remove_pool))) if n_remove > 0 else []

        # Generate 3 runs (oldest → newest)
        runs = []
        for run_offset, day_offset in enumerate([-28, -14, -1]):
            result = _make_run_summary(holdings, symbols, added if run_offset == 2 else [],
                                       removed if run_offset == 2 else [], day_offset, rng)
            run = {
                "id":           _new_id("run"),
                "manager_id":   mgr_id,
                "portfolio_id": pf_id,
                "created_at":   _now_iso(day_offset),
                "result":       result,
                "summary":      _make_summary(result),
            }
            runs.append(run)
            data["runs"].append(run)

        latest_summary = _make_summary(runs[-1]["result"])

        portfolio = {
            "id":                       pf_id,
            "manager_id":               mgr_id,
            "consumer_id":              cons_id,
            "name":                     pf_name,
            "consumer_has_portfolio":   has_portfolio,
            "consumer_name":            cons_name,
            "consumer_email":           cons_email,
            "initial_cash_naira":       None if has_portfolio else round(rng.uniform(5_000_000, 30_000_000), 2),
            "holdings":                 holdings,
            "risk_profile":             risk,
            "mandate_profile":          mandate,
            "allow_new_stocks":         True,
            "max_new_stocks":           n_add,
            "rebalance_frequency":      freq,
            "holding_period_days":      20,
            "latest_result_summary":    latest_summary,
            "created_at":               _now_iso(-50),
            "updated_at":               _now_iso(-1),
        }
        data["portfolios"].append(portfolio)
        print(f"  Portfolio: {pf_name} ({pf_id}) — {len(holdings)} holdings, 3 runs")

    # ── write store ────────────────────────────────────────────────────────────
    _save_store(data, store_path)
    print(f"\nWrote {store_path}")
    print(f"  managers:  {len(data['managers'])}")
    print(f"  consumers: {len(data['consumers'])}")
    print(f"  portfolios:{len(data['portfolios'])}")
    print(f"  runs:      {len(data['runs'])}")
    print("\nStart the server and open the Workspace tab — you should see")
    print("1 fund manager, 4 clients and 4 portfolios with 3-run histories.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed portfolio_store.json with fabricated client data.")
    parser.add_argument(
        "--store",
        default=".",
        help="Directory containing (or to contain) portfolio_store.json  [default: current directory]",
    )
    parser.add_argument(
        "--signals",
        default="signal_store.csv",
        help="Path to signal_store.csv  [default: signal_store.csv in current directory]",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate portfolio_store.json instead of appending",
    )
    args = parser.parse_args()

    store_dir  = Path(args.store)
    store_path = store_dir / "portfolio_store.json"
    signal_path = Path(args.signals)

    if args.reset and store_path.exists():
        store_path.unlink()
        print(f"Deleted existing {store_path}")

    seed(store_path, signal_path)

"""
batch_optimiser.py
------------------
Runs optimize_portfolio / construct_portfolio for a list of client portfolios
concurrently (thread-pool) and records every result in outcome_store.

Usage (from main.py):
    from batch_optimiser import run_batch_optimization
    result = run_batch_optimization(manager_id, portfolio_ids, ...)
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from outcome_store import record_optimization_trades
from portfolio_store import (
    PortfolioNotFoundError,
    get_portfolio,
    record_optimization_run,
)


MAX_WORKERS = 4   # keep concurrency sane; optimizer is already CPU-heavy


def _new_run_id() -> str:
    return f"batch_{uuid4().hex[:12]}"


def _optimize_one(
    portfolio: Dict[str, Any],
    optimizer_module: Any,
) -> Dict[str, Any]:
    """
    Run the right optimizer function for a single saved portfolio.
    Returns a dict with keys: portfolio_id, success, result | error.
    """
    portfolio_id = portfolio["id"]
    try:
        if portfolio.get("consumer_has_portfolio", True):
            result = optimizer_module.optimize_portfolio(
                holdings=portfolio["holdings"],
                risk_profile=portfolio["risk_profile"],
                mandate_profile=portfolio["mandate_profile"],
                allow_new_stocks=portfolio["allow_new_stocks"],
                max_new_stocks=portfolio["max_new_stocks"],
                rebalance_frequency=portfolio["rebalance_frequency"],
                holding_period_days=portfolio["holding_period_days"],
            )
        else:
            result = optimizer_module.construct_portfolio(
                initial_cash_naira=portfolio.get("initial_cash_naira", 0),
                risk_profile=portfolio["risk_profile"],
                mandate_profile=portfolio["mandate_profile"],
                max_stocks=portfolio["max_new_stocks"],
                rebalance_frequency=portfolio["rebalance_frequency"],
                holding_period_days=portfolio["holding_period_days"],
            )
        return {"portfolio_id": portfolio_id, "success": True, "result": result}
    except Exception as exc:
        return {
            "portfolio_id": portfolio_id,
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_batch_optimization(
    manager_id: str,
    portfolio_ids: List[str],
    optimizer_module: Any,
    *,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, Any]:
    """
    Optimize all requested portfolios concurrently.

    Returns
    -------
    {
      "batch_run_id": "batch_...",
      "started_at":   "...",
      "finished_at":  "...",
      "total":  N,
      "succeeded": N,
      "failed":    N,
      "results": [
        {
          "portfolio_id": "...",
          "consumer_name": "...",
          "success": True,
          "run": { ...portfolio_store run record... },
          "result": { ...full optimizer output... },
        },
        {
          "portfolio_id": "...",
          "consumer_name": "...",
          "success": False,
          "error": "...",
        },
      ]
    }
    """
    batch_run_id = _new_run_id()
    started_at   = datetime.now(timezone.utc).isoformat()

    # Load portfolios (filter to ones that belong to this manager)
    portfolios: List[Dict[str, Any]] = []
    load_errors: List[Dict[str, Any]] = []
    for pid in portfolio_ids:
        try:
            pf = get_portfolio(pid)
            if pf.get("manager_id") != manager_id:
                load_errors.append({
                    "portfolio_id": pid,
                    "consumer_name": "",
                    "success": False,
                    "error": f"Portfolio {pid} does not belong to manager {manager_id}.",
                })
            else:
                portfolios.append(pf)
        except PortfolioNotFoundError as exc:
            load_errors.append({
                "portfolio_id": pid,
                "consumer_name": "",
                "success": False,
                "error": str(exc),
            })

    if not portfolios:
        return {
            "batch_run_id": batch_run_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total": len(portfolio_ids),
            "succeeded": 0,
            "failed": len(portfolio_ids),
            "results": load_errors,
        }

    outcome_results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(portfolios))) as pool:
        futures = {
            pool.submit(_optimize_one, pf, optimizer_module): pf
            for pf in portfolios
        }
        for future in as_completed(futures):
            pf = futures[future]
            item = future.result()
            outcome_results[pf["id"]] = item

    # Persist each successful result
    final_results: List[Dict[str, Any]] = []

    # Load errors first
    final_results.extend(load_errors)

    for pf in portfolios:
        pid = pf["id"]
        item = outcome_results[pid]
        consumer_name = pf.get("consumer_name", "")
        consumer_id   = pf.get("consumer_id", "")

        if not item["success"]:
            final_results.append({
                "portfolio_id": pid,
                "consumer_name": consumer_name,
                "success": False,
                "error": item["error"],
            })
            continue

        opt_result = item["result"]

        # 1. Record in portfolio_store (run history)
        try:
            run_record = record_optimization_run(pid, opt_result)
        except Exception as exc:
            run_record = {"error": str(exc)}

        # 2. Record individual trade rows in outcome_store
        try:
            trade_ids = record_optimization_trades(
                run_id=batch_run_id,
                portfolio_id=pid,
                manager_id=manager_id,
                consumer_id=consumer_id,
                consumer_name=consumer_name,
                risk_profile=pf.get("risk_profile", "balanced"),
                mandate_profile=pf.get("mandate_profile", "balanced_equity"),
                holding_period_days=pf.get("holding_period_days", 20),
                result=opt_result,
            )
        except Exception as exc:
            trade_ids = []

        final_results.append({
            "portfolio_id": pid,
            "consumer_name": consumer_name,
            "success": True,
            "run": run_record,
            "result": opt_result,
            "trade_ids_recorded": len(trade_ids),
        })

    succeeded = sum(1 for r in final_results if r["success"])
    failed    = len(final_results) - succeeded

    return {
        "batch_run_id": batch_run_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total": len(portfolio_ids),
        "succeeded": succeeded,
        "failed": failed,
        "results": final_results,
    }

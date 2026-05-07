import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
STORE_FILE = PROJECT_ROOT / "portfolio_store.json"
_LOCK = Lock()


class PortfolioStoreError(Exception):
    """Base portfolio store error."""


class PortfolioNotFoundError(PortfolioStoreError):
    """Requested manager or portfolio does not exist."""


class PortfolioStoreValidationError(PortfolioStoreError):
    """Invalid saved workspace data."""


def _now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _path(store_file: str | Path | None = None) -> Path:
    return Path(store_file) if store_file is not None else STORE_FILE


def _empty_store() -> Dict[str, Any]:
    return {"managers": [], "portfolios": [], "runs": []}


def _load_store(store_file: str | Path | None = None) -> Dict[str, Any]:
    path = _path(store_file)
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortfolioStoreError(f"Portfolio store is corrupted: {path}") from exc

    data.setdefault("managers", [])
    data.setdefault("portfolios", [])
    data.setdefault("runs", [])
    return data


def _save_store(data: Dict[str, Any], store_file: str | Path | None = None) -> None:
    path = _path(store_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _summary_from_result(result: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not result:
        return None
    return {
        "generated_at": result.get("fund_manager_report", {}).get("generated_at", _now()),
        "portfolio_value": result.get("current_portfolio_value", 0),
        "compliance_status": result.get("compliance_report", {}).get("overall_status", "unknown"),
        "optimized_expected_return": result.get("summary_metrics", {}).get("optimized_expected_return", 0),
        "optimized_sharpe": result.get("summary_metrics", {}).get("optimized_sharpe", 0),
        "added_symbols": result.get("added_symbols", []),
        "removed_symbols": result.get("removed_symbols", []),
    }


def list_managers(store_file: str | Path | None = None) -> List[Dict[str, Any]]:
    with _LOCK:
        return _load_store(store_file)["managers"]


def create_manager(
    name: str,
    firm: str,
    email: str = "",
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    name = str(name or "").strip()
    firm = str(firm or "").strip()
    email = str(email or "").strip()
    if not name:
        raise PortfolioStoreValidationError("Fund manager name is required.")
    if not firm:
        raise PortfolioStoreValidationError("Firm name is required.")

    with _LOCK:
        data = _load_store(store_file)
        manager = {
            "id": _new_id("mgr"),
            "name": name,
            "firm": firm,
            "email": email,
            "created_at": _now(),
            "updated_at": _now(),
        }
        data["managers"].append(manager)
        _save_store(data, store_file)
        return manager


def get_manager(manager_id: str, store_file: str | Path | None = None) -> Dict[str, Any]:
    with _LOCK:
        data = _load_store(store_file)
        for manager in data["managers"]:
            if manager["id"] == manager_id:
                return manager
    raise PortfolioNotFoundError(f"Fund manager not found: {manager_id}")


def list_portfolios(manager_id: str, store_file: str | Path | None = None) -> List[Dict[str, Any]]:
    get_manager(manager_id, store_file)
    with _LOCK:
        data = _load_store(store_file)
        portfolios = [p for p in data["portfolios"] if p["manager_id"] == manager_id]
        return sorted(portfolios, key=lambda item: item.get("updated_at", ""), reverse=True)


def create_portfolio(
    manager_id: str,
    name: str,
    holdings: List[Dict[str, Any]],
    risk_profile: str,
    mandate_profile: str,
    allow_new_stocks: bool,
    max_new_stocks: int,
    rebalance_frequency: str,
    holding_period_days: int,
    consumer_has_portfolio: bool = True,
    initial_cash_naira: float | None = None,
    latest_result: Dict[str, Any] | None = None,
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    get_manager(manager_id, store_file)
    name = str(name or "").strip()
    if not name:
        raise PortfolioStoreValidationError("Portfolio name is required.")
    initial_cash = float(initial_cash_naira or 0.0)
    if consumer_has_portfolio and not holdings:
        raise PortfolioStoreValidationError("Saved portfolio must contain at least one holding.")
    if not consumer_has_portfolio and initial_cash <= 0:
        raise PortfolioStoreValidationError("Initial cash amount is required for a new consumer portfolio.")

    now = _now()
    portfolio = {
        "id": _new_id("pf"),
        "manager_id": manager_id,
        "name": name,
        "consumer_has_portfolio": bool(consumer_has_portfolio),
        "initial_cash_naira": initial_cash if not consumer_has_portfolio else None,
        "holdings": holdings,
        "risk_profile": risk_profile,
        "mandate_profile": mandate_profile,
        "allow_new_stocks": bool(allow_new_stocks),
        "max_new_stocks": int(max_new_stocks),
        "rebalance_frequency": rebalance_frequency,
        "holding_period_days": int(holding_period_days),
        "latest_result_summary": _summary_from_result(latest_result),
        "created_at": now,
        "updated_at": now,
    }

    with _LOCK:
        data = _load_store(store_file)
        data["portfolios"].append(portfolio)
        if latest_result:
            data["runs"].append(
                {
                    "id": _new_id("run"),
                    "manager_id": manager_id,
                    "portfolio_id": portfolio["id"],
                    "created_at": now,
                    "result": latest_result,
                    "summary": _summary_from_result(latest_result),
                }
            )
        _save_store(data, store_file)
        return portfolio


def get_portfolio(portfolio_id: str, store_file: str | Path | None = None) -> Dict[str, Any]:
    with _LOCK:
        data = _load_store(store_file)
        for portfolio in data["portfolios"]:
            if portfolio["id"] == portfolio_id:
                return portfolio
    raise PortfolioNotFoundError(f"Portfolio not found: {portfolio_id}")


def record_optimization_run(
    portfolio_id: str,
    result: Dict[str, Any],
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    now = _now()
    with _LOCK:
        data = _load_store(store_file)
        for portfolio in data["portfolios"]:
            if portfolio["id"] == portfolio_id:
                run = {
                    "id": _new_id("run"),
                    "manager_id": portfolio["manager_id"],
                    "portfolio_id": portfolio_id,
                    "created_at": now,
                    "result": result,
                    "summary": _summary_from_result(result),
                }
                portfolio["latest_result_summary"] = run["summary"]
                portfolio["updated_at"] = now
                data["runs"].append(run)
                _save_store(data, store_file)
                return run
    raise PortfolioNotFoundError(f"Portfolio not found: {portfolio_id}")


def list_runs(portfolio_id: str, store_file: str | Path | None = None) -> List[Dict[str, Any]]:
    get_portfolio(portfolio_id, store_file)
    with _LOCK:
        data = _load_store(store_file)
        runs = [run for run in data["runs"] if run["portfolio_id"] == portfolio_id]
        return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)

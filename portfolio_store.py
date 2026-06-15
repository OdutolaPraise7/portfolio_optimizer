import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4


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
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _path(store_file: str | Path | None = None) -> Path:
    return Path(store_file) if store_file is not None else STORE_FILE


def _empty_store() -> Dict[str, Any]:
    return {"managers": [], "consumers": [], "portfolios": [], "runs": []}


def _load_store(store_file: str | Path | None = None) -> Dict[str, Any]:
    path = _path(store_file)
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortfolioStoreError(f"Portfolio store is corrupted: {path}") from exc

    data.setdefault("managers", [])
    data.setdefault("consumers", [])
    data.setdefault("portfolios", [])
    data.setdefault("runs", [])
    return data


def _save_store(data: Dict[str, Any], store_file: str | Path | None = None) -> None:
    path = _path(store_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return key.hex() == key_hex
    except Exception:
        return False


def _strip_password(manager: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in manager.items() if k != "password_hash"}


def _summary_from_result(result: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not result:
        return None
    return {
        "generated_at": result.get("fund_manager_report", {}).get("generated_at", _now()),
        "portfolio_value": result.get("current_portfolio_value", 0),
        "optimized_portfolio_value": result.get(
            "optimized_portfolio_value",
            result.get("current_portfolio_value", 0),
        ),
        "compliance_status": result.get("compliance_report", {}).get("overall_status", "unknown"),
        "optimized_expected_return": result.get("summary_metrics", {}).get("optimized_expected_return", 0),
        "optimized_sharpe": result.get("summary_metrics", {}).get("optimized_sharpe", 0),
        "added_symbols": result.get("added_symbols", []),
        "removed_symbols": result.get("removed_symbols", []),
    }


def list_managers(store_file: str | Path | None = None) -> List[Dict[str, Any]]:
    with _LOCK:
        return [_strip_password(m) for m in _load_store(store_file)["managers"]]


def create_manager(
    name: str,
    firm: str,
    email: str,
    password: str,
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    name = str(name or "").strip()
    firm = str(firm or "").strip()
    email = str(email or "").strip().lower()
    if not name:
        raise PortfolioStoreValidationError("Fund manager name is required.")
    if not firm:
        raise PortfolioStoreValidationError("Firm name is required.")
    if not email:
        raise PortfolioStoreValidationError("Email address is required.")
    if len(password) < 8:
        raise PortfolioStoreValidationError("Password must be at least 8 characters.")

    with _LOCK:
        data = _load_store(store_file)
        if any(m.get("email", "").lower() == email for m in data["managers"]):
            raise PortfolioStoreValidationError("An account with this email already exists.")
        manager = {
            "id": _new_id("mgr"),
            "name": name,
            "firm": firm,
            "email": email,
            "password_hash": _hash_password(password),
            "created_at": _now(),
            "updated_at": _now(),
        }
        data["managers"].append(manager)
        _save_store(data, store_file)
        return _strip_password(manager)


def authenticate_manager(
    email: str,
    password: str,
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    email = str(email or "").strip().lower()
    with _LOCK:
        data = _load_store(store_file)
        manager = next(
            (m for m in data["managers"] if m.get("email", "").lower() == email),
            None,
        )
    if manager is None or not _verify_password(password, manager.get("password_hash", "")):
        raise PortfolioStoreValidationError("Invalid email or password.")
    return _strip_password(manager)


def get_manager(manager_id: str, store_file: str | Path | None = None) -> Dict[str, Any]:
    with _LOCK:
        data = _load_store(store_file)
        for manager in data["managers"]:
            if manager["id"] == manager_id:
                return _strip_password(manager)
    raise PortfolioNotFoundError(f"Fund manager not found: {manager_id}")


def delete_manager(manager_id: str, store_file: str | Path | None = None) -> Dict[str, Any]:
    with _LOCK:
        data = _load_store(store_file)
        manager = next((item for item in data["managers"] if item["id"] == manager_id), None)
        if manager is None:
            raise PortfolioNotFoundError(f"Fund manager not found: {manager_id}")

        portfolio_ids = {
            portfolio["id"]
            for portfolio in data["portfolios"]
            if portfolio.get("manager_id") == manager_id
        }
        data["managers"] = [item for item in data["managers"] if item["id"] != manager_id]
        data["consumers"] = [
            consumer for consumer in data["consumers"]
            if consumer.get("manager_id") != manager_id
        ]
        data["portfolios"] = [
            portfolio for portfolio in data["portfolios"]
            if portfolio.get("manager_id") != manager_id
        ]
        data["runs"] = [
            run for run in data["runs"]
            if run.get("manager_id") != manager_id and run.get("portfolio_id") not in portfolio_ids
        ]
        _save_store(data, store_file)
        return {
            "manager": manager,
            "deleted_portfolios": len(portfolio_ids),
        }


def list_consumers(manager_id: str, store_file: str | Path | None = None) -> List[Dict[str, Any]]:
    get_manager(manager_id, store_file)
    with _LOCK:
        data = _load_store(store_file)
        consumers = [consumer for consumer in data["consumers"] if consumer["manager_id"] == manager_id]
        portfolios = [portfolio for portfolio in data["portfolios"] if portfolio["manager_id"] == manager_id]
        portfolio_counts: Dict[str, int] = {}
        for portfolio in portfolios:
            consumer_id = portfolio.get("consumer_id")
            if consumer_id:
                portfolio_counts[consumer_id] = portfolio_counts.get(consumer_id, 0) + 1

        enriched = []
        for consumer in consumers:
            item = dict(consumer)
            item["portfolio_count"] = portfolio_counts.get(consumer["id"], 0)
            enriched.append(item)
        return sorted(enriched, key=lambda item: item.get("updated_at", ""), reverse=True)


def create_consumer(
    manager_id: str,
    name: str,
    email: str = "",
    consumer_has_portfolio: bool = True,
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    get_manager(manager_id, store_file)
    name = str(name or "").strip()
    email = str(email or "").strip()
    if not name:
        raise PortfolioStoreValidationError("Consumer name is required.")

    now = _now()
    consumer = {
        "id": _new_id("cons"),
        "manager_id": manager_id,
        "name": name,
        "email": email,
        "consumer_has_portfolio": bool(consumer_has_portfolio),
        "created_at": now,
        "updated_at": now,
    }
    with _LOCK:
        data = _load_store(store_file)
        data["consumers"].append(consumer)
        _save_store(data, store_file)
        return consumer


def delete_consumer(
    manager_id: str,
    consumer_id: str,
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    get_manager(manager_id, store_file)
    with _LOCK:
        data = _load_store(store_file)
        consumer = next(
            (
                item for item in data["consumers"]
                if item["id"] == consumer_id and item["manager_id"] == manager_id
            ),
            None,
        )
        if consumer is None:
            raise PortfolioNotFoundError(f"Consumer not found: {consumer_id}")

        portfolio_ids = {
            portfolio["id"]
            for portfolio in data["portfolios"]
            if portfolio.get("manager_id") == manager_id and portfolio.get("consumer_id") == consumer_id
        }
        data["consumers"] = [
            item for item in data["consumers"]
            if not (item["id"] == consumer_id and item["manager_id"] == manager_id)
        ]
        data["portfolios"] = [
            portfolio for portfolio in data["portfolios"]
            if not (portfolio.get("manager_id") == manager_id and portfolio.get("consumer_id") == consumer_id)
        ]
        data["runs"] = [
            run for run in data["runs"]
            if run.get("portfolio_id") not in portfolio_ids
        ]
        _save_store(data, store_file)
        return {
            "consumer": consumer,
            "deleted_portfolios": len(portfolio_ids),
        }


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
    consumer_id: str = "",
    consumer_name: str = "",
    consumer_email: str = "",
    latest_result: Dict[str, Any] | None = None,
    store_file: str | Path | None = None,
) -> Dict[str, Any]:
    get_manager(manager_id, store_file)
    name = str(name or "").strip()
    consumer_name = str(consumer_name or "").strip()
    consumer_email = str(consumer_email or "").strip()
    if not name:
        raise PortfolioStoreValidationError("Portfolio name is required.")
    if not consumer_name:
        raise PortfolioStoreValidationError("Consumer name is required.")
    initial_cash = float(initial_cash_naira or 0.0)
    if consumer_has_portfolio and not holdings:
        raise PortfolioStoreValidationError("Saved portfolio must contain at least one holding.")
    if not consumer_has_portfolio and initial_cash <= 0:
        raise PortfolioStoreValidationError("Initial cash amount is required for a new consumer portfolio.")

    now = _now()
    with _LOCK:
        data = _load_store(store_file)
        resolved_consumer_id = str(consumer_id or "").strip()
        if resolved_consumer_id:
            consumer = next(
                (
                    item for item in data["consumers"]
                    if item["id"] == resolved_consumer_id and item["manager_id"] == manager_id
                ),
                None,
            )
            if consumer is None:
                raise PortfolioNotFoundError(f"Consumer not found: {resolved_consumer_id}")
            consumer["name"] = consumer_name
            consumer["email"] = consumer_email
            consumer["consumer_has_portfolio"] = bool(consumer_has_portfolio)
            consumer["updated_at"] = now
        else:
            consumer = {
                "id": _new_id("cons"),
                "manager_id": manager_id,
                "name": consumer_name,
                "email": consumer_email,
                "consumer_has_portfolio": bool(consumer_has_portfolio),
                "created_at": now,
                "updated_at": now,
            }
            data["consumers"].append(consumer)
            resolved_consumer_id = consumer["id"]

        portfolio = {
            "id": _new_id("pf"),
            "manager_id": manager_id,
            "consumer_id": resolved_consumer_id,
            "name": name,
            "consumer_has_portfolio": bool(consumer_has_portfolio),
            "consumer_name": consumer_name,
            "consumer_email": consumer_email,
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

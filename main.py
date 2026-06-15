from functools import lru_cache
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app_snapshots import (
    SnapshotError,
    get_latest_price_snapshot,
    get_signal_summary,
    get_signal_watchlist,
    get_supported_symbols,
)
from outcome_store import (
    export_trades_csv,
    get_symbol_alpha_signal,
    get_symbol_outcome_history,
    list_outcome_trades,
    record_optimization_trades,
    record_price_snapshot,
)
from batch_optimiser import run_batch_optimization
from portfolio_store import (
    PortfolioNotFoundError,
    PortfolioStoreError,
    PortfolioStoreValidationError,
    apply_optimised_holdings,
    authenticate_manager,
    create_consumer,
    create_manager,
    create_portfolio,
    delete_consumer,
    delete_manager,
    get_portfolio,
    list_consumers,
    list_managers,
    list_portfolios,
    list_runs,
    record_optimization_run,
)


@lru_cache(maxsize=1)
def _optimizer_module():
    # pandas/numpy are expensive to import on small cloud instances. Load the
    # optimizer only for endpoints that actually run portfolio construction.
    import portfolio_optimiser

    return portfolio_optimiser


def optimize_portfolio(
    holdings,
    risk_profile="balanced",
    allow_new_stocks=True,
    max_new_stocks=5,
    price_file=None,
    signal_file=None,
    stale_after_hours=None,
    rebalance_frequency="monthly",
    holding_period_days=20,
    mandate_profile="balanced_equity",
    construction_amount_naira=None,
):
    optimizer = _optimizer_module()
    kwargs = {
        "holdings": holdings,
        "risk_profile": risk_profile,
        "allow_new_stocks": allow_new_stocks,
        "max_new_stocks": max_new_stocks,
        "rebalance_frequency": rebalance_frequency,
        "holding_period_days": holding_period_days,
        "mandate_profile": mandate_profile,
        "construction_amount_naira": construction_amount_naira,
    }
    if price_file is not None:
        kwargs["price_file"] = price_file
    if signal_file is not None:
        kwargs["signal_file"] = signal_file
    if stale_after_hours is not None:
        kwargs["stale_after_hours"] = stale_after_hours
    return optimizer.optimize_portfolio(**kwargs)


def construct_portfolio(
    initial_cash_naira,
    risk_profile="balanced",
    max_stocks=None,
    price_file=None,
    signal_file=None,
    stale_after_hours=None,
    rebalance_frequency="monthly",
    holding_period_days=20,
    mandate_profile="balanced_equity",
):
    optimizer = _optimizer_module()
    kwargs = {
        "initial_cash_naira": initial_cash_naira,
        "risk_profile": risk_profile,
        "max_stocks": max_stocks,
        "rebalance_frequency": rebalance_frequency,
        "holding_period_days": holding_period_days,
        "mandate_profile": mandate_profile,
    }
    if price_file is not None:
        kwargs["price_file"] = price_file
    if signal_file is not None:
        kwargs["signal_file"] = signal_file
    if stale_after_hours is not None:
        kwargs["stale_after_hours"] = stale_after_hours
    return optimizer.construct_portfolio(**kwargs)


app = FastAPI(
    title="Portfolio Optimizer API",
    version="1.0.0",
    description="Backend-first API for signal-aware NSE portfolio optimization.",
)

# During development the React app runs on a different local port, so CORS
# must allow the browser to call the FastAPI backend from that origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://portfolio-optimizer-zeta.vercel.app",
        "https://portfolio-optimizer-site.onrender.com"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HoldingInput(BaseModel):
    # Each holding is entered as a ticker plus naira value, not share count.
    symbol: str = Field(..., description="NSE equity ticker symbol")
    amount_naira: float = Field(..., gt=0, description="Current holding value in naira")


class PortfolioRequest(BaseModel):
    # This schema mirrors the frontend form, so the UI can POST directly.
    holdings: List[HoldingInput]
    risk_profile: Literal["conservative", "balanced", "aggressive"] = "balanced"
    mandate_profile: Literal[
        "balanced_equity",
        "growth_equity",
        "income_equity",
        "pension_equity",
    ] = "balanced_equity"
    allow_new_stocks: bool = True
    max_new_stocks: int = Field(5, ge=0, le=20)
    rebalance_frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    holding_period_days: int = Field(20, ge=1, le=252)


class ConstructPortfolioRequest(BaseModel):
    initial_cash_naira: float = Field(..., gt=0, description="Cash available to build a new portfolio")
    risk_profile: Literal["conservative", "balanced", "aggressive"] = "balanced"
    mandate_profile: Literal[
        "balanced_equity",
        "growth_equity",
        "income_equity",
        "pension_equity",
    ] = "balanced_equity"
    max_stocks: int | None = Field(None, ge=1, le=20)
    rebalance_frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    holding_period_days: int = Field(20, ge=1, le=252)


class ManagerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Fund manager name")
    firm: str = Field(..., min_length=1, description="Fund management firm")
    email: str = Field(..., min_length=3, description="Email address (used for login)")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")


class ManagerLoginRequest(BaseModel):
    email: str
    password: str


class ConsumerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Consumer or client name")
    email: str = Field("", description="Optional consumer contact")
    consumer_has_portfolio: bool = True


class SavedPortfolioRequest(PortfolioRequest):
    name: str = Field(..., min_length=1, description="Saved portfolio name")
    consumer_id: str = Field("", description="Optional saved consumer id")
    consumer_name: str = Field(..., min_length=1, description="Consumer or client name")
    consumer_email: str = Field("", description="Optional consumer contact")
    consumer_has_portfolio: bool = True
    initial_cash_naira: float | None = Field(None, ge=0)
    latest_result: Dict[str, Any] | None = None


def _model_to_dict(model: BaseModel) -> dict:
    # Support both Pydantic v2 (`model_dump`) and v1 (`dict`) so the API keeps
    # working across different virtual environments.
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _optimize_from_payload(payload: PortfolioRequest) -> dict:
    return optimize_portfolio(
        holdings=[_model_to_dict(holding) for holding in payload.holdings],
        risk_profile=payload.risk_profile,
        mandate_profile=payload.mandate_profile,
        allow_new_stocks=payload.allow_new_stocks,
        max_new_stocks=payload.max_new_stocks,
        rebalance_frequency=payload.rebalance_frequency,
        holding_period_days=payload.holding_period_days,
    )


def _construct_from_payload(payload: ConstructPortfolioRequest) -> dict:
    return construct_portfolio(
        initial_cash_naira=payload.initial_cash_naira,
        risk_profile=payload.risk_profile,
        mandate_profile=payload.mandate_profile,
        max_stocks=payload.max_stocks,
        rebalance_frequency=payload.rebalance_frequency,
        holding_period_days=payload.holding_period_days,
    )


def _handle_store_error(exc: PortfolioStoreError) -> HTTPException:
    if isinstance(exc, PortfolioNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, PortfolioStoreValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/symbols")
def list_symbols() -> dict:
    # The UI uses this to populate the stock dropdown.
    return {"symbols": get_supported_symbols()}


@app.get("/prices/latest")
def latest_prices() -> dict:
    # The frontend uses latest prices to convert share quantities into naira values.
    return get_latest_price_snapshot()


@app.get("/signals/summary")
def signals_summary() -> dict:
    try:
        return get_signal_summary()
    except SnapshotError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/signals/watchlist")
def signals_watchlist() -> dict:
    try:
        return get_signal_watchlist()
    except SnapshotError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/bootstrap")
def bootstrap() -> dict:
    try:
        market = get_latest_price_snapshot()
        payload = {
            "symbols": get_supported_symbols(),
            "prices": market["prices"],
            "price_updated_at": market["updated_at"],
        }
        try:
            payload["signal_summary"] = get_signal_summary()
            payload["watchlist"] = get_signal_watchlist()
        except SnapshotError as exc:
            payload["signal_error"] = str(exc)
            payload["signal_summary"] = None
            payload["watchlist"] = None
        return payload
    except SnapshotError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/optimize-portfolio")
def optimize_portfolio_endpoint(payload: PortfolioRequest) -> dict:
    optimizer = _optimizer_module()
    try:
        # The API layer is intentionally thin: it validates request shape,
        # then hands the heavy lifting to the optimizer module.
        return _optimize_from_payload(payload)
    except optimizer.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except optimizer.SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/construct-portfolio")
def construct_portfolio_endpoint(payload: ConstructPortfolioRequest) -> dict:
    optimizer = _optimizer_module()
    try:
        return _construct_from_payload(payload)
    except optimizer.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except optimizer.SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/fund-managers")
def fund_managers() -> dict:
    return {"managers": list_managers()}


@app.post("/fund-managers")
def create_fund_manager(payload: ManagerCreateRequest) -> dict:
    try:
        return {"manager": create_manager(payload.name, payload.firm, payload.email, payload.password)}
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.post("/fund-managers/login")
def login_fund_manager(payload: ManagerLoginRequest) -> dict:
    try:
        return {"manager": authenticate_manager(payload.email, payload.password)}
    except PortfolioStoreError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.delete("/fund-managers/{manager_id}")
def delete_fund_manager(manager_id: str) -> dict:
    try:
        return delete_manager(manager_id)
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.get("/fund-managers/{manager_id}/consumers")
def manager_consumers(manager_id: str) -> dict:
    try:
        return {"consumers": list_consumers(manager_id)}
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.post("/fund-managers/{manager_id}/consumers")
def create_manager_consumer(manager_id: str, payload: ConsumerCreateRequest) -> dict:
    try:
        return {
            "consumer": create_consumer(
                manager_id=manager_id,
                name=payload.name,
                email=payload.email,
                consumer_has_portfolio=payload.consumer_has_portfolio,
            )
        }
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.delete("/fund-managers/{manager_id}/consumers/{consumer_id}")
def delete_manager_consumer(manager_id: str, consumer_id: str) -> dict:
    try:
        return delete_consumer(manager_id, consumer_id)
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.get("/fund-managers/{manager_id}/portfolios")
def manager_portfolios(manager_id: str) -> dict:
    try:
        return {"portfolios": list_portfolios(manager_id)}
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.post("/fund-managers/{manager_id}/portfolios")
def save_manager_portfolio(manager_id: str, payload: SavedPortfolioRequest) -> dict:
    try:
        return {
            "portfolio": create_portfolio(
                manager_id=manager_id,
                name=payload.name,
                holdings=[_model_to_dict(holding) for holding in payload.holdings],
                risk_profile=payload.risk_profile,
                mandate_profile=payload.mandate_profile,
                allow_new_stocks=payload.allow_new_stocks,
                max_new_stocks=payload.max_new_stocks,
                rebalance_frequency=payload.rebalance_frequency,
                holding_period_days=payload.holding_period_days,
                consumer_has_portfolio=payload.consumer_has_portfolio,
                initial_cash_naira=payload.initial_cash_naira,
                consumer_id=payload.consumer_id,
                consumer_name=payload.consumer_name,
                consumer_email=payload.consumer_email,
                latest_result=payload.latest_result,
            )
        }
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.get("/portfolios/{portfolio_id}")
def saved_portfolio(portfolio_id: str) -> dict:
    try:
        portfolio = get_portfolio(portfolio_id)
        return {"portfolio": portfolio, "runs": list_runs(portfolio_id)}
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


@app.post("/portfolios/{portfolio_id}/optimize")
def optimize_saved_portfolio(portfolio_id: str) -> dict:
    optimizer = _optimizer_module()
    try:
        portfolio = get_portfolio(portfolio_id)
        if portfolio.get("consumer_has_portfolio", True):
            result = optimizer.optimize_portfolio(
                holdings=portfolio["holdings"],
                risk_profile=portfolio["risk_profile"],
                mandate_profile=portfolio["mandate_profile"],
                allow_new_stocks=portfolio["allow_new_stocks"],
                max_new_stocks=portfolio["max_new_stocks"],
                rebalance_frequency=portfolio["rebalance_frequency"],
                holding_period_days=portfolio["holding_period_days"],
            )
        else:
            result = optimizer.construct_portfolio(
                initial_cash_naira=portfolio.get("initial_cash_naira", 0),
                risk_profile=portfolio["risk_profile"],
                mandate_profile=portfolio["mandate_profile"],
                max_stocks=portfolio["max_new_stocks"],
                rebalance_frequency=portfolio["rebalance_frequency"],
                holding_period_days=portfolio["holding_period_days"],
            )
        run = record_optimization_run(portfolio_id, result)
        return {"result": result, "run": run}
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc
    except optimizer.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except optimizer.SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class ApplyOptimisedRequest(BaseModel):
    result: Dict[str, Any] = Field(..., description="Optimization result whose allocations become the new holdings")


@app.post("/portfolios/{portfolio_id}/apply-optimised")
def apply_optimised_portfolio(portfolio_id: str, payload: ApplyOptimisedRequest) -> dict:
    """Replace the saved portfolio's holdings with the optimised allocations and record a run."""
    try:
        result = payload.result
        allocations = result.get("optimized_allocations", [])
        portfolio_value = result.get("optimized_portfolio_value") or result.get("current_portfolio_value", 0)
        optimised_holdings = [
            {"symbol": a["symbol"], "amount_naira": a["optimized_weight"] * portfolio_value}
            for a in allocations
            if a.get("action") != "exit"
        ]
        portfolio = apply_optimised_holdings(portfolio_id, optimised_holdings, result)
        return {"portfolio": portfolio}
    except PortfolioStoreError as exc:
        raise _handle_store_error(exc) from exc


# ── BATCH OPTIMIZATION ────────────────────────────────────────────────────────

class BatchOptimizeRequest(BaseModel):
    portfolio_ids: List[str] = Field(..., min_length=1, description="List of saved portfolio IDs to optimize in one batch")


@app.post("/fund-managers/{manager_id}/batch-optimize")
def batch_optimize(manager_id: str, payload: BatchOptimizeRequest) -> dict:
    """
    Optimize all selected client portfolios at once.
    Results are persisted to portfolio_store (run history) and outcome_store
    (trade-level outcome tracking for cross-client learning).
    """
    optimizer = _optimizer_module()
    try:
        return run_batch_optimization(
            manager_id=manager_id,
            portfolio_ids=payload.portfolio_ids,
            optimizer_module=optimizer,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── OUTCOME STORE ─────────────────────────────────────────────────────────────

@app.get("/outcomes/trades")
def outcome_trades(
    manager_id: str | None = None,
    portfolio_id: str | None = None,
    symbol: str | None = None,
    settled_only: bool = False,
    limit: int = 500,
) -> dict:
    """
    Query recorded trade outcomes.  Supports filtering by manager, portfolio,
    symbol, and whether the outcome (realised return) has been captured yet.
    """
    trades = list_outcome_trades(
        manager_id=manager_id,
        portfolio_id=portfolio_id,
        symbol=symbol.upper() if symbol else None,
        settled_only=settled_only,
        limit=limit,
    )
    return {"trades": trades, "count": len(trades)}


@app.get("/outcomes/symbol/{symbol}")
def symbol_outcome_history(symbol: str) -> dict:
    """All recorded trade rows for a single ticker."""
    return {
        "symbol": symbol.upper(),
        "trades": get_symbol_outcome_history(symbol),
        "alpha_signal": get_symbol_alpha_signal(symbol),
    }


@app.get("/outcomes/alpha/{symbol}")
def symbol_alpha(symbol: str) -> dict:
    """
    Aggregated historical accuracy for a symbol.
    Use this to answer: 'did Chevron positions pay off in the past?'
    hit_rate, mean_realised_return, mean_prediction_error, etc.
    """
    return get_symbol_alpha_signal(symbol)


@app.post("/outcomes/record-prices")
def record_current_prices() -> dict:
    """
    Capture today's prices from PRICE_LIST and resolve any open trades whose
    holding period has elapsed.  Call this once a day (e.g. via a cron job or
    a button in the UI).
    """
    try:
        snapshot = get_latest_price_snapshot()
        prices = snapshot.get("prices", {})
        updated = record_price_snapshot(prices, capture_date=None)
        return {
            "status": "ok",
            "prices_captured": len(prices),
            "trades_resolved": updated,
            "updated_at": snapshot.get("updated_at"),
        }
    except SnapshotError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/outcomes/export-csv")
def outcomes_export_csv() -> dict:
    """Force-sync the CSV mirror from the DB and return its path."""
    path = export_trades_csv()
    return {"csv_path": str(path)}

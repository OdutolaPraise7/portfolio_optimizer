from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from portfolio_optimiser import (
    SignalStoreError,
    ValidationError,
    construct_portfolio,
    get_latest_price_snapshot,
    get_signal_summary,
    get_signal_watchlist,
    get_supported_symbols,
    optimize_portfolio,
)
from portfolio_store import (
    PortfolioNotFoundError,
    PortfolioStoreError,
    PortfolioStoreValidationError,
    create_manager,
    create_portfolio,
    get_portfolio,
    list_managers,
    list_portfolios,
    list_runs,
    record_optimization_run,
)


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
    max_stocks: int = Field(8, ge=1, le=20)
    rebalance_frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    holding_period_days: int = Field(20, ge=1, le=252)


class ManagerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Fund manager name")
    firm: str = Field(..., min_length=1, description="Fund management firm")
    email: str = Field("", description="Optional email or internal contact")


class SavedPortfolioRequest(PortfolioRequest):
    name: str = Field(..., min_length=1, description="Saved portfolio name")
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
    except SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/signals/watchlist")
def signals_watchlist() -> dict:
    try:
        return get_signal_watchlist()
    except SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/optimize-portfolio")
def optimize_portfolio_endpoint(payload: PortfolioRequest) -> dict:
    try:
        # The API layer is intentionally thin: it validates request shape,
        # then hands the heavy lifting to the optimizer module.
        return _optimize_from_payload(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/construct-portfolio")
def construct_portfolio_endpoint(payload: ConstructPortfolioRequest) -> dict:
    try:
        return _construct_from_payload(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/fund-managers")
def fund_managers() -> dict:
    return {"managers": list_managers()}


@app.post("/fund-managers")
def create_fund_manager(payload: ManagerCreateRequest) -> dict:
    try:
        return {"manager": create_manager(payload.name, payload.firm, payload.email)}
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
    try:
        portfolio = get_portfolio(portfolio_id)
        if portfolio.get("consumer_has_portfolio", True):
            result = optimize_portfolio(
                holdings=portfolio["holdings"],
                risk_profile=portfolio["risk_profile"],
                mandate_profile=portfolio["mandate_profile"],
                allow_new_stocks=portfolio["allow_new_stocks"],
                max_new_stocks=portfolio["max_new_stocks"],
                rebalance_frequency=portfolio["rebalance_frequency"],
                holding_period_days=portfolio["holding_period_days"],
            )
        else:
            result = construct_portfolio(
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
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SignalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

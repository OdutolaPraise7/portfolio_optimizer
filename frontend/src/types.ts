export type RiskProfile = 'conservative' | 'balanced' | 'aggressive';
export type MandateProfile = 'balanced_equity' | 'growth_equity' | 'income_equity' | 'pension_equity';

// These frontend types mirror the FastAPI request/response contract.
export type Holding = {
  symbol: string;
  quantity: number;
};

export type FundManager = {
  id: string;
  name: string;
  firm: string;
  email: string;
  created_at: string;
  updated_at: string;
};

export type SavedHolding = {
  symbol: string;
  amount_naira: number;
};

export type OptimizationRunSummary = {
  generated_at: string;
  portfolio_value: number;
  compliance_status: string;
  optimized_expected_return: number;
  optimized_sharpe: number;
  added_symbols: string[];
  removed_symbols: string[];
};

export type SavedPortfolio = {
  id: string;
  manager_id: string;
  name: string;
  consumer_has_portfolio?: boolean;
  initial_cash_naira?: number | null;
  holdings: SavedHolding[];
  risk_profile: RiskProfile;
  mandate_profile: MandateProfile;
  allow_new_stocks: boolean;
  max_new_stocks: number;
  rebalance_frequency: 'weekly' | 'monthly' | 'quarterly';
  holding_period_days: number;
  latest_result_summary: OptimizationRunSummary | null;
  created_at: string;
  updated_at: string;
};

export type SignalSummary = {
  path: string;
  row_count: number;
  generated_at: string;
  buy_count: number;
  sell_count: number;
  conflict_count: number;
  avg_confidence: number;
  avg_r2: number;
};

export type SignalWatchlistItem = {
  symbol: string;
  signal: string;
  sector: string;
  avg_return: number;
  avg_confidence: number;
  avg_r2: number;
  signal_score: number;
  reason: string;
};

export type SignalWatchlist = {
  top_buys: SignalWatchlistItem[];
  top_sells: SignalWatchlistItem[];
};

export type LatestPricesResponse = {
  prices: Record<string, number>;
  updated_at: string | null;
};

export type OptimizedAllocation = {
  symbol: string;
  sector: string;
  current_weight: number;
  optimized_weight: number;
  weight_delta: number;
  action: 'keep' | 'increase' | 'reduce' | 'add' | 'exit';
  latest_price: number | null;
  expected_return: number;
  signal_status: string;
  consensus_tier: number | null;
  avg_confidence: number | null;
  avg_r2: number | null;
  signal_score: number;
  avg_volume_20d: number;
  avg_trade_value_20d: number;
  volatility_20d: number;
  liquidity_score: number;
  model_votes: {
    model: string;
    signal: string;
    expected_return: number;
    confidence: number;
    r2: number;
    quality_pass: boolean;
  }[];
};

export type CurrentWeight = {
  symbol: string;
  amount_naira: number;
  weight: number;
  sector: string;
};

export type SectorAllocation = {
  sector: string;
  current_weight: number;
  optimized_weight: number;
};

export type StrategyBacktest = {
  cumulative_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  max_drawdown: number;
};

export type OptimizationResponse = {
  portfolio_mode?: 'optimization' | 'construction';
  risk_profile: RiskProfile;
  mandate_profile: MandateProfile;
  mandate_summary: {
    label: string;
    objective: string;
    benchmark: string;
    max_stock_weight: number;
    max_sector_weight: number;
    min_liquidity_score: number;
    max_turnover: number;
    max_portfolio_volatility: number | null;
  };
  prediction_engine: {
    scope: string;
    models: string[];
    symbols_scored: number;
    buy_count: number;
    sell_count: number;
    conflict_count: number;
    average_confidence: number;
    average_r2: number;
    qualified_model_coverage: number;
  };
  allow_new_stocks: boolean;
  max_new_stocks: number;
  rebalance_frequency: 'weekly' | 'monthly' | 'quarterly';
  holding_period_days: number;
  current_portfolio_value: number;
  initial_cash_naira?: number | null;
  current_weights: CurrentWeight[];
  optimized_allocations: OptimizedAllocation[];
  added_symbols: string[];
  removed_symbols: string[];
  sector_allocations: SectorAllocation[];
  constraint_summary: {
    max_stock_weight: number;
    max_sector_weight: number;
    turnover: number;
    transaction_cost_rate: number;
    estimated_transaction_cost_naira: number;
    liquidity_screened_candidates: number;
    no_trade_band: number;
  };
  compliance_report: {
    overall_status: 'pass' | 'review' | 'breach';
    mandate_profile: MandateProfile;
    mandate_label: string;
    checked_at: string;
    items: {
      rule: string;
      status: 'pass' | 'warn' | 'breach';
      observed: number | string;
      limit: number | string;
      message: string;
    }[];
  };
  benchmark_metrics: {
    expected_return: number;
    volatility: number;
    sharpe: number;
    sortino: number;
    cvar_95: number;
    max_drawdown: number;
    tracking_error: number;
    information_ratio: number;
    annualized_realized_return: number;
  };
  backtest_summary: {
    window_days: number;
    rebalance_frequency: 'weekly' | 'monthly' | 'quarterly';
    winner: string;
    strategies: {
      current_portfolio: StrategyBacktest;
      optimized_portfolio: StrategyBacktest;
      equal_weight: StrategyBacktest;
      benchmark: StrategyBacktest;
    };
  };
  fund_manager_report: {
    title: string;
    market: string;
    mandate_profile: MandateProfile;
    mandate_label: string;
    objective: string;
    benchmark: string;
    generated_at: string;
    recommendation: string;
    summary: {
      current_expected_return: number;
      optimized_expected_return: number;
      current_sharpe: number;
      optimized_sharpe: number;
      added_symbols: string[];
      removed_symbols: string[];
      compliance_status: 'pass' | 'review' | 'breach';
    };
  };
  summary_metrics: {
    candidate_count: number;
    current_expected_return: number;
    current_volatility: number;
    current_sharpe: number;
    current_sortino: number;
    current_cvar_95: number;
    current_max_drawdown: number;
    optimized_expected_return: number;
    optimized_volatility: number;
    optimized_sharpe: number;
    optimized_sortino: number;
    optimized_cvar_95: number;
    optimized_max_drawdown: number;
    optimized_tracking_error: number;
    optimized_information_ratio: number;
    optimization_objective_score: number;
  };
};

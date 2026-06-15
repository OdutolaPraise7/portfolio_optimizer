export type RiskProfile = 'conservative' | 'balanced' | 'aggressive';
export type MandateProfile = 'balanced_equity' | 'growth_equity' | 'income_equity' | 'pension_equity';

export interface Holding {
  symbol: string;
  quantity: number;
}

export interface FundManager {
  id: string;
  name: string;
  firm: string;
  email?: string;
}

export interface ManagedConsumer {
  id: string;
  manager_id: string;
  name: string;
  email?: string;
  consumer_has_portfolio: boolean;
  portfolio_count?: number;
}

export interface SignalSummary {
  buy_count: number;
  sell_count: number;
  avg_confidence: number;
  generated_at: string;
  row_count?: number;
}

export interface WatchlistItem {
  symbol: string;
  reason: string;
  avg_confidence: number;
  signal?: string;
  sector?: string;
  avg_return?: number;
  avg_r2?: number;
  signal_score?: number;
}

export interface SignalWatchlist {
  top_buys: WatchlistItem[];
  top_sells: WatchlistItem[];
}

export interface ModelVote {
  model: string;
  signal: string;
}

export interface OptimizedAllocation {
  symbol: string;
  sector: string;
  signal_status: 'BUY' | 'SELL' | 'CONFLICT' | 'HISTORICAL';
  consensus_tier: number | null;
  avg_confidence: number | null;
  signal_score: number;
  expected_return: number;
  current_weight: number;
  optimized_weight: number;
  action: 'add' | 'increase' | 'reduce' | 'keep' | 'exit';
  model_votes: ModelVote[];
  latest_price?: number | null;
  liquidity_score: number;
  avg_trade_value_20d: number;
  avg_r2: number | null;
}

export interface LatestResultSummary {
  generated_at: string;
  portfolio_value: number;
  optimized_portfolio_value?: number;
  compliance_status: 'pass' | 'review' | 'breach' | 'warn';
  optimized_expected_return: number;
  optimized_sharpe: number;
  added_symbols: string[];
  removed_symbols: string[];
}

export interface SavedPortfolio {
  id: string;
  manager_id: string;
  consumer_id?: string;
  name: string;
  consumer_name?: string;
  consumer_email?: string;
  consumer_has_portfolio?: boolean;
  initial_cash_naira?: number | null;
  holdings: { symbol: string; amount_naira: number }[];
  risk_profile: RiskProfile;
  mandate_profile: MandateProfile;
  allow_new_stocks: boolean;
  max_new_stocks: number;
  rebalance_frequency: string;
  holding_period_days: number;
  latest_result_summary?: LatestResultSummary | null;
}

interface FrontierPoint {
  volatility: number;
  expected_return: number;
  sharpe: number;
}

export interface OptimizationResponse {
  current_portfolio_value: number;
  optimized_portfolio_value?: number;
  optimized_allocations: OptimizedAllocation[];
  added_symbols?: string[];
  removed_symbols?: string[];
  rebalance_frequency: string;
  holding_period_days: number;
  summary_metrics: {
    current_expected_return: number;
    optimized_expected_return: number;
    current_sharpe: number;
    optimized_sharpe: number;
    optimized_sortino: number;
    current_volatility: number;
    optimized_volatility: number;
    optimized_max_drawdown: number;
    optimized_cvar_95: number;
    optimized_information_ratio: number;
    optimized_tracking_error: number;
  };
  constraint_summary: {
    turnover: number;
    no_trade_band: number;
    max_stock_weight: number;
    max_sector_weight: number;
    estimated_transaction_cost_naira: number;
    liquidity_screened_candidates: number;
  };
  compliance_report: {
    overall_status: 'pass' | 'review' | 'breach' | 'warn';
    items: {
      rule: string;
      status: 'pass' | 'review' | 'breach' | 'warn';
      observed: number | string;
      limit: number | string;
      message: string;
    }[];
  };
  fund_manager_report: {
    title: string;
    market: string;
    mandate_label: string;
    objective: string;
    benchmark: string;
    generated_at: string;
    recommendation: string;
    summary: {
      added_symbols: string[];
      removed_symbols: string[];
      current_expected_return: number;
      optimized_expected_return: number;
    };
  };
  prediction_engine: {
    models: string[];
    symbols_scored: number;
    buy_count: number;
    sell_count: number;
    conflict_count: number;
    average_confidence: number;
    qualified_model_coverage: number;
    scope: string;
  };
  efficient_frontier: {
    points: { volatility: number; expected_return: number }[];
    optimized: FrontierPoint;
    current: FrontierPoint;
    benchmark: FrontierPoint;
  };
  diversification_score: {
    score: number;
    message: string;
    effective_positions: number;
    active_positions: number;
    effective_sectors: number;
    sector_count: number;
    largest_weight: number;
    largest_sector_weight: number;
  };
  risk_contributions: { symbol: string; risk_contribution_pct: number; weight: number }[];
  correlation_matrix: { symbols: string[]; values: number[][] };
  mandate_summary: {
    label: string;
    benchmark: string;
    max_stock_weight: number;
    max_sector_weight: number;
    min_liquidity_score: number;
    objective: string;
  };
  sector_allocations: { sector: string; current_weight: number; optimized_weight: number }[];
  backtest_summary: {
    window_days: number;
    strategies: Record<string, { cumulative_return: number; sharpe: number }>;
  };
  benchmark_metrics: {
    expected_return: number;
    volatility: number;
    sharpe: number;
    max_drawdown: number;
  };
}

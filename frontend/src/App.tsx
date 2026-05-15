// App.tsx
import { useEffect, useMemo, useState } from 'react';
import type {
  FundManager,
  Holding,
  MandateProfile,
  OptimizationResponse,
  OptimizedAllocation,
  RiskProfile,
  SavedPortfolio,
  SignalSummary,
  SignalWatchlist,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';
type RebalanceFrequency = 'weekly' | 'monthly' | 'quarterly';
type NavTab = 'overview' | 'watchlist' | 'input' | 'dashboard' | 'workspace' | 'data';
type ConsumerPortfolioStatus = 'existing' | 'new';
type BootstrapResponse = {
  symbols?: string[];
  prices?: Record<string, number>;
  price_updated_at?: string | null;
  signal_summary?: SignalSummary | null;
  watchlist?: SignalWatchlist | null;
  managers?: FundManager[];
  signal_error?: string;
};

const riskDescriptions: Record<RiskProfile, string> = {
  conservative: 'Tighter caps, lower turnover, steadier allocations.',
  balanced: 'Middle ground between signal strength and diversification.',
  aggressive: 'Looser caps, stronger tilt toward high-conviction names.',
};

const mandateDescriptions: Record<MandateProfile, string> = {
  balanced_equity: 'Diversified NGX equity construction for a balanced fund mandate.',
  growth_equity: 'Higher-conviction equity tilt with more room for signal-led additions.',
  income_equity: 'Defensive equity sleeve with tighter concentration and liquidity rules.',
  pension_equity: 'Strict equity sleeve controls for a pension-style investment mandate.',
};

const mandateLabels: Record<MandateProfile, string> = {
  balanced_equity: 'Balanced Equity',
  growth_equity: 'Growth Equity',
  income_equity: 'Income Equity',
  pension_equity: 'Pension Equity',
};

const stableSingleStockCap: Record<RiskProfile, number> = {
  conservative: 0.08,
  balanced: 0.1,
  aggressive: 0.15,
};

const initialHoldings: Holding[] = [{ symbol: '', quantity: 0 }];

/* ── Formatters ── */
const fmtCcy = (v: number) =>
  new Intl.NumberFormat('en-NG', { style: 'currency', currency: 'NGN', maximumFractionDigits: 0 }).format(v);

const fmtCcySigned = (v: number) => `${v > 0 ? '+' : ''}${fmtCcy(v)}`;

const fmtPct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;

const fmtShares = (v: number) =>
  new Intl.NumberFormat('en-NG', { maximumFractionDigits: 2 }).format(v);

const fmtDateTime = (v: string | null) =>
  v ? new Intl.DateTimeFormat('en-NG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(v)) : 'Unavailable';

const fmtRelative = (v: string | null) => {
  if (!v) return '—';
  const ms = Date.now() - new Date(v).getTime();
  const m = Math.max(0, Math.round(ms / 60000));
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
};

const fmtStrategy = (name: string) =>
  name.split('_').map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');

/* ── Signal helpers ── */
function getSignalBadge(a: OptimizedAllocation) {
  if (a.signal_status === 'CONFLICT') return { label: 'Hold / Review', cls: 'badge-hold' };
  if (a.signal_status === 'SELL')
    return { label: a.consensus_tier === 1 ? 'Strong sell' : 'Sell signal', cls: 'badge-sell' };
  if (a.signal_status === 'BUY') {
    if (a.consensus_tier === 1) return { label: 'Strong buy', cls: 'badge-buy' };
    if (a.consensus_tier === 2) return { label: 'Positive signal', cls: 'badge-blue' };
    return { label: 'Buy signal', cls: 'badge-blue' };
  }
  return { label: 'Historical', cls: 'badge-hist' };
}

function getChangeReason(a: OptimizedAllocation) {
  if (a.action === 'add') return `Added — signal-adjusted Sharpe contribution ranked strongly in the candidate set.`;
  if (a.action === 'increase') return `Increased — optimizer recommends more shares based on return and risk contribution.`;
  if (a.action === 'reduce') return `Reduced — optimizer recommends fewer shares than the current position.`;
  if (a.action === 'exit') return `Exited — no longer improves the optimized risk-return tradeoff.`;
  return `Kept — still fits the diversified target allocation.`;
}

function getModelVoteSummary(a: OptimizedAllocation) {
  if (!a.model_votes.length) return 'Historical return model only';
  return a.model_votes.map((v) => `${v.model}:${v.signal}`).join(' · ');
}

function getAllocVals(
  a: OptimizedAllocation,
  portVal: number,
  prices: Record<string, number>,
  enteredShares: Record<string, number>,
) {
  const price = a.latest_price ?? prices[a.symbol] ?? 0;
  const curVal = a.current_weight * portVal;
  const optVal = a.optimized_weight * portVal;
  const curShares = enteredShares[a.symbol] ?? (price > 0 ? curVal / price : 0);
  const optShares = price > 0 ? optVal / price : 0;
  const delta = optShares - curShares;
  const tradeVal = optVal - curVal;
  return { price, curVal, optVal, curShares, optShares, delta, tradeVal };
}

function getTrade(delta: number) {
  if (delta > 0.01) return `Buy ${fmtShares(delta)} sh`;
  if (delta < -0.01) return `Sell ${fmtShares(Math.abs(delta))} sh`;
  return 'No trade';
}

function getComplianceBadge(status: 'pass' | 'review' | 'breach' | 'warn') {
  if (status === 'pass') return { label: 'Pass', cls: 'badge-buy' };
  if (status === 'review' || status === 'warn') return { label: 'Review', cls: 'badge-hold' };
  return { label: 'Breach', cls: 'badge-sell' };
}

function fmtComplianceValue(value: number | string, rule: string) {
  if (typeof value !== 'number') return value;
  if (rule.toLowerCase().includes('new-stock')) return value.toString();
  return fmtPct(value);
}

function csvEscape(v: string | number | null | undefined) {
  const t = typeof v === 'number' ? v.toString() : (v ?? '');
  return `"${String(t).replace(/"/g, '""')}"`;
}

/* ── Clock ── */
function Clock() {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString('en-NG', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
  useEffect(() => {
    const id = setInterval(() => setTime(new Date().toLocaleTimeString('en-NG', { hour: '2-digit', minute: '2-digit', second: '2-digit' })), 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="topbar-clock">{time} WAT</span>;
}

/* ── SVG icons ── */
const Icon = {
  grid: () => (
    <svg viewBox="0 0 14 14" fill="currentColor"><rect x="1" y="1" width="5" height="5" rx="1"/><rect x="8" y="1" width="5" height="5" rx="1"/><rect x="1" y="8" width="5" height="5" rx="1"/><rect x="8" y="8" width="5" height="5" rx="1"/></svg>
  ),
  signal: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M1 10.5c1.5-2 3-3 6-3s4.5 1 6 3"/><path d="M3.5 8C5 6.5 6 6 7 6s2 .5 3.5 2"/><circle cx="7" cy="12" r="1" fill="currentColor" stroke="none"/></svg>
  ),
  sliders: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"><path d="M2 3.5h10M2 7h10M2 10.5h10"/><circle cx="5" cy="3.5" r="1.5" fill="var(--bg-panel)" strokeWidth="1.4"/><circle cx="9" cy="7" r="1.5" fill="var(--bg-panel)" strokeWidth="1.4"/><circle cx="5" cy="10.5" r="1.5" fill="var(--bg-panel)" strokeWidth="1.4"/></svg>
  ),
  chart: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"><polyline points="1,10 4,6 7,8 10,3 13,5"/></svg>
  ),
  db: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"><ellipse cx="7" cy="4" rx="5" ry="2"/><path d="M2 4v3c0 1.1 2.24 2 5 2s5-.9 5-2V4"/><path d="M2 7v3c0 1.1 2.24 2 5 2s5-.9 5-2V7"/></svg>
  ),
  user: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"><circle cx="7" cy="4" r="2.2"/><path d="M2.5 12c.7-2.3 2.2-3.4 4.5-3.4s3.8 1.1 4.5 3.4"/></svg>
  ),
  chevron: () => (
    <svg viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M3 4l2 2 2-2"/></svg>
  ),
};

/* ══════════════════════════════════════════════════════════════ */
function App() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [holdings, setHoldings] = useState<Holding[]>(initialHoldings);
  const [consumerPortfolioStatus, setConsumerPortfolioStatus] = useState<ConsumerPortfolioStatus>('existing');
  const [initialCashNaira, setInitialCashNaira] = useState(5_000_000);
  const [riskProfile, setRiskProfile] = useState<RiskProfile>('balanced');
  const [mandateProfile, setMandateProfile] = useState<MandateProfile>('balanced_equity');
  const [allowNewStocks, setAllowNewStocks] = useState(true);
  const [maxNewStocks, setMaxNewStocks] = useState(3);
  const [rebalanceFrequency, setRebalanceFrequency] = useState<RebalanceFrequency>('monthly');
  const [holdingPeriodDays, setHoldingPeriodDays] = useState(20);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [result, setResult] = useState<OptimizationResponse | null>(null);
  const [watchlist, setWatchlist] = useState<SignalWatchlist | null>(null);
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [signalSummary, setSignalSummary] = useState<SignalSummary | null>(null);
  const [priceUpdatedAt, setPriceUpdatedAt] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [managers, setManagers] = useState<FundManager[]>([]);
  const [selectedManagerId, setSelectedManagerId] = useState('');
  const [savedPortfolios, setSavedPortfolios] = useState<SavedPortfolio[]>([]);
  const [managerName, setManagerName] = useState('');
  const [managerFirm, setManagerFirm] = useState('');
  const [managerEmail, setManagerEmail] = useState('');
  const [portfolioName, setPortfolioName] = useState('Equity Portfolio');
  const [workspaceStatus, setWorkspaceStatus] = useState<string | null>(null);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  const sortedSymbols = useMemo(() => [...symbols].sort((a, b) => a.localeCompare(b)), [symbols]);
  const selectedManager = useMemo(
    () => managers.find((manager) => manager.id === selectedManagerId) ?? null,
    [managers, selectedManagerId],
  );
  const consumerHasPortfolio = consumerPortfolioStatus === 'existing';

  useEffect(() => {
    (async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/bootstrap`, { cache: 'no-store' });
        const data: BootstrapResponse & { detail?: string } = await response.json();
        if (!response.ok) throw new Error(data.detail ?? 'Failed to load data.');

        setSymbols(data.symbols ?? []);
        setPrices(data.prices ?? {});
        setPriceUpdatedAt(data.price_updated_at ?? null);
        setSignalSummary(data.signal_summary ?? null);
        setWatchlist(data.watchlist ?? null);
        if (data.signal_error) setError(data.signal_error);

        const loadedManagers = data.managers ?? [];
        setManagers(loadedManagers);
        setSelectedManagerId((current) => current || loadedManagers[0]?.id || '');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load data.');
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedManagerId) {
      setSavedPortfolios([]);
      return;
    }
    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/portfolios`, { cache: 'no-store' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail ?? 'Could not load saved portfolios.');
        setSavedPortfolios(data.portfolios ?? []);
      } catch (e) {
        setWorkspaceError(e instanceof Error ? e.message : 'Could not load saved portfolios.');
      }
    })();
  }, [selectedManagerId]);

  const totalBudget = useMemo(
    () => consumerHasPortfolio
      ? holdings.reduce((s, h) => s + h.quantity * (prices[h.symbol.trim().toUpperCase()] ?? 0), 0)
      : initialCashNaira,
    [consumerHasPortfolio, holdings, initialCashNaira, prices],
  );

  const enteredShares = useMemo(
    () => holdings.reduce((m, h) => {
      const sym = h.symbol.trim().toUpperCase();
      if (sym) m[sym] = (m[sym] ?? 0) + h.quantity;
      return m;
    }, {} as Record<string, number>),
    [holdings],
  );

  const actionCounts = useMemo(() => {
    if (!result) return { add: 0, increase: 0, reduce: 0, keep: 0, exit: 0 };
    return result.optimized_allocations.reduce(
      (c, a) => { c[a.action]++; return c; },
      { add: 0, increase: 0, reduce: 0, keep: 0, exit: 0 } as Record<OptimizedAllocation['action'], number>,
    );
  }, [result]);

  const topAdditions = useMemo(
    () => result?.optimized_allocations.filter((a) => a.action === 'add').sort((a, b) => b.signal_score - a.signal_score).slice(0, 3) ?? [],
    [result],
  );

  const backtestRows = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.backtest_summary.strategies).map(([name, strategy]) => ({ name, label: fmtStrategy(name), strategy }));
  }, [result]);

  const maxBTReturn = useMemo(
    () => Math.max(...backtestRows.map((r) => Math.abs(r.strategy.cumulative_return)), 0.01),
    [backtestRows],
  );

  const updateHolding = (i: number, field: keyof Holding, val: string) => {
    setHoldings((c) => c.map((h, idx) => idx === i ? { ...h, [field]: field === 'quantity' ? Number(val) : val } : h));
  };

  const validateForm = () => {
    const msgs: string[] = [];
    if (!consumerHasPortfolio) {
      if (initialCashNaira <= 0) msgs.push('Initial cash amount must be greater than zero.');
      setValidationErrors(msgs);
      return msgs.length === 0;
    }
    const syms = holdings.map((h) => h.symbol.trim().toUpperCase()).filter(Boolean);
    if (holdings.some((h) => !h.symbol.trim())) msgs.push('Each row needs a symbol selected.');
    if (holdings.some((h) => h.quantity <= 0)) msgs.push('Quantity must be greater than zero.');
    if (holdings.some((h) => h.symbol.trim() && !prices[h.symbol.trim().toUpperCase()])) msgs.push('Price unavailable for one or more symbols.');
    if (new Set(syms).size !== syms.length) msgs.push('Duplicate symbols detected — they will be merged.');
    setValidationErrors(msgs);
    return !msgs.some((m) => m !== 'Duplicate symbols detected — they will be merged.');
  };

  const submitPortfolio = async () => {
    setError(null); setStatusMessage(null);
    if (!validateForm()) { setResult(null); return; }
    setIsSubmitting(true); setStatusMessage(consumerHasPortfolio ? 'Running optimizer…' : 'Constructing portfolio…');
    try {
      const endpoint = consumerHasPortfolio ? 'optimize-portfolio' : 'construct-portfolio';
      const payload = consumerHasPortfolio
        ? {
            holdings: buildHoldingPayload(),
            risk_profile: riskProfile,
            mandate_profile: mandateProfile,
            allow_new_stocks: allowNewStocks,
            max_new_stocks: maxNewStocks,
            rebalance_frequency: rebalanceFrequency,
            holding_period_days: holdingPeriodDays,
          }
        : {
            initial_cash_naira: initialCashNaira,
            risk_profile: riskProfile,
            mandate_profile: mandateProfile,
            max_stocks: Math.max(1, maxNewStocks),
            rebalance_frequency: rebalanceFrequency,
            holding_period_days: holdingPeriodDays,
          };
      const res = await fetch(`${API_BASE_URL}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? (consumerHasPortfolio ? 'Optimization failed.' : 'Portfolio construction failed.'));
      setResult(data); setActiveTab('dashboard'); setStatusMessage(consumerHasPortfolio ? 'Optimization complete.' : 'Portfolio constructed.');
    } catch (e) {
      setError(e instanceof Error ? e.message : (consumerHasPortfolio ? 'Optimization failed.' : 'Portfolio construction failed.'));
      setStatusMessage(null); setResult(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  const buildHoldingPayload = () =>
    consumerHasPortfolio ? holdings.map((h) => {
      const s = h.symbol.trim().toUpperCase();
      return { symbol: s, amount_naira: h.quantity * (prices[s] ?? 0) };
    }) : [];

  const refreshPortfolios = async (managerId = selectedManagerId) => {
    if (!managerId) return;
    const res = await fetch(`${API_BASE_URL}/fund-managers/${managerId}/portfolios`, { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail ?? 'Could not refresh saved portfolios.');
    setSavedPortfolios(data.portfolios ?? []);
  };

  const createManagerAccount = async () => {
    setWorkspaceError(null); setWorkspaceStatus(null);
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: managerName, firm: managerFirm, email: managerEmail }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not create fund manager account.');
      const manager: FundManager = data.manager;
      setManagers((current) => [manager, ...current]);
      setSelectedManagerId(manager.id);
      setManagerName(''); setManagerFirm(''); setManagerEmail('');
      setWorkspaceStatus('Fund manager workspace created.');
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not create fund manager account.');
    }
  };

  const saveCurrentPortfolio = async () => {
    setWorkspaceError(null); setWorkspaceStatus(null);
    if (!selectedManagerId) {
      setWorkspaceError('Create or select a fund manager workspace first.');
      return;
    }
    if (!validateForm()) {
      setWorkspaceError('Fix portfolio input errors before saving.');
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/portfolios`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: portfolioName,
          holdings: buildHoldingPayload(),
          risk_profile: riskProfile,
          mandate_profile: mandateProfile,
          allow_new_stocks: consumerHasPortfolio ? allowNewStocks : true,
          max_new_stocks: Math.max(1, maxNewStocks),
          rebalance_frequency: rebalanceFrequency,
          holding_period_days: holdingPeriodDays,
          consumer_has_portfolio: consumerHasPortfolio,
          initial_cash_naira: consumerHasPortfolio ? null : initialCashNaira,
          latest_result: result,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not save portfolio.');
      await refreshPortfolios(selectedManagerId);
      setWorkspaceStatus(`Saved ${data.portfolio.name}.`);
      setActiveTab('workspace');
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not save portfolio.');
    }
  };

  const loadSavedPortfolio = (portfolio: SavedPortfolio) => {
    const hasPortfolio = portfolio.consumer_has_portfolio ?? portfolio.holdings.length > 0;
    const loadedHoldings = portfolio.holdings.map((holding) => {
      const symbol = holding.symbol.trim().toUpperCase();
      const price = prices[symbol] ?? 0;
      return {
        symbol,
        quantity: price > 0 ? holding.amount_naira / price : 0,
      };
    });
    setConsumerPortfolioStatus(hasPortfolio ? 'existing' : 'new');
    setInitialCashNaira(portfolio.initial_cash_naira ?? 5_000_000);
    setHoldings(loadedHoldings.length ? loadedHoldings : initialHoldings);
    setRiskProfile(portfolio.risk_profile);
    setMandateProfile(portfolio.mandate_profile);
    setAllowNewStocks(portfolio.allow_new_stocks);
    setMaxNewStocks(portfolio.max_new_stocks);
    setRebalanceFrequency(portfolio.rebalance_frequency);
    setHoldingPeriodDays(portfolio.holding_period_days);
    setPortfolioName(portfolio.name);
    setActiveTab('input');
    setWorkspaceStatus(`Loaded ${portfolio.name}.`);
  };

  const optimizeSavedPortfolio = async (portfolio: SavedPortfolio) => {
    setWorkspaceError(null); setWorkspaceStatus(`Running optimizer for ${portfolio.name}…`);
    try {
      const res = await fetch(`${API_BASE_URL}/portfolios/${portfolio.id}/optimize`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not optimize saved portfolio.');
      setResult(data.result);
      await refreshPortfolios(portfolio.manager_id);
      setWorkspaceStatus(`Optimization recorded for ${portfolio.name}.`);
      setActiveTab('dashboard');
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not optimize saved portfolio.');
      setWorkspaceStatus(null);
    }
  };

  const exportResults = () => {
    if (!result) return;
    const rows = [
      ['Fund Manager Report', result.fund_manager_report.title],
      ['Market', result.fund_manager_report.market],
      ['Mandate', result.fund_manager_report.mandate_label],
      ['Objective', result.fund_manager_report.objective],
      ['Benchmark', result.fund_manager_report.benchmark],
      ['Generated At', result.fund_manager_report.generated_at],
      ['Recommendation', result.fund_manager_report.recommendation],
      ['Compliance Status', result.compliance_report.overall_status],
      [],
      ['Prediction Engine'],
      ['Models', result.prediction_engine.models.join(' + ')],
      ['Symbols Scored', result.prediction_engine.symbols_scored],
      ['Buy Signals', result.prediction_engine.buy_count],
      ['Sell Signals', result.prediction_engine.sell_count],
      ['Avg Confidence', result.prediction_engine.average_confidence],
      [],
      ['Compliance Checks'],
      ['Rule', 'Status', 'Observed', 'Limit', 'Message'],
      ...result.compliance_report.items.map((item) => [item.rule, item.status, item.observed, item.limit, item.message]),
      [],
      ['Optimized Allocation'],
      ['Symbol', 'Sector', 'Signal', 'Model Votes', 'Tier', 'Confidence', 'Exp Return', 'Cur Shares', 'Cur Value', 'Opt Shares', 'Opt Value', 'Trade', 'Trade Value', 'Action', 'Reason'],
      ...result.optimized_allocations.map((a) => {
        const v = getAllocVals(a, result.current_portfolio_value, prices, enteredShares);
        return [a.symbol, a.sector, a.signal_status, getModelVoteSummary(a), a.consensus_tier?.toString() ?? '', a.avg_confidence?.toString() ?? '', a.expected_return.toString(), v.curShares.toString(), v.curVal.toString(), v.optShares.toString(), v.optVal.toString(), getTrade(v.delta), v.tradeVal.toString(), a.action, getChangeReason(a)];
      }),
    ];
    const csv = rows.map((r) => r.map(csvEscape).join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
    const a = document.createElement('a');
    a.href = url; a.setAttribute('download', 'nse_optimized_portfolio.csv'); a.click();
    URL.revokeObjectURL(url);
  };

  /* ── Nav config ── */
  const navItems: { id: NavTab; label: string; icon: () => JSX.Element; badge?: string }[] = [
    { id: 'overview',   label: 'Overview',   icon: Icon.grid },
    { id: 'watchlist',  label: 'Watchlist',  icon: Icon.signal, badge: signalSummary ? `${signalSummary.buy_count}` : undefined },
    { id: 'input',      label: 'Input',      icon: Icon.sliders },
    { id: 'dashboard',  label: 'Dashboard',  icon: Icon.chart,  badge: result ? 'RDY' : undefined },
    { id: 'workspace',  label: 'Workspace',  icon: Icon.user,   badge: savedPortfolios.length ? `${savedPortfolios.length}` : undefined },
    { id: 'data',       label: 'Data',       icon: Icon.db },
  ];

  const activeLabel = navItems.find((n) => n.id === activeTab)?.label ?? '';

  /* ────────────────────────────────────────────────────────────── */
  return (
    <div className="app-shell">

      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg viewBox="0 0 14 14" fill="none" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="1,10 4,5 7,7.5 10,2 13,4"/>
              <line x1="1" y1="13" x2="13" y2="13"/>
            </svg>
          </div>
          <div className="brand-text">
            <strong>NSE Optimizer</strong>
            <span>Portfolio Intelligence</span>
          </div>
        </div>

        <span className="sidebar-section-label">Navigation</span>

        {navItems.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
            onClick={() => setActiveTab(item.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && setActiveTab(item.id)}
          >
            <item.icon />
            {item.label}
            {item.badge && (
              <span className={`nav-badge ${item.id === 'dashboard' ? 'green' : ''}`}>{item.badge}</span>
            )}
          </div>
        ))}

        <div className="sidebar-divider" />

        <div className="sidebar-footer">
          <div className="sidebar-stat">
            <span>Signal universe</span>
            <strong>{(signalSummary?.row_count ?? symbols.length) || '—'}</strong>
          </div>
          <div className="sidebar-stat">
            <span>Avg confidence</span>
            <strong>{signalSummary ? fmtPct(signalSummary.avg_confidence, 0) : '—'}</strong>
          </div>
          <div className="sidebar-stat">
            <span>Portfolio value</span>
            <strong>{totalBudget > 0 ? fmtCcy(totalBudget) : '—'}</strong>
          </div>
        </div>
      </aside>

      {/* ── TOPBAR ── */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="topbar-breadcrumb">
            <span>NSE OPTIMIZER</span>
            <svg width="8" height="8" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.2"><path d="M2 1l3 3-3 3"/></svg>
            <strong>{activeLabel.toUpperCase()}</strong>
          </div>
        </div>
        <div className="topbar-right">
          <span className="status-dot">SYSTEM ONLINE</span>
          <Clock />
        </div>
      </header>

      {/* ── MAIN ── */}
      <main className="main">

        {/* ══ OVERVIEW ══ */}
        {activeTab === 'overview' && (
          <>
            {/* KPI strip */}
            <div className="metric-strip">
              <div className="metric-card">
                <div className="metric-label">Portfolio Value</div>
                <div className="metric-value">{totalBudget > 0 ? fmtCcy(totalBudget) : '—'}</div>
                <div className="metric-sub">{holdings.length} holding{holdings.length === 1 ? '' : 's'} entered</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Signal Universe</div>
                <div className="metric-value accent">{(signalSummary?.row_count ?? symbols.length) || '—'}</div>
                <div className="metric-sub">{signalSummary ? `${signalSummary.buy_count} buy · ${signalSummary.sell_count} sell` : 'Loading…'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Avg Confidence</div>
                <div className="metric-value">{signalSummary ? fmtPct(signalSummary.avg_confidence, 0) : '—'}</div>
                <div className="metric-sub">{fmtRelative(signalSummary?.generated_at ?? null)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Optimization</div>
                <div className={`metric-value ${result ? 'up' : ''}`}>{result ? 'READY' : 'PENDING'}</div>
                <div className="metric-sub">{result ? `${result.optimized_allocations.length} recommendations` : 'Use Input tab'}</div>
              </div>
            </div>

            {/* How it works */}
            <div className="panel">
              <div className="panel-head">
                <div className="panel-title">System Architecture</div>
              </div>
              <div style={{ padding: '1rem' }}>
                <div className="steps-grid">
                  {[
                    { n: '01', t: 'User Holdings', b: 'Enter NSE symbols and share quantities. Prices are fetched live from the market data feed.' },
                    { n: '02', t: 'ML Signal Engine', b: 'XGBoost, Random Forest, and LSTM models output ranked buy/sell signals with confidence scores.' },
                    { n: '03', t: 'Mandate Optimizer', b: 'Combines expected return, covariance, mandate profile, liquidity rules, and sector diversification limits.' },
                    { n: '04', t: 'Manager Report', b: 'Outputs buy/sell decisions, Nigerian equity compliance checks, and an exportable fund manager report.' },
                  ].map((s) => (
                    <div className="step-item" key={s.n}>
                      <div className="step-num">Step {s.n}</div>
                      <div className="step-title">{s.t}</div>
                      <div className="step-body">{s.b}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {/* ══ WATCHLIST ══ */}
        {activeTab === 'watchlist' && (
          <>
            <div className="metric-strip">
              <div className="metric-card">
                <div className="metric-label">Buy Signals</div>
                <div className="metric-value up">{signalSummary?.buy_count ?? '—'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Sell Signals</div>
                <div className="metric-value down">{signalSummary?.sell_count ?? '—'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Avg Confidence</div>
                <div className="metric-value accent">{signalSummary ? fmtPct(signalSummary.avg_confidence, 0) : '—'}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Signal Updated</div>
                <div className="metric-value" style={{ fontSize: '0.95rem' }}>{fmtRelative(signalSummary?.generated_at ?? null)}</div>
              </div>
            </div>

            {watchlist ? (
              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Strong Buy Signals</div>
                    <span className="badge badge-buy">{watchlist.top_buys.length} names</span>
                  </div>
                  <div>
                    {watchlist.top_buys.map((item) => (
                      <div className="watch-item" key={`buy-${item.symbol}`}>
                        <div>
                          <div className="watch-symbol">{item.symbol}</div>
                          <div className="watch-reason">{item.reason}</div>
                        </div>
                        <span className="badge badge-buy">{fmtPct(item.avg_confidence, 0)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Weak / Sell Signals</div>
                    <span className="badge badge-sell">{watchlist.top_sells.length} names</span>
                  </div>
                  <div>
                    {watchlist.top_sells.map((item) => (
                      <div className="watch-item" key={`sell-${item.symbol}`}>
                        <div>
                          <div className="watch-symbol">{item.symbol}</div>
                          <div className="watch-reason">{item.reason}</div>
                        </div>
                        <span className="badge badge-sell">{fmtPct(item.avg_confidence, 0)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="panel">
                <div className="empty-state">
                  <div className="empty-icon">
                    <Icon.signal />
                  </div>
                  <div className="empty-state-title">Watchlist unavailable</div>
                  <div className="empty-state-sub">The signal store has not returned data yet. Check your API connection.</div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ══ INPUT ══ */}
        {activeTab === 'input' && (
          <div className="grid-form">
            {/* Left — holdings */}
            <div className="panel">
              <div className="panel-head">
                <div className="panel-title">{consumerHasPortfolio ? 'Current Holdings' : 'New Portfolio Cash'}</div>
                {consumerHasPortfolio && (
                  <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={() => setHoldings((c) => [...c, { symbol: '', quantity: 0 }])}>
                    + Add Row
                  </button>
                )}
              </div>
              <div className="panel-body-sm">
                <div className="form-field" style={{ marginBottom: '0.75rem' }}>
                  <div className="field-label">Does the consumer currently have a portfolio?</div>
                  <div className="risk-row">
                    <button className={`risk-pill ${consumerPortfolioStatus === 'existing' ? 'active' : ''}`} onClick={() => setConsumerPortfolioStatus('existing')}>Has portfolio</button>
                    <button className={`risk-pill ${consumerPortfolioStatus === 'new' ? 'active' : ''}`} onClick={() => setConsumerPortfolioStatus('new')}>No portfolio</button>
                  </div>
                </div>

                {consumerHasPortfolio ? (
                  <div className="holdings-list">
                    {holdings.map((h, i) => {
                      const sym = h.symbol.trim().toUpperCase();
                      const price = prices[sym] ?? 0;
                      const val = h.quantity * price;
                      return (
                        <div className="holding-row" key={`h-${i}`}>
                          <div className="form-field">
                            <div className="field-label">Symbol</div>
                            <input
                              className="field-input"
                              type="text"
                              list={`dl-${i}`}
                              placeholder="e.g. DANGCEM"
                              value={h.symbol}
                              onChange={(e) => updateHolding(i, 'symbol', e.target.value)}
                            />
                            <datalist id={`dl-${i}`}>
                              {sortedSymbols.map((s) => <option key={s} value={s} />)}
                            </datalist>
                          </div>
                          <div className="form-field">
                            <div className="field-label">Shares</div>
                            <input
                              className="field-input"
                              type="number"
                              min="0"
                              step="1"
                              placeholder="0"
                              value={h.quantity === 0 ? '' : h.quantity}
                              onChange={(e) => updateHolding(i, 'quantity', e.target.value)}
                            />
                          </div>
                          <div className="field-readonly">
                            <div className="field-label">Price</div>
                            <div className={`field-readonly-val ${price ? '' : 'dim'}`}>{price ? fmtCcy(price) : '—'}</div>
                          </div>
                          <div className="field-readonly">
                            <div className="field-label">Value</div>
                            <div className="field-readonly-val">{fmtCcy(val)}</div>
                          </div>
                          <button
                            className="btn btn-danger"
                            style={{ padding: '0.4rem 0.6rem', alignSelf: 'flex-end' }}
                            onClick={() => setHoldings((c) => c.filter((_, idx) => idx !== i))}
                            disabled={holdings.length === 1}
                          >
                            ✕
                          </button>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="form-field">
                    <div className="field-label">Initial cash available</div>
                    <input
                      className="field-input"
                      type="number"
                      min="1"
                      step="10000"
                      value={initialCashNaira === 0 ? '' : initialCashNaira}
                      onChange={(e) => setInitialCashNaira(Number(e.target.value) || 0)}
                    />
                  </div>
                )}

                <div style={{ marginTop: '0.75rem', padding: '0.75rem', borderTop: '1px solid var(--border)' }}>
                  <div className="summary-row"><span>{consumerHasPortfolio ? 'Total portfolio value' : 'Construction cash'}</span><strong>{fmtCcy(totalBudget)}</strong></div>
                  <div className="summary-row"><span>{consumerHasPortfolio ? 'Holdings entered' : 'Starting holdings'}</span><strong>{consumerHasPortfolio ? holdings.length : 'None'}</strong></div>
                </div>
              </div>
            </div>

            {/* Right — preferences */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Risk Profile</div>
                </div>
                <div className="panel-body">
                  <div className="risk-row">
                    {(['conservative', 'balanced', 'aggressive'] as RiskProfile[]).map((p) => (
                      <button key={p} className={`risk-pill ${p === riskProfile ? 'active' : ''}`} onClick={() => setRiskProfile(p)}>{p}</button>
                    ))}
                  </div>
                  <div style={{ marginTop: '0.6rem', fontSize: '11px', color: 'var(--text-3)', lineHeight: 1.5 }}>
                    {riskDescriptions[riskProfile]}
                  </div>
                  <div style={{ marginTop: '0.75rem' }}>
                    <div className="summary-row"><span>Single-stock cap</span><strong>{fmtPct(stableSingleStockCap[riskProfile], 0)}</strong></div>
                  </div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Equity Mandate</div>
                </div>
                <div className="panel-body">
                  <div className="mandate-grid">
                    {(['balanced_equity', 'growth_equity', 'income_equity', 'pension_equity'] as MandateProfile[]).map((p) => (
                      <button
                        key={p}
                        className={`mandate-tile ${p === mandateProfile ? 'active' : ''}`}
                        onClick={() => setMandateProfile(p)}
                      >
                        <strong>{mandateLabels[p]}</strong>
                        <span>{mandateDescriptions[p]}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Optimizer Settings</div>
                </div>
                <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <label className="toggle-row">
                    <input type="checkbox" checked={consumerHasPortfolio ? allowNewStocks : true} onChange={(e) => setAllowNewStocks(e.target.checked)} disabled={!consumerHasPortfolio} />
                    <span>{consumerHasPortfolio ? 'Allow optimizer to introduce new signal names' : 'Build from signal-approved names'}</span>
                  </label>

                  <div className="form-field">
                    <div className="field-label">{consumerHasPortfolio ? 'Max new stocks' : 'Target stocks'} — {maxNewStocks}</div>
                    <input type="range" min={consumerHasPortfolio ? '0' : '1'} max="10" value={maxNewStocks} onChange={(e) => setMaxNewStocks(Number(e.target.value))} disabled={consumerHasPortfolio && !allowNewStocks} style={{ width: '100%', accentColor: 'var(--accent)' }} />
                  </div>

                  <div className="form-field">
                    <div className="field-label">Rebalance frequency</div>
                    <select className="field-input" value={rebalanceFrequency} onChange={(e) => setRebalanceFrequency(e.target.value as RebalanceFrequency)}>
                      <option value="weekly">Weekly</option>
                      <option value="monthly">Monthly</option>
                      <option value="quarterly">Quarterly</option>
                    </select>
                  </div>

                  <div className="form-field">
                    <div className="field-label">Holding period (days)</div>
                    <input className="field-input" type="number" min="1" max="252" value={holdingPeriodDays} onChange={(e) => setHoldingPeriodDays(Number(e.target.value) || 20)} />
                  </div>

                  <button className="btn btn-primary btn-full" onClick={submitPortfolio} disabled={isSubmitting}>
                    {isSubmitting ? (consumerHasPortfolio ? 'OPTIMIZING…' : 'CONSTRUCTING…') : (consumerHasPortfolio ? '▶  RUN OPTIMIZER' : '▶  CONSTRUCT PORTFOLIO')}
                  </button>
                  <button className="btn btn-ghost btn-full" onClick={saveCurrentPortfolio} disabled={isSubmitting}>
                    Save to Workspace
                  </button>

                  {validationErrors.length > 0 && (
                    <div className="banner banner-warn">
                      {validationErrors.map((m) => <p key={m}>{m}</p>)}
                    </div>
                  )}
                  {statusMessage && <div className="banner banner-ok">{statusMessage}</div>}
                  {error && <div className="banner banner-error">{error}</div>}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ DASHBOARD ══ */}
        {activeTab === 'dashboard' && (
          result ? (
            <>
              {/* KPI strip */}
              <div className="metric-strip" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
                {[
                  { l: 'Cur Return',  v: fmtPct(result.summary_metrics.current_expected_return),   cls: '' },
                  { l: 'Opt Return',  v: fmtPct(result.summary_metrics.optimized_expected_return), cls: 'up' },
                  { l: 'Cur Sharpe', v: result.summary_metrics.current_sharpe.toFixed(3),           cls: '' },
                  { l: 'Opt Sharpe', v: result.summary_metrics.optimized_sharpe.toFixed(3),         cls: 'up' },
                  { l: 'Sortino',    v: result.summary_metrics.optimized_sortino.toFixed(3),         cls: 'accent' },
                  { l: 'Cur Vol',    v: fmtPct(result.summary_metrics.current_volatility),           cls: '' },
                  { l: 'Opt Vol',    v: fmtPct(result.summary_metrics.optimized_volatility),         cls: 'up' },
                  { l: 'Max DD',     v: fmtPct(result.summary_metrics.optimized_max_drawdown),       cls: 'down' },
                  { l: 'CVaR 95%',   v: fmtPct(result.summary_metrics.optimized_cvar_95),            cls: 'down' },
                  { l: 'Info Ratio', v: result.summary_metrics.optimized_information_ratio.toFixed(3), cls: 'accent' },
                ].map((m) => (
                  <div className="metric-card" key={m.l}>
                    <div className="metric-label">{m.l}</div>
                    <div className={`metric-value ${m.cls}`} style={{ fontSize: '1.1rem' }}>{m.v}</div>
                  </div>
                ))}
              </div>

              {/* Action counts */}
              <div className="grid-5">
                {[
                  { l: 'Adds',      v: actionCounts.add,      cls: 'up' },
                  { l: 'Increases', v: actionCounts.increase,  cls: 'up' },
                  { l: 'Reduces',   v: actionCounts.reduce,    cls: 'down' },
                  { l: 'Keeps',     v: actionCounts.keep,      cls: 'accent' },
                  { l: 'Exits',     v: actionCounts.exit,      cls: 'down' },
                ].map((a) => (
                  <div className="panel" key={a.l}>
                    <div style={{ padding: '0.75rem', textAlign: 'center' }}>
                      <div className="metric-label">{a.l}</div>
                      <div className={`metric-value ${a.v > 0 ? a.cls : ''}`} style={{ fontSize: '2rem' }}>{a.v}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Mandate + ML engine */}
              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Equity Fund Mandate</div>
                    <span className={`badge ${getComplianceBadge(result.compliance_report.overall_status).cls}`}>
                      {getComplianceBadge(result.compliance_report.overall_status).label}
                    </span>
                  </div>
                  <div className="panel-body-sm">
                    <div className="stat-pair"><span>Mandate</span><strong>{result.mandate_summary.label}</strong></div>
                    <div className="stat-pair"><span>Benchmark</span><strong>{result.mandate_summary.benchmark}</strong></div>
                    <div className="stat-pair"><span>Max stock</span><strong>{fmtPct(result.mandate_summary.max_stock_weight)}</strong></div>
                    <div className="stat-pair"><span>Max sector</span><strong>{fmtPct(result.mandate_summary.max_sector_weight)}</strong></div>
                    <div className="stat-pair"><span>Min liquidity score</span><strong>{fmtPct(result.mandate_summary.min_liquidity_score, 0)}</strong></div>
                    <div className="mandate-objective">{result.mandate_summary.objective}</div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">ML Prediction Engine</div>
                    <span className="badge badge-blue">{result.prediction_engine.models.join(' + ')}</span>
                  </div>
                  <div className="panel-body-sm">
                    <div className="stat-pair"><span>Scope</span><strong>{result.prediction_engine.scope}</strong></div>
                    <div className="stat-pair"><span>Symbols scored</span><strong>{result.prediction_engine.symbols_scored}</strong></div>
                    <div className="stat-pair"><span>Signals</span><strong>{result.prediction_engine.buy_count} buy · {result.prediction_engine.sell_count} sell</strong></div>
                    <div className="stat-pair"><span>Conflicts</span><strong>{result.prediction_engine.conflict_count}</strong></div>
                    <div className="stat-pair"><span>Avg confidence</span><strong>{fmtPct(result.prediction_engine.average_confidence, 0)}</strong></div>
                    <div className="stat-pair"><span>Qualified coverage</span><strong>{fmtPct(result.prediction_engine.qualified_model_coverage, 0)}</strong></div>
                  </div>
                </div>
              </div>

              {/* Mandate + Benchmark */}
              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Mandate Controls</div>
                    <span className="badge badge-blue">{result.rebalance_frequency}</span>
                  </div>
                  <div className="panel-body-sm">
                    {[
                      { l: 'Turnover',            v: fmtPct(result.constraint_summary.turnover) },
                      { l: 'No-trade band',        v: fmtPct(result.constraint_summary.no_trade_band) },
                      { l: 'Max stock weight',     v: fmtPct(result.constraint_summary.max_stock_weight) },
                      { l: 'Max sector weight',    v: fmtPct(result.constraint_summary.max_sector_weight) },
                      { l: 'Transaction cost est', v: fmtCcy(result.constraint_summary.estimated_transaction_cost_naira) },
                      { l: 'Liquidity candidates', v: String(result.constraint_summary.liquidity_screened_candidates) },
                      { l: 'Holding horizon',      v: `${result.holding_period_days} days` },
                    ].map((r) => <div className="stat-pair" key={r.l}><span>{r.l}</span><strong>{r.v}</strong></div>)}
                  </div>
                </div>
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Benchmark Comparison</div>
                  </div>
                  <div className="panel-body-sm">
                    {[
                      { l: 'Benchmark return',     v: fmtPct(result.benchmark_metrics.expected_return) },
                      { l: 'Benchmark volatility', v: fmtPct(result.benchmark_metrics.volatility) },
                      { l: 'Benchmark Sharpe',     v: result.benchmark_metrics.sharpe.toFixed(3) },
                      { l: 'Benchmark max DD',     v: fmtPct(result.benchmark_metrics.max_drawdown) },
                      { l: 'Tracking error',       v: fmtPct(result.summary_metrics.optimized_tracking_error) },
                    ].map((r) => <div className="stat-pair" key={r.l}><span>{r.l}</span><strong>{r.v}</strong></div>)}
                  </div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Nigerian Equity Compliance Checks</div>
                  <span className={`badge ${getComplianceBadge(result.compliance_report.overall_status).cls}`}>
                    {result.compliance_report.overall_status.toUpperCase()}
                  </span>
                </div>
                <div>
                  {result.compliance_report.items.map((item) => {
                    const badge = getComplianceBadge(item.status);
                    return (
                      <div className="compliance-row" key={item.rule}>
                        <span className={`badge ${badge.cls}`}>{badge.label}</span>
                        <div>
                          <strong>{item.rule}</strong>
                          <p>{item.message}</p>
                        </div>
                        <span className="compliance-val">
                          {fmtComplianceValue(item.observed, item.rule)}
                          {' / '}
                          {fmtComplianceValue(item.limit, item.rule)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Fund Manager Report</div>
                  <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={exportResults}>
                    Export Report CSV
                  </button>
                </div>
                <div className="panel-body-sm">
                  <div className="report-line">
                    <strong>{result.fund_manager_report.recommendation}</strong>
                    <span>{result.fund_manager_report.market}</span>
                  </div>
                  <div className="report-grid">
                    <div><span>Added</span><strong>{result.fund_manager_report.summary.added_symbols.join(', ') || 'None'}</strong></div>
                    <div><span>Removed</span><strong>{result.fund_manager_report.summary.removed_symbols.join(', ') || 'None'}</strong></div>
                    <div><span>Current return</span><strong>{fmtPct(result.fund_manager_report.summary.current_expected_return)}</strong></div>
                    <div><span>Optimized return</span><strong>{fmtPct(result.fund_manager_report.summary.optimized_expected_return)}</strong></div>
                  </div>
                </div>
              </div>

              {/* Allocation table */}
              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Optimized Allocations</div>
                  <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={exportResults}>
                    Export CSV
                  </button>
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Sector</th>
                        <th>Signal</th>
                        <th>Current</th>
                        <th>Optimized</th>
                        <th>Trade</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.optimized_allocations.map((a) => {
                        const badge = getSignalBadge(a);
                        const v = getAllocVals(a, result.current_portfolio_value, prices, enteredShares);
                        return (
                          <tr key={a.symbol}>
                            <td><span className="tbl-symbol">{a.symbol}</span></td>
                            <td><span className="tbl-muted">{a.sector}</span></td>
                            <td>
                              <span className={`badge ${badge.cls}`}>
                                {badge.label}{a.avg_confidence !== null ? ` · ${(a.avg_confidence * 100).toFixed(0)}%` : ''}
                              </span>
                              <div className="tbl-muted">{getModelVoteSummary(a)}</div>
                            </td>
                            <td>
                              <div>{fmtShares(v.curShares)} sh</div>
                              <div className="tbl-muted">{fmtCcy(v.curVal)}</div>
                            </td>
                            <td>
                              <div>{fmtShares(v.optShares)} sh</div>
                              <div className="tbl-muted">{fmtCcy(v.optVal)}</div>
                            </td>
                            <td>
                              <div className={v.tradeVal >= 0 ? 'up' : 'down'}>{getTrade(v.delta)}</div>
                              <div className="tbl-muted">{fmtCcySigned(v.tradeVal)}</div>
                            </td>
                            <td><span className={`action-tag action-${a.action}`}>{a.action}</span></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Sector + Backtest */}
              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Sector Exposure</div>
                  </div>
                  {result.sector_allocations.map((a) => (
                    <div className="sector-row" key={`sec-${a.sector}`}>
                      <strong>{a.sector}</strong>
                      <span>Before {fmtPct(a.current_weight)}</span>
                      <span className="after">After {fmtPct(a.optimized_weight)}</span>
                    </div>
                  ))}
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Backtest Snapshot</div>
                    <span className="badge badge-hist">{result.backtest_summary.window_days}d window</span>
                  </div>
                  {backtestRows.map(({ name, label, strategy }) => {
                    const w = Math.max(4, Math.min(100, (Math.abs(strategy.cumulative_return) / maxBTReturn) * 100));
                    return (
                      <div className="bt-row" key={`bt-${name}`}>
                        <div className="bt-header">
                          <strong>{label}</strong>
                          <span>{fmtPct(strategy.cumulative_return)} · Sharpe {strategy.sharpe.toFixed(3)}</span>
                        </div>
                        <div className="bt-track">
                          <div className={strategy.cumulative_return >= 0 ? 'bt-fill-pos' : 'bt-fill-neg'} style={{ width: `${w}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Value comparison */}
              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Before vs After — Position Comparison</div>
                </div>
                <div>
                  {result.optimized_allocations.slice(0, 10).map((a) => {
                    const v = getAllocVals(a, result.current_portfolio_value, prices, enteredShares);
                    return (
                      <div className="compare-item" key={`cmp-${a.symbol}`}>
                        <div className="compare-header">
                          <span className="tbl-symbol">{a.symbol}</span>
                          <span className={`action-tag action-${a.action}`}>{a.action}</span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                          <div>
                            <div className="bar-label">Before</div>
                            <div className="bar-track" style={{ margin: '4px 0' }}>
                              <div className="bar-fill-before" style={{ width: `${a.current_weight * 100}%` }} />
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--text-3)' }}>{fmtShares(v.curShares)} sh · {fmtCcy(v.curVal)}</div>
                          </div>
                          <div>
                            <div className="bar-label">After</div>
                            <div className="bar-track" style={{ margin: '4px 0' }}>
                              <div className="bar-fill-after" style={{ width: `${a.optimized_weight * 100}%` }} />
                            </div>
                            <div style={{ fontSize: '11px' }}>
                              <span style={{ color: 'var(--text-2)' }}>{fmtShares(v.optShares)} sh · {fmtCcy(v.optVal)}</span>
                              {' '}
                              <span className={v.tradeVal >= 0 ? 'up' : 'down'}>({getTrade(v.delta)})</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Top additions + explanations */}
              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Top Additions</div>
                  </div>
                  {topAdditions.length > 0 ? topAdditions.map((a) => {
                    const badge = getSignalBadge(a);
                    const v = getAllocVals(a, result.current_portfolio_value, prices, enteredShares);
                    return (
                      <div key={`add-${a.symbol}`} style={{ padding: '0.7rem 0.75rem', borderBottom: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                          <span className="tbl-symbol">{a.symbol}</span>
                          <span className={`badge ${badge.cls}`}>{badge.label}</span>
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-3)', lineHeight: 1.5 }}>
                          Buy {fmtShares(v.optShares)} sh · {fmtCcy(v.optVal)} · exp. return {fmtPct(a.expected_return)}
                        </div>
                        <div style={{ fontSize: '10.5px', color: 'var(--slate-500)', marginTop: '0.2rem' }}>
                          {a.sector} · Liquidity {(a.liquidity_score * 100).toFixed(0)}%
                        </div>
                      </div>
                    );
                  }) : (
                    <div style={{ padding: '1rem', fontSize: '11.5px', color: 'var(--text-3)' }}>No new stocks introduced in this run.</div>
                  )}
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Change Explanations</div>
                  </div>
                  {result.optimized_allocations.filter((a) => a.action !== 'keep').slice(0, 5).map((a) => {
                    const v = getAllocVals(a, result.current_portfolio_value, prices, enteredShares);
                    return (
                      <div className="explain-item" key={`why-${a.symbol}`}>
                        <div className="explain-header">
                          <div className="explain-dot" style={{ background: v.tradeVal >= 0 ? 'var(--green-400)' : 'var(--red-400)' }} />
                          <strong>{a.symbol}</strong>
                          <span className={`action-tag action-${a.action}`} style={{ marginLeft: 'auto' }}>{a.action}</span>
                        </div>
                        <div className="explain-body">{getChangeReason(a)}</div>
                        <div className="explain-meta">
                          {a.sector} · 20d vol {fmtCcy(a.avg_trade_value_20d)} · R² {a.avg_r2 !== null ? a.avg_r2.toFixed(2) : 'n/a'}
                        </div>
                        <div className={`explain-trade ${v.tradeVal >= 0 ? 'up' : 'down'}`}>
                          {getTrade(v.delta)} ({fmtCcySigned(v.tradeVal)})
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="panel">
              <div className="empty-state">
                <div className="empty-icon">
                  <Icon.chart />
                </div>
                <div className="empty-state-title">No optimization results</div>
                <div className="empty-state-sub">
                  Enter your current NSE holdings in the Input tab and click "Run Optimizer" to populate this dashboard.
                </div>
                <button className="btn btn-primary" style={{ marginTop: '1rem' }} onClick={() => setActiveTab('input')}>
                  Go to Input
                </button>
              </div>
            </div>
          )
        )}

        {/* ══ WORKSPACE ══ */}
        {activeTab === 'workspace' && (
          <>
            <div className="grid-2">
              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Fund Manager Workspace</div>
                  {selectedManager && <span className="badge badge-blue">{selectedManager.firm}</span>}
                </div>
                <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {managers.length > 0 && (
                    <div className="form-field">
                      <div className="field-label">Active manager</div>
                      <select className="field-input" value={selectedManagerId} onChange={(e) => setSelectedManagerId(e.target.value)}>
                        {managers.map((manager) => (
                          <option key={manager.id} value={manager.id}>{manager.name} — {manager.firm}</option>
                        ))}
                      </select>
                    </div>
                  )}
                  <div className="workspace-manager-card">
                    <strong>{selectedManager ? selectedManager.name : 'No workspace selected'}</strong>
                    <span>{selectedManager ? `${selectedManager.email || 'No contact email'} · Consumer ${consumerHasPortfolio ? 'has an existing portfolio' : 'needs a first portfolio'}` : 'Create a fund manager workspace to save multiple portfolios.'}</span>
                  </div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div className="panel-title">Create Manager</div>
                </div>
                <div className="panel-body" style={{ display: 'grid', gap: '0.75rem' }}>
                  <div className="form-field">
                    <div className="field-label">Manager name</div>
                    <input className="field-input" value={managerName} onChange={(e) => setManagerName(e.target.value)} placeholder="e.g. Amina Bello" />
                  </div>
                  <div className="form-field">
                    <div className="field-label">Firm</div>
                    <input className="field-input" value={managerFirm} onChange={(e) => setManagerFirm(e.target.value)} placeholder="e.g. Lagos Asset Management" />
                  </div>
                  <div className="form-field">
                    <div className="field-label">Email</div>
                    <input className="field-input" value={managerEmail} onChange={(e) => setManagerEmail(e.target.value)} placeholder="optional" />
                  </div>
                  <button className="btn btn-primary" onClick={createManagerAccount}>Create Workspace</button>
                </div>
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <div className="panel-title">{consumerHasPortfolio ? 'Save Current Equity Portfolio' : 'Save New Consumer Portfolio'}</div>
                <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={saveCurrentPortfolio}>
                  {consumerHasPortfolio ? 'Save Current' : 'Save New'}
                </button>
              </div>
              <div className="panel-body">
                <div className="grid-3">
                  <div className="form-field">
                    <div className="field-label">Portfolio name</div>
                    <input className="field-input" value={portfolioName} onChange={(e) => setPortfolioName(e.target.value)} />
                  </div>
                  <div className="form-field">
                    <div className="field-label">Consumer status</div>
                    <select className="field-input" value={consumerPortfolioStatus} onChange={(e) => setConsumerPortfolioStatus(e.target.value as ConsumerPortfolioStatus)}>
                      <option value="existing">Currently has portfolio</option>
                      <option value="new">No portfolio yet</option>
                    </select>
                  </div>
                  <div className="field-readonly">
                    <div className="field-label">{consumerHasPortfolio ? 'Current value' : 'Initial cash'}</div>
                    <div className="field-readonly-val">{fmtCcy(totalBudget)}</div>
                  </div>
                  {!consumerHasPortfolio && (
                    <div className="form-field">
                      <div className="field-label">Initial cash amount</div>
                      <input className="field-input" type="number" min="1" step="10000" value={initialCashNaira === 0 ? '' : initialCashNaira} onChange={(e) => setInitialCashNaira(Number(e.target.value) || 0)} />
                    </div>
                  )}
                  <div className="field-readonly">
                    <div className="field-label">Latest run</div>
                    <div className="field-readonly-val">{result ? result.compliance_report.overall_status.toUpperCase() : 'Not run'}</div>
                  </div>
                </div>
                {workspaceStatus && <div className="banner banner-ok">{workspaceStatus}</div>}
                {workspaceError && <div className="banner banner-error">{workspaceError}</div>}
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <div className="panel-title">Saved Portfolios</div>
                <span className="badge badge-blue">{savedPortfolios.length} portfolio{savedPortfolios.length === 1 ? '' : 's'}</span>
              </div>
              {savedPortfolios.length > 0 ? (
                <div className="workspace-list">
                  {savedPortfolios.map((portfolio) => (
                    <div className="workspace-portfolio" key={portfolio.id}>
                      <div>
                        <div className="watch-symbol">{portfolio.name}</div>
                        <div className="watch-reason">
                          {mandateLabels[portfolio.mandate_profile]} · {portfolio.risk_profile} · {(portfolio.consumer_has_portfolio ?? portfolio.holdings.length > 0) ? `${portfolio.holdings.length} holding${portfolio.holdings.length === 1 ? '' : 's'}` : `new portfolio from ${fmtCcy(portfolio.initial_cash_naira ?? 0)}`}
                        </div>
                        {portfolio.latest_result_summary && (
                          <div className="workspace-run">
                            Last run {fmtRelative(portfolio.latest_result_summary.generated_at)} · {portfolio.latest_result_summary.compliance_status} · Sharpe {portfolio.latest_result_summary.optimized_sharpe.toFixed(3)}
                          </div>
                        )}
                      </div>
                      <div className="workspace-actions">
                        <button className="btn btn-ghost" onClick={() => loadSavedPortfolio(portfolio)}>Load</button>
                        <button className="btn btn-primary" onClick={() => optimizeSavedPortfolio(portfolio)}>Optimize</button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon"><Icon.user /></div>
                  <div className="empty-state-title">No saved portfolios</div>
                  <div className="empty-state-sub">Save the current holdings to track optimization history for this fund manager.</div>
                </div>
              )}
            </div>
          </>
        )}

        {/* ══ DATA ══ */}
        {activeTab === 'data' && (
          <div className="panel">
            <div className="panel-head">
              <div className="panel-title">Data Freshness</div>
            </div>
            <div className="panel-body">
              <div className="fresh-grid">
                <div className="fresh-item">
                  <div className="metric-label">Signals generated</div>
                  <div className="fresh-val">{fmtDateTime(signalSummary?.generated_at ?? null)}</div>
                  <div className="fresh-badge">{fmtRelative(signalSummary?.generated_at ?? null)}</div>
                </div>
                <div className="fresh-item">
                  <div className="metric-label">Prices updated</div>
                  <div className="fresh-val">{fmtDateTime(priceUpdatedAt)}</div>
                  <div className="fresh-badge">{fmtRelative(priceUpdatedAt)}</div>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;

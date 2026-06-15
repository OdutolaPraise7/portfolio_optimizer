// App.tsx
import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  FundManager,
  Holding,
  ManagedConsumer,
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
type NavTab = 'overview' | 'watchlist' | 'input' | 'dashboard' | 'workspace' | 'clients';
type ConsumerPortfolioStatus = 'existing' | 'new';
type BootstrapResponse = {
  symbols?: string[];
  prices?: Record<string, number>;
  price_updated_at?: string | null;
  signal_summary?: SignalSummary | null;
  watchlist?: SignalWatchlist | null;
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

const mandateSingleStockCap: Partial<Record<MandateProfile, number>> = {
  growth_equity: 0.15,
  income_equity: 0.08,
  pension_equity: 0.07,
};

const effectiveSingleStockCap = (risk: RiskProfile, mandate: MandateProfile) =>
  Math.min(stableSingleStockCap[risk], mandateSingleStockCap[mandate] ?? stableSingleStockCap[risk]);

const minimumStocksForCap = (cap: number) => Math.ceil((1 - 1e-12) / cap);

const minimumStocksForSetup = (risk: RiskProfile, mandate: MandateProfile) =>
  minimumStocksForCap(effectiveSingleStockCap(risk, mandate));

const preferredStockTarget = (minimum: number) => Math.min(20, Math.max(minimum, minimum + 3));

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

const fmtDate = (v: string | null) =>
  v ? new Intl.DateTimeFormat('en-NG', { day: 'numeric', month: 'long', year: 'numeric' }).format(new Date(v)) : 'Unavailable';

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
  currentPortVal: number,
  optimizedPortVal: number,
  prices: Record<string, number>,
  enteredShares: Record<string, number>,
) {
  const price = a.latest_price ?? prices[a.symbol] ?? 0;
  const curVal = a.current_weight * currentPortVal;
  const optVal = a.optimized_weight * optimizedPortVal;
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

function getOptimizedPortfolioValue(result: OptimizationResponse) {
  if (typeof result.optimized_portfolio_value === 'number' && Number.isFinite(result.optimized_portfolio_value)) {
    return result.optimized_portfolio_value;
  }
  const grossValue = result.optimized_allocations.reduce(
    (sum, a) => sum + a.optimized_weight * result.current_portfolio_value,
    0,
  );
  const baseValue = grossValue > 0 ? grossValue : result.current_portfolio_value;
  return Math.max(baseValue - result.constraint_summary.estimated_transaction_cost_naira, 0);
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

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

function corrCellBg(value: number) {
  const alpha = 0.12 + clamp(Math.abs(value), 0, 1) * 0.5;
  return value >= 0 ? `rgba(74, 222, 128, ${alpha})` : `rgba(248, 113, 113, ${alpha})`;
}

function csvEscape(v: string | number | null | undefined) {
  const t = typeof v === 'number' ? v.toString() : (v ?? '');
  return `"${String(t).replace(/"/g, '""')}"`;
}

function friendlyErrorMessage(message: string, fallback: string) {
  if (!message) return fallback;
  if (message.includes('Mandate infeasible')) {
    return 'This mandate needs a broader eligible stock set. I adjusted the stock target where possible; try allowing more new stocks or choosing a less restrictive mandate.';
  }
  return message;
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
  user: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"><circle cx="7" cy="4" r="2.2"/><path d="M2.5 12c.7-2.3 2.2-3.4 4.5-3.4s3.8 1.1 4.5 3.4"/></svg>
  ),
  users: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"><circle cx="5" cy="4" r="2"/><path d="M1.5 12c.6-2 2-3 3.5-3s2.9 1 3.5 3"/><circle cx="10" cy="4.5" r="1.5"/><path d="M12.5 12c-.5-1.8-1.5-2.6-2.5-2.6"/></svg>
  ),
  chevron: () => (
    <svg viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M3 4l2 2 2-2"/></svg>
  ),
  chevronRight: () => (
    <svg viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M4 3l2 2-2 2"/></svg>
  ),
  history: () => (
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"><path d="M1.5 7a5.5 5.5 0 1 0 1-3.2"/><polyline points="1.5,2 1.5,5 4.5,5"/><path d="M7 4.5V7l1.8 1.8"/></svg>
  ),
};

/* ══════════════════════════════════════════════════════════════ */
function App() {
  const mainRef = useRef<HTMLElement | null>(null);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [holdings, setHoldings] = useState<Holding[]>(initialHoldings);
  const [consumerPortfolioStatus, setConsumerPortfolioStatus] = useState<ConsumerPortfolioStatus>('existing');
  const [initialCashNaira, setInitialCashNaira] = useState(0);
  const [riskProfile, setRiskProfile] = useState<RiskProfile>('balanced');
  const [mandateProfile, setMandateProfile] = useState<MandateProfile>('balanced_equity');
  const [allowNewStocks, setAllowNewStocks] = useState(true);
  const [maxNewStocks, setMaxNewStocks] = useState(10);
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
  const [consumers, setConsumers] = useState<ManagedConsumer[]>([]);
  const [selectedConsumerId, setSelectedConsumerId] = useState('');
  const [savedPortfolios, setSavedPortfolios] = useState<SavedPortfolio[]>([]);
  const [managerName, setManagerName] = useState('');
  const [managerFirm, setManagerFirm] = useState('');
  const [managerEmail, setManagerEmail] = useState('');
  const [portfolioName, setPortfolioName] = useState('Equity Portfolio');
  const [consumerName, setConsumerName] = useState('');
  const [consumerEmail, setConsumerEmail] = useState('');
  const [workspaceStatus, setWorkspaceStatus] = useState<string | null>(null);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [managerPassword, setManagerPassword] = useState('');
  const [authView, setAuthView] = useState<'register' | 'login'>('register');

  const [activePortfolioId, setActivePortfolioId] = useState<string | null>(null);
  const [applyingOptimised, setApplyingOptimised] = useState(false);

  // Clients tab state
  const [expandedPortfolioId, setExpandedPortfolioId] = useState<string | null>(null);
  const [portfolioRuns, setPortfolioRuns] = useState<Record<string, any[]>>({});
  const [runsLoading, setRunsLoading] = useState<string | null>(null);
  const [batchSelected, setBatchSelected] = useState<Set<string>>(new Set());
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResult, setBatchResult] = useState<{ succeeded: number; failed: number; results: any[] } | null>(null);

  const sortedSymbols = useMemo(() => [...symbols].sort((a, b) => a.localeCompare(b)), [symbols]);
  const selectedManager = useMemo(
    () => managers.find((manager) => manager.id === selectedManagerId) ?? null,
    [managers, selectedManagerId],
  );
  const consumerHasPortfolio = consumerPortfolioStatus === 'existing';
  const selectedConsumer = useMemo(
    () => consumers.find((consumer) => consumer.id === selectedConsumerId) ?? null,
    [consumers, selectedConsumerId],
  );
  const activeSingleStockCap = useMemo(
    () => effectiveSingleStockCap(riskProfile, mandateProfile),
    [riskProfile, mandateProfile],
  );
  const minimumActiveStocks = useMemo(
    () => minimumStocksForCap(activeSingleStockCap),
    [activeSingleStockCap],
  );
  const currentHoldingCount = useMemo(
    () => new Set(holdings.map((h) => h.symbol.trim().toUpperCase()).filter(Boolean)).size,
    [holdings],
  );
  const minimumNewStocks = useMemo(
    () => Math.max(0, minimumActiveStocks - currentHoldingCount),
    [minimumActiveStocks, currentHoldingCount],
  );
  const minimumSliderStocks = consumerHasPortfolio ? minimumNewStocks : minimumActiveStocks;
  const mustAllowNewStocks = consumerHasPortfolio && minimumNewStocks > 0;
  const effectiveAllowNewStocks = !consumerHasPortfolio || allowNewStocks || mustAllowNewStocks;
  const effectiveStockTarget = Math.min(20, Math.max(maxNewStocks, minimumSliderStocks));

  const chooseRiskProfile = (profile: RiskProfile) => {
    setRiskProfile(profile);
    const requiredActive = minimumStocksForSetup(profile, mandateProfile);
    const requiredNew = consumerHasPortfolio ? Math.max(0, requiredActive - currentHoldingCount) : requiredActive;
    setMaxNewStocks(preferredStockTarget(requiredNew));
  };

  const chooseMandateProfile = (profile: MandateProfile) => {
    setMandateProfile(profile);
    const requiredActive = minimumStocksForSetup(riskProfile, profile);
    const requiredNew = consumerHasPortfolio ? Math.max(0, requiredActive - currentHoldingCount) : requiredActive;
    setMaxNewStocks(preferredStockTarget(requiredNew));
  };

  const updateStockTarget = (value: number) => {
    setMaxNewStocks(Math.max(minimumSliderStocks, value));
  };

  useEffect(() => {
    setMaxNewStocks((current) => Math.max(current, preferredStockTarget(minimumSliderStocks)));
  }, [minimumSliderStocks]);

  useEffect(() => {
    if (mustAllowNewStocks) setAllowNewStocks(true);
  }, [mustAllowNewStocks]);

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
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load data.');
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedManagerId) {
      setSavedPortfolios([]);
      setConsumers([]);
      setSelectedConsumerId('');
      return;
    }
    (async () => {
      try {
        const [portfolioRes, consumerRes] = await Promise.all([
          fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/portfolios`, { cache: 'no-store' }),
          fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/consumers`, { cache: 'no-store' }),
        ]);
        const portfolioData = await portfolioRes.json();
        const consumerData = await consumerRes.json();
        if (!portfolioRes.ok) throw new Error(portfolioData.detail ?? 'Could not load saved portfolios.');
        if (!consumerRes.ok) throw new Error(consumerData.detail ?? 'Could not load managed consumers.');
        const loadedConsumers: ManagedConsumer[] = consumerData.consumers ?? [];
        setSavedPortfolios(portfolioData.portfolios ?? []);
        setConsumers(loadedConsumers);
        setSelectedConsumerId((current) => (
          loadedConsumers.some((consumer) => consumer.id === current) ? current : ''
        ));
      } catch (e) {
        setWorkspaceError(e instanceof Error ? e.message : 'Could not load workspace data.');
      }
    })();
  }, [selectedManagerId]);

  const totalBudget = useMemo(
    () => consumerHasPortfolio
      ? holdings.reduce((s, h) => s + h.quantity * (prices[h.symbol.trim().toUpperCase()] ?? 0), 0)
      : initialCashNaira,
    [consumerHasPortfolio, holdings, initialCashNaira, prices],
  );

  const currentPortfolioValue = result?.current_portfolio_value ?? totalBudget;
  const optimizedPortfolioValue = result ? getOptimizedPortfolioValue(result) : totalBudget;
  const displayedPortfolioValue = result ? optimizedPortfolioValue : totalBudget;
  const valueDelta = optimizedPortfolioValue - currentPortfolioValue;

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
            allow_new_stocks: effectiveAllowNewStocks,
            max_new_stocks: effectiveStockTarget,
            rebalance_frequency: rebalanceFrequency,
            holding_period_days: holdingPeriodDays,
          }
        : {
            initial_cash_naira: initialCashNaira,
            risk_profile: riskProfile,
            mandate_profile: mandateProfile,
            max_stocks: Math.max(1, effectiveStockTarget),
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
      const fallback = consumerHasPortfolio ? 'Optimization failed.' : 'Portfolio construction failed.';
      setError(friendlyErrorMessage(e instanceof Error ? e.message : '', fallback));
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

  const refreshConsumers = async (managerId = selectedManagerId) => {
    if (!managerId) return;
    const res = await fetch(`${API_BASE_URL}/fund-managers/${managerId}/consumers`, { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail ?? 'Could not refresh managed consumers.');
    setConsumers(data.consumers ?? []);
  };

  const loginManager = async () => {
    setWorkspaceError(null); setWorkspaceStatus(null);
    if (!loginEmail.trim()) { setWorkspaceError('Enter your email address.'); return; }
    if (!loginPassword) { setWorkspaceError('Enter your password.'); return; }
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: loginEmail.trim().toLowerCase(), password: loginPassword }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Login failed.');
      const manager: FundManager = data.manager;
      setManagers((current) => current.some((m) => m.id === manager.id) ? current : [manager, ...current]);
      setSelectedManagerId(manager.id);
      setLoginEmail('');
      setLoginPassword('');
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Login failed.');
    }
  };

  const createManagerAccount = async () => {
    setWorkspaceError(null); setWorkspaceStatus(null);
    if (managerPassword.length < 8) {
      setWorkspaceError('Password must be at least 8 characters.');
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: managerName, firm: managerFirm, email: managerEmail, password: managerPassword }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not create account.');
      const manager: FundManager = data.manager;
      setManagers((current) => [manager, ...current]);
      setSelectedManagerId(manager.id);
      setManagerName(''); setManagerFirm(''); setManagerEmail(''); setManagerPassword('');
      setWorkspaceStatus('Account created. Welcome!');
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not create account.');
    }
  };

  const selectWorkspace = (managerId: string) => {
    setSelectedManagerId(managerId);
    setSelectedConsumerId('');
    setConsumerName('');
    setConsumerEmail('');
    setWorkspaceError(null);
    setWorkspaceStatus(managerId ? 'Workspace activated.' : null);
  };

  const selectConsumer = (consumerId: string) => {
    setSelectedConsumerId(consumerId);
    const consumer = consumers.find((item) => item.id === consumerId);
    if (consumer) {
      setConsumerName(consumer.name);
      setConsumerEmail(consumer.email ?? '');
      setConsumerPortfolioStatus(consumer.consumer_has_portfolio ? 'existing' : 'new');
      setWorkspaceStatus('Consumer selected.');
    } else {
      setConsumerName('');
      setConsumerEmail('');
      setWorkspaceStatus(null);
    }
    setWorkspaceError(null);
  };

  const registerConsumer = async (): Promise<ManagedConsumer | null> => {
    setWorkspaceError(null); setWorkspaceStatus(null);
    if (!selectedManagerId) {
      setWorkspaceError('Select a manager workspace before adding a consumer.');
      return null;
    }
    if (!consumerName.trim()) {
      setWorkspaceError('Enter the consumer name before adding.');
      return null;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/consumers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: consumerName,
          email: consumerEmail,
          consumer_has_portfolio: consumerHasPortfolio,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not add consumer.');
      const consumer: ManagedConsumer = data.consumer;
      setConsumers((current) => [consumer, ...current]);
      setSelectedConsumerId(consumer.id);
      setConsumerName(consumer.name);
      setConsumerEmail(consumer.email ?? '');
      setConsumerPortfolioStatus(consumer.consumer_has_portfolio ? 'existing' : 'new');
      setWorkspaceStatus(`Added ${consumer.name}.`);
      return consumer;
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not add consumer.');
      return null;
    }
  };

  const startNewConsumer = () => {
    setSelectedConsumerId('');
    setConsumerName('');
    setConsumerEmail('');
    setConsumerPortfolioStatus('existing');
    setWorkspaceStatus('Ready for a new consumer.');
    setWorkspaceError(null);
  };

  const exitConsumer = () => {
    setSelectedConsumerId('');
    setConsumerName('');
    setConsumerEmail('');
    setWorkspaceStatus('Exited active consumer.');
    setWorkspaceError(null);
  };

  const deleteActiveConsumer = async () => {
    if (!selectedManagerId) {
      setWorkspaceError('Select a manager workspace before deleting a consumer.');
      return;
    }
    if (!selectedConsumer) {
      setWorkspaceError('Select a consumer before deleting.');
      return;
    }
    const confirmed = window.confirm(
      `Delete only ${selectedConsumer.name}'s consumer profile and saved portfolios? Other consumers will remain.`,
    );
    if (!confirmed) return;

    setWorkspaceError(null); setWorkspaceStatus(null);
    try {
      const res = await fetch(
        `${API_BASE_URL}/fund-managers/${selectedManagerId}/consumers/${selectedConsumer.id}`,
        { method: 'DELETE' },
      );
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not delete consumer.');
      setSelectedConsumerId('');
      setConsumerName('');
      setConsumerEmail('');
      await refreshConsumers(selectedManagerId);
      await refreshPortfolios(selectedManagerId);
      setWorkspaceStatus(`Deleted ${data.consumer?.name ?? selectedConsumer.name}.`);
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not delete consumer.');
    }
  };

  const exitWorkspace = () => {
    setSelectedManagerId('');
    setSelectedConsumerId('');
    setSavedPortfolios([]);
    setConsumerName('');
    setConsumerEmail('');
    setLoginEmail('');
    setLoginPassword('');
    setAuthView('register');
    setWorkspaceError(null);
    setWorkspaceStatus(null);
  };

  const deleteActiveWorkspace = async () => {
    if (!selectedManager) {
      setWorkspaceError('Select a workspace before deleting.');
      return;
    }
    const confirmed = window.confirm(
      `Delete only ${selectedManager.name}'s selected workspace and its saved portfolios? Other workspaces will remain.`,
    );
    if (!confirmed) return;

    setWorkspaceError(null); setWorkspaceStatus(null);
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers/${selectedManager.id}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not delete workspace.');
      setManagers((current) => current.filter((manager) => manager.id !== selectedManager.id));
      setSelectedManagerId('');
      setSelectedConsumerId('');
      setSavedPortfolios([]);
      setConsumers([]);
      setConsumerName('');
      setConsumerEmail('');
      setWorkspaceStatus(`Deleted ${data.manager?.name ?? selectedManager.name}'s workspace.`);
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not delete workspace.');
    }
  };

  const saveCurrentPortfolio = async () => {
    setWorkspaceError(null); setWorkspaceStatus(null);
    if (!selectedManagerId) {
      setWorkspaceError('Create or select a fund manager workspace first.');
      return;
    }
    if (!consumerName.trim()) {
      setWorkspaceError('Enter the consumer name before saving.');
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
          consumer_id: selectedConsumerId,
          consumer_name: consumerName,
          consumer_email: consumerEmail,
          holdings: buildHoldingPayload(),
          risk_profile: riskProfile,
          mandate_profile: mandateProfile,
          allow_new_stocks: effectiveAllowNewStocks,
          max_new_stocks: Math.max(1, effectiveStockTarget),
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
      await refreshConsumers(selectedManagerId);
      if (data.portfolio?.consumer_id) setSelectedConsumerId(data.portfolio.consumer_id);
      if (data.portfolio?.id) setActivePortfolioId(data.portfolio.id);
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
    setInitialCashNaira(portfolio.initial_cash_naira ?? 0);
    setHoldings(loadedHoldings.length ? loadedHoldings : initialHoldings);
    setRiskProfile(portfolio.risk_profile);
    setMandateProfile(portfolio.mandate_profile);
    setAllowNewStocks(portfolio.allow_new_stocks);
    setMaxNewStocks(Math.min(20, portfolio.max_new_stocks));
    setRebalanceFrequency(portfolio.rebalance_frequency as RebalanceFrequency);
    setHoldingPeriodDays(portfolio.holding_period_days);
    setPortfolioName(portfolio.name);
    setSelectedConsumerId(portfolio.consumer_id ?? '');
    setConsumerName(portfolio.consumer_name ?? '');
    setConsumerEmail(portfolio.consumer_email ?? '');
    setActivePortfolioId(portfolio.id);
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
      setActivePortfolioId(portfolio.id);
      setConsumerName(portfolio.consumer_name ?? '');
      setConsumerEmail(portfolio.consumer_email ?? '');
      setSelectedConsumerId(portfolio.consumer_id ?? '');
      await refreshPortfolios(portfolio.manager_id);
      setWorkspaceStatus(`Optimization recorded for ${portfolio.name}.`);
      setActiveTab('dashboard');
    } catch (e) {
      setWorkspaceError(friendlyErrorMessage(e instanceof Error ? e.message : '', 'Could not optimize saved portfolio.'));
      setWorkspaceStatus(null);
    }
  };

  const applyOptimisedPortfolio = async () => {
    if (!result || !selectedManagerId) return;
    setApplyingOptimised(true);
    setWorkspaceError(null);
    try {
      let portfolioId = activePortfolioId;

      if (!portfolioId) {
        // No saved portfolio yet — save the current inputs with optimised holdings first
        if (!consumerName.trim()) {
          setWorkspaceError('Enter a consumer name before applying the optimised portfolio.');
          return;
        }
        const optHoldings = result.optimized_allocations
          .filter((a) => a.action !== 'exit')
          .map((a) => ({ symbol: a.symbol, amount_naira: a.optimized_weight * optimizedPortfolioValue }));
        const saveRes = await fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/portfolios`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: portfolioName,
            consumer_id: selectedConsumerId,
            consumer_name: consumerName,
            consumer_email: consumerEmail,
            holdings: optHoldings,
            risk_profile: riskProfile,
            mandate_profile: mandateProfile,
            allow_new_stocks: effectiveAllowNewStocks,
            max_new_stocks: Math.max(1, effectiveStockTarget),
            rebalance_frequency: rebalanceFrequency,
            holding_period_days: holdingPeriodDays,
            consumer_has_portfolio: true,
            initial_cash_naira: null,
            latest_result: result,
          }),
        });
        const saveData = await saveRes.json();
        if (!saveRes.ok) throw new Error(saveData.detail ?? 'Could not save portfolio.');
        portfolioId = saveData.portfolio.id;
        setActivePortfolioId(portfolioId);
        if (saveData.portfolio?.consumer_id) setSelectedConsumerId(saveData.portfolio.consumer_id);
      } else {
        // Update the existing portfolio's holdings to the optimised allocations
        const res = await fetch(`${API_BASE_URL}/portfolios/${portfolioId}/apply-optimised`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ result }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail ?? 'Could not apply optimised portfolio.');
      }

      // Update frontend holdings to the optimised state
      const newHoldings = result.optimized_allocations
        .filter((a) => a.action !== 'exit')
        .map((a) => {
          const price = a.latest_price ?? prices[a.symbol] ?? 0;
          return {
            symbol: a.symbol,
            quantity: price > 0 ? (a.optimized_weight * optimizedPortfolioValue) / price : 0,
          };
        });
      setHoldings(newHoldings.length > 0 ? newHoldings : initialHoldings);
      setConsumerPortfolioStatus('existing');

      await refreshPortfolios(selectedManagerId);
      await refreshConsumers(selectedManagerId);
      setWorkspaceStatus(`${consumerName || 'Client'}'s portfolio updated to optimised allocations.`);
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not apply optimised portfolio.');
    } finally {
      setApplyingOptimised(false);
    }
  };

  const loadPortfolioRuns = async (portfolioId: string) => {
    if (portfolioRuns[portfolioId]) {
      // toggle closed if already open
      setExpandedPortfolioId((current) => current === portfolioId ? null : portfolioId);
      return;
    }
    setRunsLoading(portfolioId);
    try {
      const res = await fetch(`${API_BASE_URL}/portfolios/${portfolioId}`, { cache: 'no-store' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Could not load run history.');
      setPortfolioRuns((current) => ({ ...current, [portfolioId]: data.runs ?? [] }));
      setExpandedPortfolioId(portfolioId);
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Could not load run history.');
    } finally {
      setRunsLoading(null);
    }
  };

  const toggleBatchSelect = (portfolioId: string) => {
    setBatchSelected((current) => {
      const next = new Set(current);
      next.has(portfolioId) ? next.delete(portfolioId) : next.add(portfolioId);
      return next;
    });
  };

  const selectAllPortfolios = () => {
    setBatchSelected(new Set(savedPortfolios.map((p) => p.id)));
  };

  const clearBatchSelection = () => {
    setBatchSelected(new Set());
    setBatchResult(null);
  };

  const runBatchOptimize = async () => {
    if (!selectedManagerId || batchSelected.size === 0) return;
    setBatchRunning(true);
    setBatchResult(null);
    setWorkspaceError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/fund-managers/${selectedManagerId}/batch-optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ portfolio_ids: [...batchSelected] }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Batch optimization failed.');
      setBatchResult({ succeeded: data.succeeded, failed: data.failed, results: data.results });
      // Refresh portfolios so latest_result_summary updates
      await refreshPortfolios(selectedManagerId);
      // Invalidate run caches for re-fetch on next expand
      setBatchSelected(new Set());
      setPortfolioRuns((current) => {
        const next = { ...current };
        data.results.forEach((r: any) => { if (r.success) delete next[r.portfolio_id]; });
        return next;
      });
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : 'Batch optimization failed.');
    } finally {
      setBatchRunning(false);
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
      ['Current Portfolio Value', result.current_portfolio_value],
      ['Net Post-Trade Value', optimizedPortfolioValue],
      ['Estimated Transaction Cost', result.constraint_summary.estimated_transaction_cost_naira],
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
        const v = getAllocVals(a, result.current_portfolio_value, optimizedPortfolioValue, prices, enteredShares);
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
    { id: 'overview',  label: 'Overview',  icon: Icon.grid },
    { id: 'watchlist', label: 'Watchlist', icon: Icon.signal, badge: signalSummary ? `${signalSummary.buy_count}` : undefined },
    { id: 'workspace', label: 'Workspace', icon: Icon.user,   badge: savedPortfolios.length ? `${savedPortfolios.length}` : undefined },
    ...(selectedManagerId ? [
      { id: 'clients'   as NavTab, label: 'Clients',   icon: Icon.users,  badge: consumers.length ? `${consumers.length}` : undefined },
      { id: 'input'     as NavTab, label: 'Input',     icon: Icon.sliders },
      { id: 'dashboard' as NavTab, label: 'Dashboard', icon: Icon.chart,  badge: result ? 'RDY' : undefined },
    ] : []),
  ];

  const activeLabel = navItems.find((n) => n.id === activeTab)?.label ?? '';

  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [activeTab, result]);

  useEffect(() => {
    if (!selectedManagerId && (activeTab === 'clients' || activeTab === 'input' || activeTab === 'dashboard')) {
      setActiveTab('workspace');
    }
  }, [selectedManagerId]);

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
            <span>{result ? 'Net post-trade value' : 'Portfolio value'}</span>
            <strong>{displayedPortfolioValue > 0 ? fmtCcy(displayedPortfolioValue) : '—'}</strong>
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
      <main className="main" ref={mainRef}>

        {/* ══ OVERVIEW ══ */}
        {activeTab === 'overview' && (
          <>
            {/* KPI strip */}
            <div className="metric-strip">
              <div className="metric-card">
                <div className="metric-label">{result ? 'Net Post-Trade Value' : 'Portfolio Value'}</div>
                <div className="metric-value">{displayedPortfolioValue > 0 ? fmtCcy(displayedPortfolioValue) : '—'}</div>
                <div className="metric-sub">
                  {result ? `Before ${fmtCcy(currentPortfolioValue)}` : `${holdings.length} holding${holdings.length === 1 ? '' : 's'} entered`}
                </div>
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
                <div className="panel-title">System Flow</div>
              </div>
              <div style={{ padding: '1rem' }}>
                <div className="steps-grid">
                  {[
                    { n: '01', t: 'Signal Watchlist', b: 'Browse ranked Nigerian equity buy and sell signals from the ML engine. The Watchlist is always accessible without logging in.' },
                    { n: '02', t: 'Manager Login', b: 'Go to the Workspace tab to log in to an existing fund manager account or create a new one to unlock client management tools.' },
                    { n: '03', t: 'Client Setup', b: 'From your manager dashboard, create a new client profile or select a saved client to begin or continue their portfolio workflow.' },
                    { n: '04', t: 'Portfolio Input', b: "Enter the client's current holdings to rebalance, or provide a cash amount to construct a fresh portfolio from the signal universe." },
                    { n: '05', t: 'Mandate & Optimization', b: 'Set risk profile, equity mandate, holding period, and target stocks, then run the ML optimizer to generate trade recommendations.' },
                    { n: '06', t: 'Analytics & Save', b: 'Review trades, compliance checks, efficient frontier, diversification score, and risk attribution, then save the run to the client record.' },
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

            {/* Data freshness */}
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
                    <div className="metric-label">Prices last available</div>
                    <div className="fresh-val">{fmtDate(priceUpdatedAt)}</div>
                    <div className="fresh-badge">{fmtRelative(priceUpdatedAt)}</div>
                  </div>
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
          <>
            <div className="panel">
              <div className="panel-head">
                <div className="panel-title">Active Workspace Identity</div>
                <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={() => setActiveTab('workspace')}>
                  Change Workspace
                </button>
              </div>
              <div className="panel-body">
                <div className="grid-3">
                  <div className="field-readonly">
                    <div className="field-label">Manager</div>
                    <div className="field-readonly-val">{selectedManager ? selectedManager.name : 'No active manager'}</div>
                    <div className="metric-sub">{selectedManager ? selectedManager.firm : 'Select or create a workspace first'}</div>
                  </div>
                  <div className="form-field">
                    <div className="field-label">Managed consumer</div>
                    <select className="field-input" value={selectedConsumerId} onChange={(e) => selectConsumer(e.target.value)} disabled={!selectedManagerId || consumers.length === 0}>
                      <option value="">{consumers.length ? 'Select consumer' : 'No consumers registered'}</option>
                      {consumers.map((consumer) => (
                        <option key={consumer.id} value={consumer.id}>
                          {consumer.name} — {consumer.consumer_has_portfolio ? 'has portfolio' : 'new portfolio'}
                        </option>
                      ))}
                    </select>
                    {selectedConsumer && (
                      <div className="workspace-actions" style={{ marginTop: '0.45rem' }}>
                        <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={exitConsumer}>Exit Consumer</button>
                        <button className="btn btn-danger" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={deleteActiveConsumer}>Delete Consumer</button>
                      </div>
                    )}
                  </div>
                  <div className="form-field">
                    <div className="field-label">Consumer name</div>
                    <input className="field-input" value={consumerName} onChange={(e) => setConsumerName(e.target.value)} placeholder="Consumer or client name" />
                  </div>
                  <div className="form-field">
                    <div className="field-label">Consumer contact</div>
                    <input className="field-input" value={consumerEmail} onChange={(e) => setConsumerEmail(e.target.value)} placeholder="optional email or phone" />
                  </div>
                </div>
              </div>
            </div>

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
                      <button key={p} className={`risk-pill ${p === riskProfile ? 'active' : ''}`} onClick={() => chooseRiskProfile(p)}>{p}</button>
                    ))}
                  </div>
                  <div style={{ marginTop: '0.6rem', fontSize: '11px', color: 'var(--text-3)', lineHeight: 1.5 }}>
                    {riskDescriptions[riskProfile]}
                  </div>
                  <div style={{ marginTop: '0.75rem' }}>
                    <div className="summary-row"><span>Single-stock cap</span><strong>{fmtPct(activeSingleStockCap, 0)}</strong></div>
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
                        onClick={() => chooseMandateProfile(p)}
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
                    <input type="checkbox" checked={effectiveAllowNewStocks} onChange={(e) => setAllowNewStocks(e.target.checked)} disabled={!consumerHasPortfolio || mustAllowNewStocks} />
                    <span>
                      {mustAllowNewStocks
                        ? 'New signal names required to satisfy this mandate'
                        : consumerHasPortfolio ? 'Allow optimizer to introduce new signal names' : 'Build from signal-approved names'}
                    </span>
                  </label>

                  <div className="form-field">
                    <div className="field-label">{consumerHasPortfolio ? 'Max new stocks' : 'Target stocks'} — {effectiveStockTarget}</div>
                    <input type="range" min={String(minimumSliderStocks)} max="20" value={effectiveStockTarget} onChange={(e) => updateStockTarget(Number(e.target.value))} disabled={consumerHasPortfolio && !effectiveAllowNewStocks} style={{ width: '100%', accentColor: 'var(--accent)' }} />
                  </div>
                  <div className="field-readonly">
                    <div className="field-label">{consumerHasPortfolio ? 'Minimum new stocks for this setup' : 'Minimum active stocks for this setup'}</div>
                    <div className="field-readonly-val">{consumerHasPortfolio ? minimumNewStocks : minimumActiveStocks}</div>
                    <div className="metric-sub">
                      {fmtPct(activeSingleStockCap, 0)} max per stock means the portfolio needs at least {minimumActiveStocks} active stocks. {consumerHasPortfolio ? `${currentHoldingCount} current holding${currentHoldingCount === 1 ? '' : 's'} entered, so at least ${minimumNewStocks} new stock${minimumNewStocks === 1 ? '' : 's'} may be needed.` : 'The target starts at that minimum to avoid concentration breaches.'}
                    </div>
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
                    {isSubmitting ? (consumerHasPortfolio ? 'OPTIMIZING…' : 'CONSTRUCTING…') : (consumerHasPortfolio ? '▶  RUN OPTIMIZER' : '▶  CONSTRUCT FIRST PORTFOLIO')}
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
          </>
        )}

        {/* ══ DASHBOARD ══ */}
        {activeTab === 'dashboard' && (
          result ? (
            <>
              {/* Client identity + apply banner */}
              <div className="panel">
                <div className="panel-head" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div>
                    <div className="panel-title" style={{ fontSize: '1rem' }}>
                      {consumerName ? `${consumerName}'s Portfolio` : 'Optimised Portfolio'}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-3)', marginTop: '0.2rem' }}>
                      {result.fund_manager_report.mandate_label} · {result.fund_manager_report.generated_at ? fmtDateTime(result.fund_manager_report.generated_at) : ''}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                    {selectedManagerId && (
                      <button
                        className="btn btn-primary"
                        style={{ fontSize: '10.5px', padding: '0.4rem 1rem' }}
                        onClick={applyOptimisedPortfolio}
                        disabled={applyingOptimised}
                        title={activePortfolioId ? 'Update this client\'s saved portfolio to the optimised allocations' : 'Save and apply the optimised allocations as this client\'s current portfolio'}
                      >
                        {applyingOptimised ? 'Applying…' : `Apply Optimised to ${consumerName || 'Client'}`}
                      </button>
                    )}
                    <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.4rem 0.8rem' }} onClick={exportResults}>
                      Export CSV
                    </button>
                  </div>
                </div>
                {workspaceStatus && <div className="banner banner-ok" style={{ margin: '0 0.75rem 0.75rem' }}>{workspaceStatus}</div>}
                {workspaceError && <div className="banner banner-error" style={{ margin: '0 0.75rem 0.75rem' }}>{workspaceError}</div>}
              </div>

              <div className="metric-strip" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                <div className="metric-card">
                  <div className="metric-label">Current Portfolio Value</div>
                  <div className="metric-value" style={{ fontSize: '1.25rem' }}>{fmtCcy(currentPortfolioValue)}</div>
                  <div className="metric-sub">Value before recommended trades</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Net Post-Trade Value</div>
                  <div className="metric-value up" style={{ fontSize: '1.25rem' }}>{fmtCcy(optimizedPortfolioValue)}</div>
                  <div className="metric-sub">Same capital after estimated transaction costs</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Trading Cost Impact</div>
                  <div className={`metric-value ${valueDelta < 0 ? 'down' : valueDelta > 0 ? 'up' : ''}`} style={{ fontSize: '1.25rem' }}>
                    {fmtCcySigned(valueDelta)}
                  </div>
                  <div className="metric-sub">Cost estimate {fmtCcy(result.constraint_summary.estimated_transaction_cost_naira)}</div>
                </div>
              </div>

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

              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Efficient Frontier</div>
                    <span className="badge badge-blue">Risk vs return</span>
                  </div>
                  <div className="analytics-body">
                    {(() => {
                      const frontier = result.efficient_frontier;
                      const points = frontier.points.length ? frontier.points : [frontier.optimized];
                      const markers = [
                        { label: 'Current', point: frontier.current, cls: 'frontier-current' },
                        { label: 'Optimized', point: frontier.optimized, cls: 'frontier-optimized' },
                        { label: 'Benchmark', point: frontier.benchmark, cls: 'frontier-benchmark' },
                      ];
                      const allPoints = [...points, ...markers.map((m) => m.point)];
                      const minVol = Math.min(...allPoints.map((p) => p.volatility));
                      const maxVol = Math.max(...allPoints.map((p) => p.volatility));
                      const minRet = Math.min(...allPoints.map((p) => p.expected_return));
                      const maxRet = Math.max(...allPoints.map((p) => p.expected_return));
                      const volPad = Math.max((maxVol - minVol) * 0.12, 0.01);
                      const retPad = Math.max((maxRet - minRet) * 0.12, 0.01);
                      const volLo = Math.max(0, minVol - volPad);
                      const volHi = maxVol + volPad;
                      const retLo = minRet - retPad;
                      const retHi = maxRet + retPad;
                      const plot = { left: 52, right: 332, top: 24, bottom: 192 };
                      const chartWidth = plot.right - plot.left;
                      const chartHeight = plot.bottom - plot.top;
                      const x = (vol: number) => plot.left + ((vol - volLo) / Math.max(volHi - volLo, 0.000001)) * chartWidth;
                      const y = (ret: number) => plot.bottom - ((ret - retLo) / Math.max(retHi - retLo, 0.000001)) * chartHeight;
                      const xTicks = [volLo, (volLo + volHi) / 2, volHi];
                      const yTicks = [retLo, (retLo + retHi) / 2, retHi];
                      const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(p.volatility).toFixed(1)} ${y(p.expected_return).toFixed(1)}`).join(' ');
                      return (
                        <>
                          <svg className="frontier-chart" viewBox="0 0 360 235" role="img" aria-label="Efficient frontier chart">
                            {xTicks.map((tick) => (
                              <g key={`x-${tick}`}>
                                <line x1={x(tick)} y1={plot.top} x2={x(tick)} y2={plot.bottom} className="chart-grid" />
                                <text x={x(tick)} y="211" className="chart-tick" textAnchor="middle">{fmtPct(tick, 0)}</text>
                              </g>
                            ))}
                            {yTicks.map((tick) => (
                              <g key={`y-${tick}`}>
                                <line x1={plot.left} y1={y(tick)} x2={plot.right} y2={y(tick)} className="chart-grid" />
                                <text x="44" y={y(tick) + 3} className="chart-tick" textAnchor="end">{fmtPct(tick, 0)}</text>
                              </g>
                            ))}
                            <line x1={plot.left} y1={plot.bottom} x2={plot.right} y2={plot.bottom} className="chart-axis" />
                            <line x1={plot.left} y1={plot.bottom} x2={plot.left} y2={plot.top} className="chart-axis" />
                            <path d={path} className="frontier-line" />
                            {points.map((p, i) => (
                              <circle key={`frontier-${i}`} cx={x(p.volatility)} cy={y(p.expected_return)} r="2.2" className="frontier-dot" />
                            ))}
                            {markers.map((m) => {
                              const cx = x(m.point.volatility);
                              const cy = y(m.point.expected_return);
                              const labelAnchor = cx > 255 ? 'end' : 'start';
                              const labelX = cx > 255 ? cx - 7 : cx + 7;
                              return (
                                <g key={m.label}>
                                  <circle cx={cx} cy={cy} r="5.2" className={m.cls} />
                                  <text x={labelX} y={cy - 7} className="frontier-label" textAnchor={labelAnchor}>{m.label}</text>
                                  <title>{`${m.label}: return ${fmtPct(m.point.expected_return)}, volatility ${fmtPct(m.point.volatility)}, Sharpe ${m.point.sharpe.toFixed(3)}`}</title>
                                </g>
                              );
                            })}
                            <text x="192" y="229" className="chart-axis-label" textAnchor="middle">Volatility / Risk</text>
                            <text x="13" y="108" className="chart-axis-label" textAnchor="middle" transform="rotate(-90 13 108)">Expected Return</text>
                          </svg>
                          <div className="frontier-summary">
                            <div><span>Optimized Return</span><strong>{fmtPct(frontier.optimized.expected_return)}</strong></div>
                            <div><span>Optimized Risk</span><strong>{fmtPct(frontier.optimized.volatility)}</strong></div>
                            <div><span>Sharpe</span><strong>{frontier.optimized.sharpe.toFixed(3)}</strong></div>
                          </div>
                          <div className="frontier-legend">
                            {markers.map((m) => (
                              <span key={m.label}><i className={m.cls} />{m.label}</span>
                            ))}
                          </div>
                          <div className="analytics-note">X-axis is volatility. Y-axis is expected return. The optimized point is the selected risk-adjusted portfolio.</div>
                        </>
                      );
                    })()}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Diversification Score</div>
                    <span className={`badge ${result.diversification_score.score >= 75 ? 'badge-buy' : result.diversification_score.score >= 55 ? 'badge-hold' : 'badge-sell'}`}>
                      {result.diversification_score.score}/100
                    </span>
                  </div>
                  <div className="analytics-body">
                    <div className="score-ring" style={{ background: `conic-gradient(var(--green-400) ${result.diversification_score.score * 3.6}deg, rgba(255,255,255,0.08) 0deg)` }}>
                      <div>{result.diversification_score.score}</div>
                    </div>
                    <div className="analytics-note">{result.diversification_score.message}</div>
                    <div className="panel-body-sm" style={{ padding: 0 }}>
                      <div className="stat-pair"><span>Effective positions</span><strong>{result.diversification_score.effective_positions} / {result.diversification_score.active_positions}</strong></div>
                      <div className="stat-pair"><span>Effective sectors</span><strong>{result.diversification_score.effective_sectors} / {result.diversification_score.sector_count}</strong></div>
                      <div className="stat-pair"><span>Largest stock</span><strong>{fmtPct(result.diversification_score.largest_weight)}</strong></div>
                      <div className="stat-pair"><span>Largest sector</span><strong>{fmtPct(result.diversification_score.largest_sector_weight)}</strong></div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid-2">
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Risk Contribution</div>
                    <span className="badge badge-blue">Top drivers</span>
                  </div>
                  <div className="analytics-body">
                    {result.risk_contributions.slice(0, 8).map((row) => (
                      <div className="risk-row-item" key={row.symbol}>
                        <div className="risk-row-head">
                          <strong>{row.symbol}</strong>
                          <span>{fmtPct(row.risk_contribution_pct)} risk · {fmtPct(row.weight)} weight</span>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill-after" style={{ width: `${clamp(row.risk_contribution_pct * 100, 2, 100)}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">Correlation Heatmap</div>
                    <span className="badge badge-blue">{result.correlation_matrix.symbols.length} names</span>
                  </div>
                  <div className="heatmap-wrap">
                    <table className="corr-table">
                      <thead>
                        <tr>
                          <th />
                          {result.correlation_matrix.symbols.map((symbol) => <th key={`corr-head-${symbol}`}>{symbol}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {result.correlation_matrix.symbols.map((rowSymbol, rowIndex) => (
                          <tr key={`corr-row-${rowSymbol}`}>
                            <th>{rowSymbol}</th>
                            {result.correlation_matrix.symbols.map((colSymbol, colIndex) => {
                              const value = result.correlation_matrix.values[rowIndex]?.[colIndex] ?? 0;
                              return (
                                <td key={`corr-${rowSymbol}-${colSymbol}`} style={{ background: corrCellBg(value) }}>
                                  {value.toFixed(2)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
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
                        const v = getAllocVals(a, result.current_portfolio_value, optimizedPortfolioValue, prices, enteredShares);
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
                              <div className="tbl-muted">{fmtCcy(v.curVal)} · {fmtPct(a.current_weight)}</div>
                            </td>
                            <td>
                              <div>{fmtShares(v.optShares)} sh</div>
                              <div className="tbl-muted">{fmtCcy(v.optVal)} · {fmtPct(a.optimized_weight)}</div>
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
                    const v = getAllocVals(a, result.current_portfolio_value, optimizedPortfolioValue, prices, enteredShares);
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
                    const v = getAllocVals(a, result.current_portfolio_value, optimizedPortfolioValue, prices, enteredShares);
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
                    const v = getAllocVals(a, result.current_portfolio_value, optimizedPortfolioValue, prices, enteredShares);
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
            {!selectedManagerId ? (
              /* ── MANAGER LOGIN / CREATE ACCOUNT ── */
              <div style={{ maxWidth: '440px', margin: '0 auto', width: '100%', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div style={{ textAlign: 'center', paddingTop: '0.25rem' }}>
                  <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text)', marginBottom: '0.2rem' }}>Fund Manager Platform</div>
                  <div style={{ fontSize: '13px', color: 'var(--text-3)' }}>
                    {authView === 'register' ? 'Create your account to get started.' : 'Log in to your account to continue.'}
                  </div>
                </div>

                {authView === 'register' ? (
                  <>
                    <div className="panel">
                      <div className="panel-head">
                        <div className="panel-title">Create Account</div>
                      </div>
                      <div className="panel-body" style={{ display: 'grid', gap: '0.75rem' }}>
                        <div className="form-field">
                          <div className="field-label">Full name</div>
                          <input className="field-input" value={managerName} onChange={(e) => setManagerName(e.target.value)} placeholder="e.g. Amina Bello" />
                        </div>
                        <div className="form-field">
                          <div className="field-label">Firm</div>
                          <input className="field-input" value={managerFirm} onChange={(e) => setManagerFirm(e.target.value)} placeholder="e.g. Lagos Asset Management" />
                        </div>
                        <div className="form-field">
                          <div className="field-label">Email address</div>
                          <input className="field-input" type="email" value={managerEmail} onChange={(e) => setManagerEmail(e.target.value)} placeholder="you@example.com" />
                        </div>
                        <div className="form-field">
                          <div className="field-label">Password <span style={{ color: 'var(--text-3)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>(min 8 characters)</span></div>
                          <input className="field-input" type="password" value={managerPassword} onChange={(e) => setManagerPassword(e.target.value)} placeholder="Enter a password" />
                        </div>
                        <button className="btn btn-primary" onClick={createManagerAccount} disabled={managerPassword.length > 0 && managerPassword.length < 8}>Create Account</button>
                        {managerPassword.length > 0 && managerPassword.length < 8 && (
                          <div style={{ fontSize: '12px', color: '#b45309' }}>{managerPassword.length}/8 characters — password too short</div>
                        )}
                        {workspaceStatus && <div className="banner banner-ok">{workspaceStatus}</div>}
                        {workspaceError && <div className="banner banner-error">{workspaceError}</div>}
                      </div>
                    </div>
                    <div style={{ textAlign: 'center', fontSize: '13px', color: 'var(--text-3)' }}>
                      Already have an account?{' '}
                      <button
                        style={{ background: 'none', border: 'none', padding: 0, color: '#2563eb', fontWeight: 600, fontSize: '13px', cursor: 'pointer' }}
                        onClick={() => { setAuthView('login'); setWorkspaceError(null); setWorkspaceStatus(null); }}
                      >
                        Log in
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="panel">
                      <div className="panel-head">
                        <div className="panel-title">Log In</div>
                      </div>
                      <div className="panel-body" style={{ display: 'grid', gap: '0.75rem' }}>
                        <div className="form-field">
                          <div className="field-label">Email address</div>
                          <input
                            className="field-input"
                            type="email"
                            value={loginEmail}
                            onChange={(e) => setLoginEmail(e.target.value)}
                            placeholder="you@example.com"
                            onKeyDown={(e) => e.key === 'Enter' && loginManager()}
                          />
                        </div>
                        <div className="form-field">
                          <div className="field-label">Password</div>
                          <input
                            className="field-input"
                            type="password"
                            value={loginPassword}
                            onChange={(e) => setLoginPassword(e.target.value)}
                            placeholder="Enter your password"
                            onKeyDown={(e) => e.key === 'Enter' && loginManager()}
                          />
                        </div>
                        <button className="btn btn-primary" onClick={loginManager}>Log In</button>
                        {workspaceError && <div className="banner banner-error">{workspaceError}</div>}
                      </div>
                    </div>
                    <div style={{ textAlign: 'center', fontSize: '13px', color: 'var(--text-3)' }}>
                      Don't have an account?{' '}
                      <button
                        style={{ background: 'none', border: 'none', padding: 0, color: '#2563eb', fontWeight: 600, fontSize: '13px', cursor: 'pointer' }}
                        onClick={() => { setAuthView('register'); setWorkspaceError(null); setWorkspaceStatus(null); }}
                      >
                        Create one
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              /* ── MANAGER DASHBOARD ── */
              <>
                {/* Manager identity bar */}
                <div className="panel">
                  <div className="panel-body" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
                    <div>
                      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text)' }}>{selectedManager!.name}</div>
                      <div style={{ fontSize: '12.5px', color: 'var(--text-3)', marginTop: '0.1rem' }}>
                        {selectedManager!.firm}{selectedManager!.email ? ` · ${selectedManager!.email}` : ''}
                      </div>
                    </div>
                    <div className="workspace-actions">
                      <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={exitWorkspace}>Switch Account</button>
                      <button className="btn btn-danger" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={deleteActiveWorkspace}>Delete Workspace</button>
                    </div>
                  </div>
                </div>

                {/* Two primary actions */}
                <div className="grid-2">
                  {/* New client */}
                  <div className="panel">
                    <div className="panel-head">
                      <div className="panel-title">New Client</div>
                    </div>
                    <div className="panel-body" style={{ display: 'grid', gap: '0.75rem' }}>
                      <div className="form-field">
                        <div className="field-label">Client name</div>
                        <input className="field-input" value={consumerName} onChange={(e) => setConsumerName(e.target.value)} placeholder="e.g. Chinedu Okafor" />
                      </div>
                      <div className="form-field">
                        <div className="field-label">Contact</div>
                        <input className="field-input" value={consumerEmail} onChange={(e) => setConsumerEmail(e.target.value)} placeholder="optional email or phone" />
                      </div>
                      <div className="form-field">
                        <div className="field-label">Portfolio status</div>
                        <div className="risk-row">
                          <button className={`risk-pill ${consumerPortfolioStatus === 'existing' ? 'active' : ''}`} onClick={() => setConsumerPortfolioStatus('existing')}>Has portfolio</button>
                          <button className={`risk-pill ${consumerPortfolioStatus === 'new' ? 'active' : ''}`} onClick={() => setConsumerPortfolioStatus('new')}>No portfolio</button>
                        </div>
                      </div>
                      <button
                        className="btn btn-primary"
                        onClick={async () => {
                          const consumer = await registerConsumer();
                          if (consumer) setActiveTab('input');
                        }}
                        disabled={!consumerName.trim()}
                      >
                        Add Client &amp; Continue
                      </button>
                      <button className="btn btn-ghost" onClick={() => { startNewConsumer(); setActiveTab('input'); }}>
                        Continue without saving
                      </button>
                      {workspaceStatus && <div className="banner banner-ok">{workspaceStatus}</div>}
                      {workspaceError && <div className="banner banner-error">{workspaceError}</div>}
                    </div>
                  </div>

                  {/* Select existing client */}
                  <div className="panel">
                    <div className="panel-head">
                      <div className="panel-title">Select Client</div>
                      <span className="badge badge-blue">{consumers.length} client{consumers.length === 1 ? '' : 's'}</span>
                    </div>
                    {consumers.length > 0 ? (
                      <div>
                        {consumers.map((consumer) => (
                          <button
                            key={consumer.id}
                            className="manager-login-row"
                            onClick={() => { selectConsumer(consumer.id); setActiveTab('input'); }}
                          >
                            <div style={{ textAlign: 'left' }}>
                              <div className="manager-login-name">{consumer.name}</div>
                              <div className="manager-login-firm">
                                {consumer.email || 'No contact'} · {consumer.consumer_has_portfolio ? 'Has portfolio' : 'New portfolio'}
                                {(consumer.portfolio_count ?? 0) > 0 ? ` · ${consumer.portfolio_count} run${consumer.portfolio_count === 1 ? '' : 's'}` : ''}
                              </div>
                            </div>
                            <Icon.chevronRight />
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="empty-state" style={{ padding: '2rem 1rem' }}>
                        <div className="empty-icon"><Icon.user /></div>
                        <div className="empty-state-title">No clients yet</div>
                        <div className="empty-state-sub">Add a new client using the form on the left.</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Save portfolio utility panel */}
                <div className="panel">
                  <div className="panel-head">
                    <div className="panel-title">{consumerHasPortfolio ? 'Save Current Equity Portfolio' : 'Save New Consumer Portfolio'}</div>
                    <div className="workspace-actions">
                      <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={() => setActiveTab('input')}>
                        Go to Input
                      </button>
                      <button className="btn btn-ghost" style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }} onClick={saveCurrentPortfolio}>
                        {consumerHasPortfolio ? 'Save Current' : 'Save New'}
                      </button>
                    </div>
                  </div>
                  <div className="panel-body">
                    <div className="grid-3">
                      <div className="form-field">
                        <div className="field-label">Managed consumer</div>
                        <select className="field-input" value={selectedConsumerId} onChange={(e) => selectConsumer(e.target.value)} disabled={!selectedManagerId || consumers.length === 0}>
                          <option value="">{consumers.length ? 'Select consumer' : 'No consumers registered'}</option>
                          {consumers.map((consumer) => (
                            <option key={consumer.id} value={consumer.id}>
                              {consumer.name} — {consumer.consumer_has_portfolio ? 'has portfolio' : 'new portfolio'}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="form-field">
                        <div className="field-label">Portfolio name</div>
                        <input className="field-input" value={portfolioName} onChange={(e) => setPortfolioName(e.target.value)} />
                      </div>
                      <div className="form-field">
                        <div className="field-label">Consumer name</div>
                        <input className="field-input" value={consumerName} onChange={(e) => setConsumerName(e.target.value)} placeholder="e.g. Chinedu Okafor" />
                      </div>
                      <div className="form-field">
                        <div className="field-label">Consumer contact</div>
                        <input className="field-input" value={consumerEmail} onChange={(e) => setConsumerEmail(e.target.value)} placeholder="optional email or phone" />
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
                    {workspaceStatus && <div className="banner banner-ok" style={{ marginTop: '0.75rem' }}>{workspaceStatus}</div>}
                    {workspaceError && <div className="banner banner-error" style={{ marginTop: '0.75rem' }}>{workspaceError}</div>}
                  </div>
                </div>

                {/* Saved portfolios */}
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
                              {(portfolio.consumer_name || 'Unnamed consumer')} · {mandateLabels[portfolio.mandate_profile]} · {portfolio.risk_profile} · {(portfolio.consumer_has_portfolio ?? portfolio.holdings.length > 0) ? `${portfolio.holdings.length} holding${portfolio.holdings.length === 1 ? '' : 's'}` : `new portfolio from ${fmtCcy(portfolio.initial_cash_naira ?? 0)}`}
                            </div>
                            {portfolio.latest_result_summary && (
                              <div className="workspace-run">
                                Last run {fmtRelative(portfolio.latest_result_summary.generated_at)} · {portfolio.latest_result_summary.compliance_status} · Value {fmtCcy(portfolio.latest_result_summary.optimized_portfolio_value ?? portfolio.latest_result_summary.portfolio_value)} · Sharpe {portfolio.latest_result_summary.optimized_sharpe.toFixed(3)}
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
                      <div className="empty-state-sub">Save holdings from the Input tab to track optimization history.</div>
                    </div>
                  )}
                </div>
              </>
            )}
          </>
        )}

        {/* ══ CLIENTS ══ */}
        {activeTab === 'clients' && (
          <>
            {/* Top bar: manager selector + batch actions */}
            <div className="panel">
              <div className="panel-head">
                <div className="panel-title">Client Portfolios</div>
                <div className="workspace-actions">
                  {selectedManager && <span className="badge badge-blue">{selectedManager.firm}</span>}
                  {batchSelected.size > 0 && (
                    <>
                      <span className="badge badge-hist">{batchSelected.size} selected</span>
                      <button
                        className="btn btn-ghost"
                        style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }}
                        onClick={clearBatchSelection}
                      >
                        Clear
                      </button>
                      <button
                        className="btn btn-primary"
                        style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }}
                        onClick={runBatchOptimize}
                        disabled={batchRunning || !selectedManagerId}
                      >
                        {batchRunning ? 'Running…' : `Optimize ${batchSelected.size} Portfolio${batchSelected.size === 1 ? '' : 's'}`}
                      </button>
                    </>
                  )}
                  {batchSelected.size === 0 && savedPortfolios.length > 0 && (
                    <button
                      className="btn btn-ghost"
                      style={{ fontSize: '10.5px', padding: '0.3rem 0.7rem' }}
                      onClick={selectAllPortfolios}
                    >
                      Select All
                    </button>
                  )}
                </div>
              </div>
              <div className="panel-body" style={{ paddingBottom: '0.5rem' }}>
                {managers.length > 0 && (
                  <div className="form-field" style={{ maxWidth: '320px' }}>
                    <div className="field-label">Active manager</div>
                    <select className="field-input" value={selectedManagerId} onChange={(e) => selectWorkspace(e.target.value)}>
                      <option value="">Select workspace</option>
                      {managers.map((m) => (
                        <option key={m.id} value={m.id}>{m.name} — {m.firm}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            </div>

            {/* Batch result banner */}
            {batchResult && (
              <div className={`banner ${batchResult.failed > 0 ? 'banner-warn' : 'banner-ok'}`}>
                <p>
                  Batch complete — <strong>{batchResult.succeeded}</strong> succeeded
                  {batchResult.failed > 0 && <>, <strong>{batchResult.failed}</strong> failed</>}.
                </p>
                {batchResult.results.filter((r) => !r.success).map((r) => (
                  <p key={r.portfolio_id} style={{ fontSize: '12px', marginTop: '0.2rem' }}>
                    ✕ {r.consumer_name || r.portfolio_id}: {r.error}
                  </p>
                ))}
              </div>
            )}

            {workspaceError && <div className="banner banner-error">{workspaceError}</div>}

            {/* Portfolio list — grouped by consumer */}
            {!selectedManagerId ? (
              <div className="empty-state">
                <div className="empty-icon"><Icon.users /></div>
                <div className="empty-state-title">No workspace selected</div>
                <div className="empty-state-sub">Select a fund manager workspace above to view client portfolios.</div>
              </div>
            ) : savedPortfolios.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon"><Icon.users /></div>
                <div className="empty-state-title">No saved portfolios</div>
                <div className="empty-state-sub">Save client portfolios from the Workspace tab to track them here.</div>
                <button className="btn btn-primary" style={{ marginTop: '1rem' }} onClick={() => setActiveTab('workspace')}>
                  Go to Workspace
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {savedPortfolios.map((portfolio) => {
                  const isExpanded = expandedPortfolioId === portfolio.id;
                  const runs: any[] = portfolioRuns[portfolio.id] ?? [];
                  const isLoadingRuns = runsLoading === portfolio.id;
                  const isChecked = batchSelected.has(portfolio.id);
                  const summary = portfolio.latest_result_summary;

                  return (
                    <div
                      key={portfolio.id}
                      className="panel"
                      style={{ border: isChecked ? '1px solid var(--accent)' : undefined }}
                    >
                      {/* Portfolio header row */}
                      <div
                        className="panel-head"
                        style={{ cursor: 'pointer', userSelect: 'none' }}
                        onClick={() => loadPortfolioRuns(portfolio.id)}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
                          {/* Batch select checkbox */}
                          <input
                            type="checkbox"
                            checked={isChecked}
                            style={{ width: '13px', height: '13px', cursor: 'pointer', flexShrink: 0, accentColor: 'var(--accent)' }}
                            onClick={(e) => e.stopPropagation()}
                            onChange={() => toggleBatchSelect(portfolio.id)}
                          />
                          <div style={{ minWidth: 0 }}>
                            <div className="panel-title" style={{ fontSize: '13px' }}>{portfolio.name}</div>
                            <div style={{ fontSize: '12px', color: 'var(--text-3)', marginTop: '1px' }}>
                              {portfolio.consumer_name || 'Unnamed client'} · {mandateLabels[portfolio.mandate_profile]} · {portfolio.risk_profile}
                            </div>
                          </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexShrink: 0 }}>
                          {summary && (
                            <span className={`badge ${getComplianceBadge(summary.compliance_status as any).cls}`} style={{ fontSize: '10.5px' }}>
                              {summary.compliance_status}
                            </span>
                          )}
                          {summary && (
                            <span style={{ fontSize: '12px', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                              {fmtCcy(summary.optimized_portfolio_value ?? summary.portfolio_value)}
                            </span>
                          )}
                          <div style={{ width: '14px', height: '14px', color: 'var(--text-3)', transform: isExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms' }}>
                            <Icon.chevron />
                          </div>
                        </div>
                      </div>

                      {/* Expanded: latest summary + run history */}
                      {isExpanded && (
                        <div>
                          {/* Latest snapshot metrics */}
                          {summary && (
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: 'var(--border)', borderBottom: '1px solid var(--border)' }}>
                              {[
                                { label: 'Portfolio value',  value: fmtCcy(summary.optimized_portfolio_value ?? summary.portfolio_value) },
                                { label: 'Expected return',  value: fmtPct(summary.optimized_expected_return) },
                                { label: 'Sharpe ratio',     value: summary.optimized_sharpe.toFixed(3) },
                                { label: 'Last run',         value: fmtRelative(summary.generated_at) },
                              ].map((m) => (
                                <div key={m.label} className="metric-card" style={{ padding: '0.65rem 0.9rem' }}>
                                  <div className="metric-label">{m.label}</div>
                                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13.5px', fontWeight: 600, color: 'var(--text)', marginTop: '0.2rem' }}>{m.value}</div>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Quick-action buttons */}
                          <div style={{ display: 'flex', gap: '0.5rem', padding: '0.65rem 1rem', borderBottom: '1px solid var(--border)', flexWrap: 'wrap' }}>
                            <button className="btn btn-ghost" style={{ fontSize: '11px', padding: '0.3rem 0.7rem' }} onClick={() => loadSavedPortfolio(portfolio)}>
                              Load to Input
                            </button>
                            <button className="btn btn-primary" style={{ fontSize: '11px', padding: '0.3rem 0.7rem' }} onClick={() => optimizeSavedPortfolio(portfolio)}>
                              Optimize Now
                            </button>
                            {(summary?.added_symbols?.length ?? 0) > 0 && (
                              <span style={{ fontSize: '11.5px', color: 'var(--text-3)', alignSelf: 'center' }}>
                                Last added: {summary?.added_symbols?.slice(0, 4).join(', ')}{(summary?.added_symbols?.length ?? 0) > 4 ? ` +${(summary?.added_symbols?.length ?? 0) - 4}` : ''}
                              </span>
                            )}
                            {(summary?.removed_symbols?.length ?? 0) > 0 && (
                              <span style={{ fontSize: '11.5px', color: 'var(--red-400)', alignSelf: 'center' }}>
                                Removed: {summary?.removed_symbols?.slice(0, 4).join(', ')}{(summary?.removed_symbols?.length ?? 0) > 4 ? ` +${(summary?.removed_symbols?.length ?? 0) - 4}` : ''}
                              </span>
                            )}
                          </div>

                          {/* Run history */}
                          <div style={{ padding: '0.5rem 0' }}>
                            <div style={{ padding: '0.4rem 1rem 0.3rem', fontSize: '10.5px', fontWeight: 600, letterSpacing: '1px', textTransform: 'uppercase', color: 'var(--text-3)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                              Portfolio State History ({runs.length} snapshot{runs.length === 1 ? '' : 's'})
                            </div>
                            {isLoadingRuns ? (
                              <div style={{ padding: '1rem', fontSize: '12.5px', color: 'var(--text-3)' }}>Loading history…</div>
                            ) : runs.length === 0 ? (
                              <div style={{ padding: '0.75rem 1rem', fontSize: '12.5px', color: 'var(--text-3)' }}>
                                No runs recorded yet. Click "Optimize Now" to record the first snapshot.
                              </div>
                            ) : (
                              <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '220px' }}>
                                <table className="data-table">
                                  <thead style={{ position: 'sticky', top: 0, zIndex: 1 }}>
                                    <tr>
                                      <th style={{ width: '2.5rem' }}>#</th>
                                      <th>Date &amp; Time</th>
                                      <th>Portfolio Value</th>
                                      <th>Post-Trade Value</th>
                                      <th>Exp. Return</th>
                                      <th>Sharpe</th>
                                      <th>Compliance</th>
                                      <th>Holdings Added / Removed</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {runs.map((run: any, idx: number) => {
                                      const s = run.summary;
                                      if (!s) return null;
                                      const badge = getComplianceBadge(s.compliance_status);
                                      const added: string[] = s.added_symbols ?? [];
                                      const removed: string[] = s.removed_symbols ?? [];
                                      const runDate = run.created_at ?? s.generated_at;
                                      const runNum = runs.length - idx;
                                      return (
                                        <tr key={run.id}>
                                          <td style={{ color: 'var(--text-3)', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>{runNum}</td>
                                          <td style={{ whiteSpace: 'nowrap', fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>
                                            {fmtDateTime(runDate)}
                                          </td>
                                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>{fmtCcy(s.portfolio_value)}</td>
                                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>{fmtCcy(s.optimized_portfolio_value ?? s.portfolio_value)}</td>
                                          <td className={s.optimized_expected_return >= 0 ? 'up' : 'down'} style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>
                                            {fmtPct(s.optimized_expected_return)}
                                          </td>
                                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>{s.optimized_sharpe.toFixed(3)}</td>
                                          <td><span className={`badge ${badge.cls}`} style={{ fontSize: '10px' }}>{s.compliance_status}</span></td>
                                          <td style={{ fontSize: '11px' }}>
                                            {added.length > 0 && (
                                              <span style={{ color: 'var(--green-300)', marginRight: '0.4rem' }}>
                                                +{added.slice(0, 3).join(', ')}{added.length > 3 ? ` +${added.length - 3}` : ''}
                                              </span>
                                            )}
                                            {removed.length > 0 && (
                                              <span style={{ color: 'var(--red-400)' }}>
                                                −{removed.slice(0, 3).join(', ')}{removed.length > 3 ? ` +${removed.length - 3}` : ''}
                                              </span>
                                            )}
                                            {added.length === 0 && removed.length === 0 && (
                                              <span style={{ color: 'var(--text-3)' }}>—</span>
                                            )}
                                          </td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}


      </main>
    </div>
  );
}

export default App;
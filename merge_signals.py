"""
merge_signals.py
────────────────
Runs XGBoost, Random Forest, and LSTM in sequence for every valid NSE equity
symbol and writes the merged consensus output directly to signal_store.csv.

Unlike the standalone model scripts, this orchestrator imports the training
functions directly and disables per-symbol charts/model exports so one run can
produce the signal store without generating hundreds of extra files.
"""

import gc
import os
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import numpy as np

from preprocess import preprocess
from xgb_model import xgboost_train
from rf_model import rf_train
from lstm_model import lstm_train, DEVICE
from classification_models import (
    lstm_direction_train,
    rf_direction_train,
    xgboost_direction_train,
)

MIN_ROWS = 500
OUTPUT_FILE = 'signal_store.csv'
REPORT_FILE = 'merge_signals_report.txt'
# Merge-mode defaults trade a little model fidelity for much faster iteration.
CHECKPOINT_EVERY = 25
FAST_XGB_IN_MERGE = True
FAST_LSTM_IN_MERGE = True
FAST_LSTM_EPOCHS = 4
FAST_LSTM_LOOKBACK = 20
FAST_LSTM_BATCH_SIZE = 256
FAST_LSTM_PATIENCE = 2
MAX_SYMBOLS = 40
RUN_LSTM_ON_CPU = True
LSTM_MAX_SYMBOLS_ON_CPU = 12
USE_CLASSIFICATION_SIGNALS = True
MIN_MODEL_R2_FOR_VOTE = 0.0
MIN_MODEL_DIRECTION_FOR_VOTE = 50.0
MIN_CLASS_PROB_EDGE_FOR_VOTE = 0.05


def is_equity(symbol):
    """Filter out non-equity instruments (bonds, ETFs, REITs)."""
    non_equity_keywords = [
        'FGS', 'FG1', 'FG2', 'FG6', 'FG9', 'FGB',
        'LAB', 'DIF', 'IAO', 'UCAP2', 'ZAM', 'FCM2', 'FBQ', 'EPF',
        'CSF', 'LASUK', 'FGSUK', 'FHSUK', 'FID2', 'TAJ',
        'LAFARGEWAPCO', 'CHAPELHILL', 'IBTCINFRA',
        'ETF', 'LOTUS', 'NEWGOLD', 'NGX30', 'NGX50', 'STANBICETF',
        'SIAMLETF', 'GREENWETF', 'VETGRIF', 'VETINDETF', 'VSPBOND',
        'NGXAFR', 'LOTUSHAL',
        'REIT', 'UHOMREIT', 'SFSREIT', 'UPDCREIT', 'UHOREIT',
        'NIGFUND', 'IBTCNEF', 'NESF', 'NIDF', 'MERGROWTH', 'MERVALUE',
        'NGXMERI', 'NGXPENSION',
    ]
    return not any(kw in symbol.upper() for kw in non_equity_keywords)


def load_market_data(filepath='PRICE_LIST.csv'):
    # merge_signals reads the market data once, then slices it per symbol for each model.
    df = preprocess(filepath)
    symbol_cols = [col for col in df.columns if col.startswith('SYMBOL_')]
    if not symbol_cols:
        raise ValueError("No one-hot encoded SYMBOL_ columns found in preprocessed data.")
    df = df.copy()
    df['SYMBOL'] = df[symbol_cols].idxmax(axis=1).str.replace('SYMBOL_', '', regex=False)
    return df


def get_int_env(name, default):
    # Environment variables let you speed up or widen the run without editing code.
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_bool_env(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def is_quality_vote(model, min_r2, min_direction):
    if model is None:
        return False
    if model.get('Model_Type') == 'classification':
        probability = model.get('Buy_Probability (%)')
        if probability is None:
            return False
        if abs(float(probability) / 100.0 - 0.5) < float(os.getenv("MERGE_MIN_CLASS_PROB_EDGE_FOR_VOTE", MIN_CLASS_PROB_EDGE_FOR_VOTE)):
            return False
    return (
        float(model.get('R2', 0.0)) >= min_r2
        and float(model.get('Direction_Acc (%)', 0.0)) >= min_direction
        and float(model.get('Confidence_Score', 0.0)) > 0.0
    )


def merge_results(xgb_res, rf_res, lstm_res):
    # The final signal store is symbol-centric: each row combines all model outputs
    # plus a quality-gated consensus rule the optimizer can consume later.
    rows = []
    min_vote_r2 = float(os.getenv("MERGE_MIN_MODEL_R2_FOR_VOTE", MIN_MODEL_R2_FOR_VOTE))
    min_vote_direction = float(os.getenv("MERGE_MIN_MODEL_DIRECTION_FOR_VOTE", MIN_MODEL_DIRECTION_FOR_VOTE))

    for symbol in sorted(set(xgb_res) | set(rf_res) | set(lstm_res)):
        xg = xgb_res.get(symbol)
        rf = rf_res.get(symbol)
        ls = lstm_res.get(symbol)
        model_slots = [('XGB', xg), ('RF', rf), ('LSTM', ls)]
        available = [model for _, model in model_slots if model is not None]
        if len(available) < 2:
            continue

        qualified = [
            (name, model) for name, model in model_slots
            if is_quality_vote(model, min_vote_r2, min_vote_direction)
        ]
        vote_models = [model for _, model in qualified]
        vote_source = 'quality' if vote_models else 'none'

        if not vote_models:
            signals = []
        else:
            signals = [model['Signal'] for model in vote_models]
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        model_count = len(available)
        qualified_count = len(vote_models)

        consensus = (
            'CONFLICT' if qualified_count == 0 else
            'BUY' if buy_count == qualified_count else
            'SELL' if sell_count == qualified_count else
            'BUY' if buy_count > sell_count else
            'SELL' if sell_count > buy_count else
            'CONFLICT'
        )

        tier = (
            1 if qualified_count == 3 and (buy_count == 3 or sell_count == 3) else
            2 if qualified_count >= 2 and consensus in {'BUY', 'SELL'} else
            3
        )
        qualified_names = ",".join(name for name, _ in qualified)
        quality_models = vote_models if vote_models else []

        rows.append({
            'Symbol': symbol,
            'Last_Close (₦)': available[0]['Last_Close (₦)'],
            'XGB_Signal': xg['Signal'] if xg else None,
            'XGB_Return (%)': xg['Predicted_Return (%)'] if xg else None,
            'XGB_R2': xg['R2'] if xg else None,
            'XGB_Direction_Acc': xg['Direction_Acc (%)'] if xg else None,
            'XGB_Balanced_Acc': xg.get('Balanced_Acc (%)') if xg else None,
            'XGB_Buy_Probability': xg.get('Buy_Probability (%)') if xg else None,
            'XGB_F1_BUY': xg.get('F1_BUY (%)') if xg else None,
            'XGB_Confidence': xg['Confidence_Score'] if xg else None,
            'RF_Signal': rf['Signal'] if rf else None,
            'RF_Return (%)': rf['Predicted_Return (%)'] if rf else None,
            'RF_R2': rf['R2'] if rf else None,
            'RF_Direction_Acc': rf['Direction_Acc (%)'] if rf else None,
            'RF_Balanced_Acc': rf.get('Balanced_Acc (%)') if rf else None,
            'RF_Buy_Probability': rf.get('Buy_Probability (%)') if rf else None,
            'RF_F1_BUY': rf.get('F1_BUY (%)') if rf else None,
            'RF_Confidence': rf['Confidence_Score'] if rf else None,
            'LSTM_Signal': ls['Signal'] if ls else None,
            'LSTM_Return (%)': ls['Predicted_Return (%)'] if ls else None,
            'LSTM_R2': ls['R2'] if ls else None,
            'LSTM_Direction_Acc': ls['Direction_Acc (%)'] if ls else None,
            'LSTM_Balanced_Acc': ls.get('Balanced_Acc (%)') if ls else None,
            'LSTM_Buy_Probability': ls.get('Buy_Probability (%)') if ls else None,
            'LSTM_F1_BUY': ls.get('F1_BUY (%)') if ls else None,
            'LSTM_Confidence': ls['Confidence_Score'] if ls else None,
            'XGB_Quality_Pass': is_quality_vote(xg, min_vote_r2, min_vote_direction),
            'RF_Quality_Pass': is_quality_vote(rf, min_vote_r2, min_vote_direction),
            'LSTM_Quality_Pass': is_quality_vote(ls, min_vote_r2, min_vote_direction),
            'Models_Run': model_count,
            'Qualified_Models': qualified_count,
            'Qualified_Model_Names': qualified_names,
            'Vote_Source': vote_source,
            'Consensus_Signal': consensus,
            'Consensus_Tier': tier,
            'Avg_R2': round(np.mean([model['R2'] for model in available]), 4),
            'Avg_Quality_R2': round(np.mean([model['R2'] for model in quality_models]), 4) if quality_models else 0.0,
            'Avg_Confidence': round(np.mean([model['Confidence_Score'] for model in quality_models]), 4) if quality_models else 0.0,
        })

    return pd.DataFrame(rows)


def save_report(df):
    # This text report is a quick offline summary for research/debugging.
    lines = [
        "=" * 60,
        "        SIGNAL STORE — MERGE REPORT",
        "=" * 60,
        f"\nTotal symbols          : {len(df)}",
        f"3-model results        : {(df['Models_Run'] == 3).sum()}",
        f"2-model results        : {(df['Models_Run'] == 2).sum()}",
        f"Quality-voted symbols  : {(df['Qualified_Models'] > 0).sum() if 'Qualified_Models' in df else 0}",
        f"2+ quality votes       : {(df['Qualified_Models'] >= 2).sum() if 'Qualified_Models' in df else 0}",
        "\n── Consensus tier breakdown ─────────────────────────",
        f"  Tier 1 (all agree)   : {(df['Consensus_Tier'] == 1).sum()} symbols",
        f"  Tier 2 (majority)    : {(df['Consensus_Tier'] == 2).sum()} symbols",
        f"  Tier 3 (conflicted)  : {(df['Consensus_Tier'] == 3).sum()} symbols",
        "\n── Signal direction ──────────────────────────────────",
        f"  BUY      : {(df['Consensus_Signal'] == 'BUY').sum()}",
        f"  SELL     : {(df['Consensus_Signal'] == 'SELL').sum()}",
        f"  CONFLICT : {(df['Consensus_Signal'] == 'CONFLICT').sum()}",
        "\n── Tier 1 BUY — highest conviction ──────────────────",
    ]

    cols = [
        'Symbol', 'Last_Close (₦)', 'XGB_Return (%)', 'RF_Return (%)',
        'LSTM_Return (%)', 'Qualified_Models', 'Avg_Quality_R2', 'Avg_Confidence'
    ]
    available_cols = [col for col in cols if col in df.columns]

    tier1_buy = df[
        (df['Consensus_Tier'] == 1) & (df['Consensus_Signal'] == 'BUY')
    ].sort_values('Avg_Confidence', ascending=False)
    lines.append(tier1_buy[available_cols].to_string(index=False) if not tier1_buy.empty else "  None.")

    lines.append("\n── Tier 1 SELL — highest conviction ─────────────────")
    tier1_sell = df[
        (df['Consensus_Tier'] == 1) & (df['Consensus_Signal'] == 'SELL')
    ].sort_values('Avg_Confidence', ascending=False)
    lines.append(tier1_sell[available_cols].to_string(index=False) if not tier1_sell.empty else "  None.")

    lines.append("\n── Average quality score per model ───────────────────")
    for col, label in [('XGB_R2', 'XGBoost'), ('RF_R2', 'Random Forest'), ('LSTM_R2', 'LSTM')]:
        values = df[col].dropna() if col in df.columns else pd.Series(dtype='float64')
        if len(values):
            lines.append(f"  {label:<16} : {values.mean():.4f}  (n={len(values)})")

    lines.append("\n── Average direction accuracy per model ──────────────")
    for col, label in [('XGB_Direction_Acc', 'XGBoost'), ('RF_Direction_Acc', 'Random Forest'), ('LSTM_Direction_Acc', 'LSTM')]:
        values = df[col].dropna() if col in df.columns else pd.Series(dtype='float64')
        if len(values):
            lines.append(f"  {label:<16} : {values.mean():.2f}%  (n={len(values)})")

    lines.append("\n── Quality pass rate per model ───────────────────────")
    for col, label in [('XGB_Quality_Pass', 'XGBoost'), ('RF_Quality_Pass', 'Random Forest'), ('LSTM_Quality_Pass', 'LSTM')]:
        values = df[col].dropna() if col in df.columns else pd.Series(dtype='bool')
        if len(values):
            lines.append(f"  {label:<16} : {values.astype(bool).mean() * 100:.1f}%  (n={len(values)})")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    print(report)
    with open(REPORT_FILE, 'w') as file:
        file.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


def main():
    print("=" * 60)
    print("  MERGE SIGNALS — XGBoost + Random Forest + LSTM")
    print("=" * 60)

    print("\nLoading data...")
    # All three models run symbol-by-symbol off the same preprocessed table.
    market_df = load_market_data('PRICE_LIST.csv')
    symbol_counts = market_df[market_df['SYMBOL'].apply(is_equity)].groupby('SYMBOL').size()
    valid_symbols = [
        symbol for symbol in symbol_counts.sort_values(ascending=False).index
        if symbol_counts[symbol] >= MIN_ROWS
    ]

    max_symbols = get_int_env("MERGE_MAX_SYMBOLS", MAX_SYMBOLS)
    if max_symbols and max_symbols > 0:
        valid_symbols = valid_symbols[:max_symbols]

    use_classification = get_bool_env("MERGE_CLASSIFICATION_SIGNALS", USE_CLASSIFICATION_SIGNALS)
    fast_lstm = os.getenv("MERGE_FAST_LSTM", "1") != "0"
    run_lstm_default = True if DEVICE.type != "cpu" else RUN_LSTM_ON_CPU
    run_lstm = get_bool_env("MERGE_RUN_LSTM", run_lstm_default)
    fast_lstm_epochs = get_int_env("MERGE_LSTM_EPOCHS", FAST_LSTM_EPOCHS)
    fast_lstm_lookback = get_int_env("MERGE_LSTM_LOOKBACK", FAST_LSTM_LOOKBACK)
    fast_lstm_batch_size = get_int_env("MERGE_LSTM_BATCH_SIZE", FAST_LSTM_BATCH_SIZE)
    fast_lstm_patience = get_int_env("MERGE_LSTM_PATIENCE", FAST_LSTM_PATIENCE)
    lstm_max_symbols_on_cpu = get_int_env("MERGE_LSTM_MAX_SYMBOLS_ON_CPU", LSTM_MAX_SYMBOLS_ON_CPU)
    lstm_verbose = get_bool_env("MERGE_LSTM_VERBOSE", True)
    debug_errors = get_bool_env("MERGE_DEBUG_ERRORS", False)

    print(f"Found {len(valid_symbols)} valid equity symbols | device: {DEVICE}")
    print(
        f"Fast merge settings: XGB={'on' if FAST_XGB_IN_MERGE else 'off'}, "
        f"signal_mode={'classification' if use_classification else 'regression'}, "
        f"LSTM={'on' if run_lstm else 'off'}, "
            f"fast_lstm={'on' if fast_lstm else 'off'}, "
            f"max_symbols={max_symbols or 'all'}, "
            f"cpu_lstm_limit={lstm_max_symbols_on_cpu if DEVICE.type == 'cpu' else 'n/a'}\n"
    )
    if DEVICE.type == "cpu" and run_lstm:
        if lstm_max_symbols_on_cpu <= 0:
            print("CPU detected: running compact fast LSTM for every selected symbol.")
        else:
            print(
                "CPU detected: running compact fast LSTM only for the first "
                f"{lstm_max_symbols_on_cpu} symbols by default."
            )
            print("Set MERGE_LSTM_MAX_SYMBOLS_ON_CPU=0 to run LSTM for every selected symbol.")
        print()
    elif DEVICE.type == "cpu":
        print("CPU detected: LSTM disabled. Set MERGE_RUN_LSTM=1 to enable compact fast LSTM.\n")

    xgb_res = {}
    rf_res = {}
    lstm_res = {}
    total = len(valid_symbols)

    for index, symbol in enumerate(valid_symbols, start=1):
        symbol_df = market_df[market_df['SYMBOL'] == symbol].copy()
        print(f"\n[{index}/{total}] {symbol}")

        # Each model is isolated so a failure in one does not kill the whole merge run.
        try:
            if use_classification:
                result = xgboost_direction_train(
                    symbol_df.copy(),
                    symbol_name=symbol,
                    save_artifacts=False,
                    fast_mode=FAST_XGB_IN_MERGE,
                    verbose=False,
                )
            else:
                result = xgboost_train(
                    symbol_df.copy(),
                    symbol_name=symbol,
                    save_artifacts=False,
                    fast_mode=FAST_XGB_IN_MERGE,
                    verbose=False,
                )
            xgb_res[symbol] = result
            if use_classification:
                print(
                    f"  XGB  Acc={result['Direction_Acc (%)']:.2f}% "
                    f"P(BUY)={result['Buy_Probability (%)']:.2f}%  {result['Signal']}"
                )
            else:
                print(f"  XGB  R²={result['R2']:.4f}  {result['Signal']}")
        except Exception as exc:
            print(f"  XGB  skipped: {exc}")

        try:
            if use_classification:
                result = rf_direction_train(
                    symbol_df.copy(),
                    symbol_name=symbol,
                    save_artifacts=False,
                    verbose=False,
                )
            else:
                result = rf_train(
                    symbol_df.copy(),
                    symbol_name=symbol,
                    save_artifacts=False,
                    verbose=False,
                )
            rf_res[symbol] = result
            if use_classification:
                print(
                    f"  RF   Acc={result['Direction_Acc (%)']:.2f}% "
                    f"P(BUY)={result['Buy_Probability (%)']:.2f}%  {result['Signal']}"
                )
            else:
                print(f"  RF   R²={result['R2']:.4f}  {result['Signal']}")
        except Exception as exc:
            print(f"  RF   skipped: {exc}")

        should_run_lstm = run_lstm and (
            DEVICE.type != "cpu"
            or lstm_max_symbols_on_cpu <= 0
            or index <= lstm_max_symbols_on_cpu
        )

        if should_run_lstm:
            try:
                print("  LSTM starting...")
                if use_classification:
                    result = lstm_direction_train(
                        symbol_df.copy(),
                        symbol_name=symbol,
                        save_artifacts=False,
                        verbose=lstm_verbose,
                        epochs_override=fast_lstm_epochs if fast_lstm else None,
                        lookback_override=fast_lstm_lookback if fast_lstm else None,
                        batch_size_override=fast_lstm_batch_size if fast_lstm else None,
                        patience_override=fast_lstm_patience if fast_lstm else None,
                        fast_mode=fast_lstm,
                    )
                else:
                    result = lstm_train(
                        symbol_df.copy(),
                        symbol_name=symbol,
                        save_artifacts=False,
                        verbose=lstm_verbose,
                        epochs_override=fast_lstm_epochs if fast_lstm else None,
                        lookback_override=fast_lstm_lookback if fast_lstm else None,
                        batch_size_override=fast_lstm_batch_size if fast_lstm else None,
                        patience_override=fast_lstm_patience if fast_lstm else None,
                        fast_mode=fast_lstm,
                    )
                lstm_res[symbol] = result
                if use_classification:
                    print(
                        f"  LSTM Acc={result['Direction_Acc (%)']:.2f}% "
                        f"P(BUY)={result['Buy_Probability (%)']:.2f}%  {result['Signal']}"
                    )
                else:
                    print(f"  LSTM R²={result['R2']:.4f}  {result['Signal']}")
            except Exception as exc:
                print(f"  LSTM skipped: {exc}")
                if debug_errors:
                    traceback.print_exc()
        elif run_lstm:
            print("  LSTM skipped (CPU fast-run symbol limit reached)")
        else:
            print("  LSTM skipped (disabled for this merge run)")

        # Periodic checkpoints protect long runs from losing all progress.
        if index % CHECKPOINT_EVERY == 0 or index == total:
            merged = merge_results(xgb_res, rf_res, lstm_res)
            merged.to_csv(OUTPUT_FILE, index=False)
            print(f"  Checkpoint saved to {OUTPUT_FILE} ({index}/{total})")
            gc.collect()

    print(f"\n{'=' * 60}")
    signal_store = merge_results(xgb_res, rf_res, lstm_res)
    signal_store.to_csv(OUTPUT_FILE, index=False)
    print(f"{OUTPUT_FILE} saved ({len(signal_store)} symbols)")
    save_report(signal_store)


if __name__ == "__main__":
    main()

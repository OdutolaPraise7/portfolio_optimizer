import re
import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess import preprocess

DATA_FILEPATH = 'PRICE_LIST.csv'
TARGET_HORIZON = 20

# ── SPEED & QUALITY CONFIG ──────────────────────────────────────────────────────
# Minimum number of rows a symbol must have to be included in training.
# 500 rows ≈ 2 years of daily trading data — the standard practical minimum
# for financial time series ML.
MIN_ROWS = 500

# Number of trees in the forest.
# 200 gives a good bias-variance trade-off without being too slow.
# Random Forest is not sensitive to this the way XGBoost is — more trees
# never hurts accuracy, they just take longer.
N_ESTIMATORS = 200

# Number of CPU cores to use. -1 = all available cores.
# Random Forest parallelises perfectly across trees so this matters more
# than it does for XGBoost.
N_JOBS = -1

# Minimum direction accuracy for a stock to appear in BUY or SELL rankings.
# 50% is the break-even point — below this the model is worse than a coin flip.
MIN_DIRECTION_ACCURACY = 50.0

# Minimum R² for a stock to appear in rankings
MIN_R2 = 0.5
# ────────────────────────────────────────────────────────────────────────────────

# ── RANDOM FOREST HYPERPARAMETERS ────────────────────────────────────────────────
# These are fixed params chosen for financial time series on NSE data.
# Random Forest is robust to hyperparameter choice — unlike XGBoost it does
# not require fallback tuning because bagging inherently reduces variance.
#
# max_features='sqrt'  — each tree sees sqrt(63) ≈ 8 features per split.
#                        This decorrelates the trees, which is the core
#                        mechanism that makes Random Forest better than
#                        a single decision tree.
# max_depth=20         — deep enough to capture non-linear patterns in
#                        technical indicators without memorising noise.
# min_samples_leaf=5   — each leaf must contain at least 5 samples,
#                        preventing the model from overfitting to single
#                        data points in volatile periods.
RF_PARAMS = {
    'n_estimators'   : N_ESTIMATORS,
    'max_features'   : 'sqrt',
    'max_depth'      : 20,
    'min_samples_leaf': 5,
    'n_jobs'         : N_JOBS,
    'random_state'   : 42,
}
# ────────────────────────────────────────────────────────────────────────────────


def sanitize_column_names(df):
    df_clean = df.copy()
    new_names = [
        col if col == 'SYMBOL' else re.sub(r'[\[\]<]', '_', col)
        for col in df_clean.columns
    ]
    df_clean.columns = new_names
    return df_clean


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
    for kw in non_equity_keywords:
        if kw in symbol.upper():
            return False
    return True


def load_training_data(filepath=DATA_FILEPATH):
    return preprocess(filepath)


def rf_train(df, symbol_name, save_artifacts=True, verbose=True):
    """
    Trains a Random Forest model for a single equity symbol and returns a
    result dict containing both BUY and SELL signal information for ranking.

    Random Forest differs from XGBoost in one key way: it builds many
    decision trees independently (bagging) and averages their predictions,
    whereas XGBoost builds trees sequentially, each correcting the errors
    of the previous one (boosting). This makes Random Forest more stable
    and less prone to overfitting on noisy financial data, but typically
    less accurate than a well-tuned XGBoost on the same dataset.

    No fallback tuning is needed — Random Forest is robust by design.
    """
    df = sanitize_column_names(df)
    scaler = MinMaxScaler()

    symbol_cols = [col for col in df.columns if col.startswith('SYMBOL_')]
    df.drop(columns=symbol_cols, inplace=True)

    df = df[df['CLOSE_PRICE'] > 0]
    df = df[df['PREV_CLOSE'] > 0]
    df.dropna(subset=['CLOSE_PRICE'], inplace=True)

    def safe_mape(actual, predicted):
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    # ── FEATURE ENGINEERING ─────────────────────────────────────────────────────
    # Identical to xgb_model.py — ensures a fair comparison between models.
    # The only difference between XGBoost, Random Forest and LSTM results
    # should be the algorithm, not the input features.

    # a. Lag features
    for lag in [1, 3, 5, 10]:
        df[f'CLOSE_lag_{lag}'] = df['CLOSE_PRICE'].shift(lag)

    # b. Technical indicators
    # RSI — momentum oscillator (0-100), window=14 days standard
    df['RSI_14'] = ta.momentum.RSIIndicator(df['CLOSE_PRICE'], window=14).rsi()

    # MACD — trend + momentum indicator
    macd_ind          = ta.trend.MACD(df['CLOSE_PRICE'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd_ind.macd()
    df['MACD_Signal'] = macd_ind.macd_signal()
    df['MACD_Hist']   = macd_ind.macd_diff()

    # Bollinger Bands — volatility bands around 20-day SMA
    bb                = ta.volatility.BollingerBands(df['CLOSE_PRICE'], window=20, window_dev=2)
    df['BB_Upper']    = bb.bollinger_hband()
    df['BB_Middle']   = bb.bollinger_mavg()
    df['BB_Lower']    = bb.bollinger_lband()
    df['BB_Width']    = bb.bollinger_wband()
    df['BB_Position'] = bb.bollinger_pband()   # 0 = at lower band, 1 = at upper band

    # EMA — Exponential Moving Averages (trend direction)
    df['EMA_9']          = ta.trend.EMAIndicator(df['CLOSE_PRICE'], window=9).ema_indicator()
    df['EMA_21']         = ta.trend.EMAIndicator(df['CLOSE_PRICE'], window=21).ema_indicator()
    df['EMA_50']         = ta.trend.EMAIndicator(df['CLOSE_PRICE'], window=50).ema_indicator()
    df['EMA_9_21_cross'] = df['EMA_9'] - df['EMA_21']

    # SMA — Simple Moving Averages (trend baseline)
    df['SMA_10']         = ta.trend.SMAIndicator(df['CLOSE_PRICE'], window=10).sma_indicator()
    df['SMA_20']         = ta.trend.SMAIndicator(df['CLOSE_PRICE'], window=20).sma_indicator()
    df['SMA_50']         = ta.trend.SMAIndicator(df['CLOSE_PRICE'], window=50).sma_indicator()
    df['Price_vs_SMA20'] = df['CLOSE_PRICE'] / df['SMA_20']

    # c. Date features
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df['TRANS_DATE']) if 'TRANS_DATE' in df.columns else pd.Series(pd.to_datetime(df.index))
    df['Day_of_Week']    = dates.dt.dayofweek
    df['Month']          = dates.dt.month
    df['Quarter']        = dates.dt.quarter
    df['Day_of_Month']   = dates.dt.day
    df['Week_of_Year']   = dates.dt.isocalendar().week.astype(int)
    df['Year']           = dates.dt.year
    df['Is_Monday']      = (df['Day_of_Week'] == 0).astype(int)
    df['Is_Friday']      = (df['Day_of_Week'] == 4).astype(int)
    df['Is_Month_Start'] = dates.dt.is_month_start.astype(int)
    df['Is_Month_End']   = dates.dt.is_month_end.astype(int)
    df['Is_Quarter_End'] = dates.dt.is_quarter_end.astype(int)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOW_sin']   = np.sin(2 * np.pi * df['Day_of_Week'] / 5)
    df['DOW_cos']   = np.cos(2 * np.pi * df['Day_of_Week'] / 5)

    # d. Rolling statistics
    # min_periods=1 ensures rolling works correctly on a datetime-indexed Series
    # in pandas 2.0 — without it, integer window on datetime index returns all NaN
    for w in [7, 14, 30]:
        df[f'Rolling_Mean_{w}']   = df['CLOSE_PRICE'].rolling(window=w, min_periods=1).mean()
        df[f'Rolling_Std_{w}']    = df['CLOSE_PRICE'].rolling(window=w, min_periods=1).std()
        df[f'Rolling_Min_{w}']    = df['CLOSE_PRICE'].rolling(window=w, min_periods=1).min()
        df[f'Rolling_Max_{w}']    = df['CLOSE_PRICE'].rolling(window=w, min_periods=1).max()
        df[f'Rolling_Return_{w}'] = df['CLOSE_PRICE'].pct_change(periods=w)

    # .values used to avoid pandas 2.0 Timestamp vs int index alignment issue
    df['Price_Position_7']  = (df['CLOSE_PRICE'].values - df['Rolling_Min_7'].values)  / (df['Rolling_Max_7'].values  - df['Rolling_Min_7'].values)
    df['Price_Position_30'] = (df['CLOSE_PRICE'].values - df['Rolling_Min_30'].values) / (df['Rolling_Max_30'].values - df['Rolling_Min_30'].values)
    df['Volatility_Ratio']  = df['Rolling_Std_7'] / df['Rolling_Std_30']

    df['Target'] = df['CLOSE_PRICE'].shift(-TARGET_HORIZON) / df['CLOSE_PRICE'] - 1
    df['Target_Future_Price'] = df['CLOSE_PRICE'].shift(-TARGET_HORIZON)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=['Target', 'Target_Future_Price', 'CLOSE_PRICE'])
    y = df['Target']

    if 'TRANS_DATE' in X.columns:
        X = X.drop(columns=['TRANS_DATE'])
    if 'SYMBOL' in X.columns:
        X = X.drop(columns=['SYMBOL'])

    X = X.astype('float64')

    # ── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
    split_idx  = int(len(df) * 0.80)
    X_train    = X.iloc[:split_idx]
    X_test     = X.iloc[split_idx:]
    y_train    = y.iloc[:split_idx]
    y_test     = y.iloc[split_idx:]

    if verbose:
        print(f"  Training samples : {len(X_train)} | Testing samples : {len(X_test)}")

    # ── SCALING ──────────────────────────────────────────────────────────────────
    # Note: Random Forest is not sensitive to feature scaling the way neural
    # networks are — it makes decisions based on split thresholds, not distances.
    # We scale anyway to keep the pipeline identical to XGBoost and LSTM,
    # which ensures any performance difference is due to the algorithm alone.
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # ── MODEL TRAINING ───────────────────────────────────────────────────────────
    # No baseline + fallback tuning needed — Random Forest does not require it.
    # Bagging (the core mechanism) already controls variance without tuning,
    # and the fixed RF_PARAMS above work reliably across all NSE symbols.
    if verbose:
        print(f"  Training Random Forest ({N_ESTIMATORS} trees)...")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train_scaled, y_train)
    if save_artifacts:
        joblib.dump(model, f'rf_{symbol_name}_model.pkl')

    # ── EVALUATION ───────────────────────────────────────────────────────────────
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae  = mean_absolute_error(y_test, predictions)
    mape = safe_mape(y_test.values, predictions)
    r2   = r2_score(y_test, predictions)

    if verbose:
        print("=" * 40)
        print(f"  RMSE  : {rmse:.4f}")
        print(f"  MAE   : {mae:.4f}")
        print(f"  MAPE  : {mape:.2f}%")
        print(f"  R²    : {r2:.4f}")
        print("=" * 40)

    # ── VISUALISATIONS ───────────────────────────────────────────────────────────
    if save_artifacts:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5))
        plt.plot(y_test.index, y_test.values, label='Actual Forward Return', color='blue')
        plt.plot(y_test.index, predictions, label='Predicted Forward Return', color='orange', linestyle='--')
        plt.title(f'Random Forest – Predicted vs Actual {TARGET_HORIZON}-Day Return ({symbol_name})')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'rf_predicted_vs_actual_{symbol_name}.png', dpi=150)
        plt.close()

        # Feature importance — Random Forest uses mean decrease in impurity (Gini),
        # which is different from XGBoost's gain-based importance but equally valid
        feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
        feat_imp = feat_imp.sort_values(ascending=True).tail(20)
        fig, ax  = plt.subplots(figsize=(10, 8))
        feat_imp.plot(kind='barh', ax=ax, color='steelblue', alpha=0.85)
        ax.set_title(f'Top 20 Most Important Features — Random Forest ({symbol_name})')
        ax.set_xlabel('Mean Decrease in Impurity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'rf_feature_importance_{symbol_name}.png', dpi=150)
        plt.close()

        residuals = y_test.values - predictions
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(y_test.index, residuals, color='red', alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.title('Residuals Over Time')
        plt.xlabel('Date')
        plt.ylabel('Error (₦)')
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=40, color='steelblue', edgecolor='white')
        plt.title('Residual Distribution')
        plt.xlabel('Error (₦)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'rf_residuals_{symbol_name}.png', dpi=150)
        plt.close()

    # ── BUY / SELL SIGNALS ───────────────────────────────────────────────────────
    signal_df = pd.DataFrame({
        'Actual'   : y_test.values,
        'Predicted': predictions
    }, index=y_test.index)

    signal_df['Signal'] = np.where(signal_df['Predicted'] > 0, 'BUY', 'SELL')
    signal_df['Actual_Direction'] = np.sign(signal_df['Actual'])
    signal_df['Predicted_Direction'] = np.sign(signal_df['Predicted'])
    direction_accuracy = (
        signal_df['Actual_Direction'] == signal_df['Predicted_Direction']
    ).mean() * 100

    if verbose:
        print(f"  Direction Accuracy: {direction_accuracy:.2f}%")

    # ── RESULT DICT ──────────────────────────────────────────────────────────────
    latest_actual_price = df['CLOSE_PRICE'].iloc[-1]
    predicted_return = float(predictions[-1])
    latest_predicted_price = latest_actual_price * (1 + predicted_return)
    signal = 'BUY' if predicted_return > 0 else 'SELL'

    # Confidence Score: R² × |predicted_return| × (direction_accuracy / 100)
    # Identical formula to XGBoost and LSTM for fair three-way comparison.
    confidence_score = max(r2, 0) * abs(predicted_return) * (direction_accuracy / 100)

    return {
        'Symbol'              : symbol_name,
        'Last_Close (₦)'     : latest_actual_price,
        'Predicted_Price'     : round(latest_predicted_price, 2),
        'Predicted_Return (%)': round(predicted_return * 100, 2),
        'Signal'              : signal,
        'R2'                  : round(r2, 4),
        'RMSE'                : round(rmse, 4),
        'MAPE'                : round(mape, 2),
        'Direction_Acc (%)'   : round(direction_accuracy, 2),
        'Confidence_Score'    : round(confidence_score, 4),
    }


def rank_investments(results):
    """
    Splits all results into BUY and SELL ranked tables.

    BUY ranking  — stocks the model expects to be undervalued (predicted > prev close)
    SELL ranking — stocks the model expects to be overvalued  (predicted < prev close)

    Both sides apply the same quality filters:
      - R²                ≥ MIN_R2                (model must have reasonable fit)
      - Direction_Acc (%) ≥ MIN_DIRECTION_ACCURACY (model must be directionally reliable)

    Sorted by Confidence_Score descending so the strongest, most reliable
    signals appear at the top of each section.
    """
    df_all = pd.DataFrame(results)

    # Apply shared quality filters — same criteria for both BUY and SELL
    qualified = df_all[
        (df_all['R2']               >= MIN_R2) &
        (df_all['Direction_Acc (%)'] >= MIN_DIRECTION_ACCURACY)
    ].copy()

    # ── BUY ranking ─────────────────────────────────────────────────────────────
    # Stocks where predicted_return > 0 — model thinks price is currently below
    # where historical trends suggest it should be → potential upside
    buy_df = qualified[qualified['Signal'] == 'BUY'].copy()
    buy_df = buy_df.sort_values('Confidence_Score', ascending=False).reset_index(drop=True)

    # ── SELL ranking ────────────────────────────────────────────────────────────
    # Stocks where predicted_return < 0 — model thinks price is currently above
    # where historical trends suggest it should be → potential downside / overvalued
    # Sorted by Confidence_Score descending so the most overvalued stocks
    # (largest absolute negative return + reliable model) appear first
    sell_df = qualified[qualified['Signal'] == 'SELL'].copy()
    sell_df = sell_df.sort_values('Confidence_Score', ascending=False).reset_index(drop=True)

    return buy_df, sell_df


def save_combined_csv(buy_df, sell_df, filepath='rf_investment_recommendations.csv'):
    """
    Saves BUY and SELL rankings to a single CSV with clearly labelled sections.

    Structure:
        === BUY RECOMMENDATIONS ===
        <buy table rows>
        (blank row)
        === SELL RECOMMENDATIONS ===
        <sell table rows>

    This makes the file human-readable while keeping everything in one place,
    which is convenient for the portfolio optimisation module to consume.
    """
    with open(filepath, 'w') as f:
        f.write('=== BUY RECOMMENDATIONS ===\n')
        buy_df.to_csv(f, index=False)
        f.write('\n')
        f.write('=== SELL RECOMMENDATIONS ===\n')
        sell_df.to_csv(f, index=False)

    print(f"✅ Combined recommendations saved to {filepath}")


def plot_combined_recommendations(buy_df, sell_df, top_n=10):
    """
    Plots BUY and SELL recommendations side by side on a single chart.

    BUY  bars → green  (positive predicted return, ascending from bottom)
    SELL bars → red    (negative predicted return, shown as absolute value)

    Both sides are sorted so the strongest signal is closest to the centre,
    making it easy to visually compare the best BUY vs best SELL candidates.

    For portfolio optimisation this chart immediately shows:
    - Which stocks to overweight  (green, right side)
    - Which stocks to underweight (red, left side)
    """
    import matplotlib.pyplot as plt

    top_buy  = buy_df.head(top_n).copy()
    top_sell = sell_df.head(top_n).copy()

    # Sort so strongest signal is closest to the centre of the chart
    top_buy  = top_buy.sort_values('Predicted_Return (%)', ascending=True)
    top_sell = top_sell.sort_values('Predicted_Return (%)', ascending=False)  # most negative first

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, max(len(top_buy), len(top_sell)) * 0.6)))
    fig.suptitle('Investment Recommendations (Random Forest)', fontsize=14, fontweight='bold', y=1.01)

    # ── LEFT PANEL: SELL recommendations ────────────────────────────────────────
    ax_sell = axes[0]
    if not top_sell.empty:
        # Use absolute value of predicted return for bar length — negative returns
        # are shown as positive bar length extending left, coloured red
        sell_returns = top_sell['Predicted_Return (%)'].abs()
        bars = ax_sell.barh(top_sell['Symbol'], sell_returns, color='#d32f2f', alpha=0.85)
        ax_sell.set_xlabel('Predicted Return — Absolute (%)')
        ax_sell.set_title(f'⬇  Top {len(top_sell)} SELL / AVOID', color='#d32f2f', fontweight='bold')
        ax_sell.invert_xaxis()  # flip so bars grow left (visually away from centre)
        for bar, val in zip(bars, top_sell['Predicted_Return (%)']):
            ax_sell.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', ha='right', fontsize=9, color='#d32f2f'
            )
    else:
        ax_sell.text(0.5, 0.5, 'No SELL signals\nmet quality filters',
                     ha='center', va='center', transform=ax_sell.transAxes, fontsize=11)
        ax_sell.set_title('⬇  SELL / AVOID', color='#d32f2f', fontweight='bold')

    # ── RIGHT PANEL: BUY recommendations ────────────────────────────────────────
    ax_buy = axes[1]
    if not top_buy.empty:
        bars = ax_buy.barh(top_buy['Symbol'], top_buy['Predicted_Return (%)'],
                           color='#2e7d32', alpha=0.85)
        ax_buy.set_xlabel('Predicted Return (%)')
        ax_buy.set_title(f'⬆  Top {len(top_buy)} BUY', color='#2e7d32', fontweight='bold')
        for bar, val in zip(bars, top_buy['Predicted_Return (%)']):
            ax_buy.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=9, color='#2e7d32'
            )
    else:
        ax_buy.text(0.5, 0.5, 'No BUY signals\nmet quality filters',
                    ha='center', va='center', transform=ax_buy.transAxes, fontsize=11)
        ax_buy.set_title('⬆  BUY', color='#2e7d32', fontweight='bold')

    # Style both panels consistently
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    plt.savefig('rf_investment_recommendations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved to rf_investment_recommendations.png")


if __name__ == "__main__":
    df_ = load_training_data()
    symbol_cols  = [col for col in df_.columns if col.startswith('SYMBOL_')]
    df_['SYMBOL'] = df_[symbol_cols].idxmax(axis=1).str.replace('SYMBOL_', '', regex=False)

    all_symbols    = df_['SYMBOL'].unique()
    equity_symbols = [s for s in all_symbols if is_equity(s)]

    symbol_counts  = df_[df_['SYMBOL'].isin(equity_symbols)].groupby('SYMBOL').size()
    ranked_symbols = symbol_counts.sort_values(ascending=False).index.tolist()
    valid_symbols  = [s for s in ranked_symbols if symbol_counts[s] >= MIN_ROWS]

    print(f"Total symbols      : {len(all_symbols)}")
    print(f"Equity symbols     : {len(equity_symbols)}  (after filtering bonds/ETFs/REITs)")
    print(f"Symbols ≥ {MIN_ROWS} rows : {len(valid_symbols)}  (after removing thin data symbols)")
    print(f"Model              : Random Forest ({N_ESTIMATORS} trees, max_depth={RF_PARAMS['max_depth']})\n")

    all_results = []
    trained     = 0

    for symbol in valid_symbols:
        symbol_df = df_[df_['SYMBOL'] == symbol].copy()
        trained  += 1
        print(f"\n[{trained}/{len(valid_symbols)}] {symbol}")

        try:
            result = rf_train(symbol_df, symbol_name=symbol)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping {symbol} due to error: {e}")
            continue

    # ── RANKING & OUTPUT ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("        INVESTMENT RECOMMENDATION RANKING")
    print(f"{'=' * 60}")

    if all_results:
        buy_df, sell_df = rank_investments(all_results)

        # ── Print BUY table ──────────────────────────────────────────────────────
        print(f"\n🟢 BUY RECOMMENDATIONS  ({len(buy_df)} stocks)")
        print("   Stocks the model predicts are currently undervalued")
        print("   → Overweight in portfolio\n")
        if buy_df.empty:
            print("   No BUY signals met quality filters.")
        else:
            print(buy_df.to_string(index=False))

        # ── Print SELL table ─────────────────────────────────────────────────────
        print(f"\n🔴 SELL RECOMMENDATIONS  ({len(sell_df)} stocks)")
        print("   Stocks the model predicts are currently overvalued")
        print("   → Underweight or avoid in portfolio\n")
        if sell_df.empty:
            print("   No SELL signals met quality filters.")
        else:
            print(sell_df.to_string(index=False))

        # ── Save outputs ─────────────────────────────────────────────────────────
        save_combined_csv(buy_df, sell_df)
        plot_combined_recommendations(buy_df, sell_df, top_n=10)

    else:
        print("⚠️  No results collected — check that symbols have sufficient data.")

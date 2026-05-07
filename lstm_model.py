import re
import os
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# On some macOS/PyTorch CPU builds, the default BLAS/OpenMP thread pool can
# stall during recurrent training. Conservative defaults keep merge runs moving.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import preprocess

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

DATA_FILEPATH = 'PRICE_LIST.csv'
TARGET_HORIZON = 20

# ── CONFIG ──────────────────────────────────────────────────────────────────────
MIN_ROWS          = 500
LOOKBACK          = 60     # number of past days each prediction uses
EPOCHS            = 30     # max training epochs
BATCH_SIZE        = 32
LEARNING_RATE     = 0.001
PATIENCE          = 5      # early stopping patience
MIN_R2            = 0.5
MIN_DIRECTION_ACC = 50.0
FAST_FEATURES = [
    'PREV_CLOSE',
    'OPEN_PRICE',
    'HIGH_PRICE',
    'LOW_PRICE',
    'CLOSE_lag_1',
    'CLOSE_lag_3',
    'CLOSE_lag_5',
    'CLOSE_lag_10',
    'RSI_14',
    'MACD',
    'MACD_Signal',
    'EMA_9',
    'EMA_21',
    'EMA_9_21_cross',
    'SMA_20',
    'Price_vs_SMA20',
    'Rolling_Return_7',
    'Rolling_Return_14',
    'Rolling_Return_30',
    'Volatility_Ratio',
    'Month_sin',
    'Month_cos',
    'DOW_sin',
    'DOW_cos',
]

# Use GPU if available, otherwise CPU
DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available()
                       else 'cpu')
if DEVICE.type == "cpu":
    try:
        torch.set_num_threads(int(os.getenv("LSTM_TORCH_THREADS", "1")))
        torch.set_num_interop_threads(int(os.getenv("LSTM_TORCH_INTEROP_THREADS", "1")))
    except RuntimeError:
        pass
# ────────────────────────────────────────────────────────────────────────────────


def sanitize_column_names(df):
    df_clean  = df.copy()
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


def add_technical_indicators(df):
    """
    Identical feature engineering to xgb_model.py.
    Keeping features identical ensures XGBoost vs LSTM is a fair comparison —
    the only difference is the learning algorithm, not the input information.
    """
    # Reset to clean sequential index before all indicator calculations —
    # the incoming index contains non-sequential integers which cause index
    # misalignment when ta indicators are assigned back to df after reset_index.
    df = df.reset_index(drop=True)
    close = df['CLOSE_PRICE']

    # a. Lag features
    for lag in [1, 3, 5, 10]:
        df[f'CLOSE_lag_{lag}'] = close.shift(lag)

    # b. RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # c. MACD
    macd_ind          = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd_ind.macd()
    df['MACD_Signal'] = macd_ind.macd_signal()
    df['MACD_Hist']   = macd_ind.macd_diff()

    # d. Bollinger Bands
    bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_Upper']    = bb.bollinger_hband()
    df['BB_Middle']   = bb.bollinger_mavg()
    df['BB_Lower']    = bb.bollinger_lband()
    df['BB_Width']    = bb.bollinger_wband()
    df['BB_Position'] = bb.bollinger_pband()

    # e. EMA
    df['EMA_9']          = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df['EMA_21']         = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['EMA_50']         = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df['EMA_9_21_cross'] = df['EMA_9'] - df['EMA_21']

    # f. SMA
    df['SMA_10']         = ta.trend.SMAIndicator(close, window=10).sma_indicator()
    df['SMA_20']         = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df['SMA_50']         = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df['Price_vs_SMA20'] = close / df['SMA_20']

    # h. Rolling statistics — BEFORE datetime conversion so close and df
    # share the same integer index throughout. min_periods=1 avoids NaN
    # at the start of short series.
    for w in [7, 14, 30]:
        df[f'Rolling_Mean_{w}']   = close.rolling(window=w, min_periods=1).mean()
        df[f'Rolling_Std_{w}']    = close.rolling(window=w, min_periods=1).std()
        df[f'Rolling_Min_{w}']    = close.rolling(window=w, min_periods=1).min()
        df[f'Rolling_Max_{w}']    = close.rolling(window=w, min_periods=1).max()
        df[f'Rolling_Return_{w}'] = close.pct_change(periods=w)

    # .values prevents pandas 2.0 index alignment issues
    df['Price_Position_7']  = (close.values - df['Rolling_Min_7'].values)  / (df['Rolling_Max_7'].values  - df['Rolling_Min_7'].values)
    df['Price_Position_30'] = (close.values - df['Rolling_Min_30'].values) / (df['Rolling_Max_30'].values - df['Rolling_Min_30'].values)
    df['Volatility_Ratio']  = df['Rolling_Std_7'] / df['Rolling_Std_30']

    # g. Date features — AFTER rolling to avoid datetime index breaking alignment.
    # Uses TRANS_DATE column for real trading dates where available.
    if 'TRANS_DATE' in df.columns:
        dates = pd.to_datetime(df['TRANS_DATE'])
    else:
        dates = pd.to_datetime(df.index)
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
    df['Month_sin']      = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos']      = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOW_sin']        = np.sin(2 * np.pi * df['Day_of_Week'] / 5)
    df['DOW_cos']        = np.cos(2 * np.pi * df['Day_of_Week'] / 5)

    return df


def build_sequences(X_scaled, y, lookback=LOOKBACK):
    """
    Reshapes flat 2D data into 3D sequences for LSTM input.
    Shape: (samples, timesteps, features)
    Each prediction uses the previous `lookback` days of features.
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback: i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def to_torch_tensor(array):
    # Prefer zero-copy NumPy conversion. The list fallback is kept for older
    # environments where PyTorch was built against an incompatible NumPy API.
    try:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    except Exception:
        return torch.tensor(array.tolist(), dtype=torch.float32)


# ── PYTORCH LSTM MODEL ───────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    Single-layer LSTM model for time series price prediction.

    Architecture:
      LSTM(64, 1 layer) → Dropout(0.2) → Linear(1)

    Single layer is faster and sufficient for daily NSE stock price patterns.
    Can be increased to num_layers=2 if accuracy needs improvement.

    Parameters
    ----------
    input_size  : number of input features per timestep
    hidden_size : number of LSTM hidden units (default 64)
    num_layers  : number of stacked LSTM layers (default 1)
    dropout     : dropout rate between LSTM layers (default 0.2)
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # batch_first=True → input shape is (batch, seq_len, features)
        # which matches our (samples, LOOKBACK, n_features) sequence format
        # dropout only applies between layers — set to 0 when num_layers=1
        # to avoid a silent PyTorch warning
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)  # single output: predicted price

    def forward(self, x):
        # Initialise hidden and cell states to zero for each batch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # out shape: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Take only the last timestep's output — this summarises the full sequence
        out = self.dropout(out[:, -1, :])   # shape: (batch, hidden_size)
        out = self.fc(out)                  # shape: (batch, 1)
        return out


def train_lstm(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    patience=PATIENCE,
    verbose=False,
    progress_label=None,
):
    """
    Trains the LSTM model with early stopping.

    Uses MSE loss and Adam optimiser — standard choices for regression.
    Early stopping halts training if validation loss doesn't improve
    for `patience` consecutive epochs, then restores the best weights.

    Parameters
    ----------
    model        : LSTMModel instance
    train_loader : DataLoader for training sequences
    val_loader   : DataLoader for validation sequences
    epochs       : maximum number of training epochs
    patience     : early stopping patience

    Returns
    -------
    train_losses : list of training loss per epoch
    val_losses   : list of validation loss per epoch
    best_epoch   : epoch at which best validation loss was achieved
    """
    criterion  = nn.MSELoss()
    optimiser  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss  = float('inf')
    best_weights   = None
    patience_count = 0
    train_losses   = []
    val_losses     = []

    for epoch in range(epochs):
        # ── Training pass ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimiser.zero_grad()
            preds = model(X_batch).squeeze()
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation pass ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                preds    = model(X_batch).squeeze()
                val_loss += criterion(preds, y_batch).item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose:
            label = f" {progress_label}" if progress_label else ""
            print(
                f"  LSTM{label} epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                flush=True,
            )

        # ── Early stopping ───────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch     = epoch + 1
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return train_losses, val_losses, best_epoch


def lstm_train(
    df,
    symbol_name,
    save_artifacts=True,
    verbose=True,
    epochs_override=None,
    lookback_override=None,
    batch_size_override=None,
    patience_override=None,
    fast_mode=False,
    hidden_size_override=None,
):
    """
    Full LSTM training pipeline for a single equity symbol.

    Steps:
    1. Feature engineering (same as xgb_model.py for fair comparison)
    2. 80/20 chronological train/test split
    3. MinMaxScaler — fit on train only
    4. Sequence construction (LOOKBACK=60 days)
    5. PyTorch DataLoader construction
    6. LSTM training with early stopping
    7. Evaluation: RMSE, MAE, MAPE, R², Direction Accuracy
    8. BUY/SELL signal generation
    9. Visualisations saved to disk
    10. Returns result dict for comparison report
    """
    df       = sanitize_column_names(df)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    lookback = lookback_override if lookback_override is not None else LOOKBACK
    epochs = epochs_override if epochs_override is not None else (4 if fast_mode else EPOCHS)
    batch_size = batch_size_override if batch_size_override is not None else (256 if fast_mode else BATCH_SIZE)
    patience = patience_override if patience_override is not None else (2 if fast_mode else PATIENCE)
    hidden_size = hidden_size_override if hidden_size_override is not None else (16 if fast_mode else 64)

    symbol_cols = [col for col in df.columns if col.startswith('SYMBOL_')]
    df.drop(columns=symbol_cols, inplace=True)

    df = df[df['CLOSE_PRICE'] > 0]
    df = df[df['PREV_CLOSE']  > 0]
    df.dropna(subset=['CLOSE_PRICE'], inplace=True)

    def safe_mape(actual, predicted):
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    df = add_technical_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['Target_Return'] = df['CLOSE_PRICE'].shift(-TARGET_HORIZON) / df['CLOSE_PRICE'] - 1
    df['Target_Future_Price'] = df['CLOSE_PRICE'].shift(-TARGET_HORIZON)
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns
                    if c not in ['CLOSE_PRICE', 'TRANS_DATE', 'SYMBOL', 'Target_Return', 'Target_Future_Price']]
    if fast_mode:
        feature_cols = [col for col in FAST_FEATURES if col in feature_cols]
        if verbose:
            print(f"  Fast LSTM mode: using {len(feature_cols)} compact features")

    if not feature_cols:
        raise ValueError("No usable LSTM features after preprocessing.")
    X = df[feature_cols].astype('float64').values
    y = df['Target_Return'].values.reshape(-1, 1)

    # ── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if verbose:
        print(f"  Training samples : {len(X_train)} | Testing samples : {len(X_test)}")

    # ── SCALING ──────────────────────────────────────────────────────────────────
    # Fit scalers on training data only — prevents data leakage from test period.
    # Separate scalers for X and y so y can be inverse-transformed back to ₦.
    X_train_scaled = scaler_X.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler_X.transform(X_test).astype(np.float32)
    y_train_scaled = scaler_y.fit_transform(y_train).astype(np.float32)
    y_test_scaled  = scaler_y.transform(y_test).astype(np.float32)

    # ── SEQUENCE CONSTRUCTION ────────────────────────────────────────────────────
    X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train_scaled.flatten(), lookback=lookback)
    X_test_seq,  y_test_seq  = build_sequences(X_test_scaled,  y_test_scaled.flatten(), lookback=lookback)

    if verbose:
        print(f"  Sequence shape   : {X_train_seq.shape}  (samples × lookback × features)")

    if len(X_train_seq) < 10 or len(X_test_seq) < 5:
        raise ValueError(f"Not enough sequences after applying LOOKBACK={lookback}.")

    # ── PYTORCH DATALOADERS ──────────────────────────────────────────────────────
    # Split last 10% of training sequences for validation (mirrors Keras val_split=0.1)
    val_split    = int(len(X_train_seq) * 0.9)
    X_tr, X_val  = X_train_seq[:val_split], X_train_seq[val_split:]
    y_tr, y_val  = y_train_seq[:val_split], y_train_seq[val_split:]

    train_dataset = TensorDataset(to_torch_tensor(X_tr), to_torch_tensor(y_tr))
    val_dataset   = TensorDataset(to_torch_tensor(X_val), to_torch_tensor(y_val))
    test_dataset  = TensorDataset(to_torch_tensor(X_test_seq), to_torch_tensor(y_test_seq))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ── MODEL TRAINING ───────────────────────────────────────────────────────────
    n_features  = X_train_seq.shape[2]
    model       = LSTMModel(input_size=n_features, hidden_size=hidden_size, dropout=0.1 if fast_mode else 0.2).to(DEVICE)
    train_losses, val_losses, best_epoch = train_lstm(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        verbose=verbose,
        progress_label=symbol_name,
    )

    if verbose:
        print(f"  Training stopped at epoch {best_epoch}/{epochs}")

    # ── PREDICTIONS ──────────────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds   = model(X_batch).squeeze().detach().cpu().reshape(-1)
            all_preds.extend(preds.tolist())

    predictions_scaled = np.array(all_preds, dtype=np.float32).reshape(-1, 1)
    predictions        = scaler_y.inverse_transform(predictions_scaled).flatten()
    y_test_actual      = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    test_index         = df.index[split_idx + lookback:]

    # ── EVALUATION ───────────────────────────────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae  = mean_absolute_error(y_test_actual, predictions)
    mape = safe_mape(y_test_actual, predictions)
    r2   = r2_score(y_test_actual, predictions)

    if verbose:
        print("=" * 40)
        print(f"  RMSE  : {rmse:.4f}")
        print(f"  MAE   : {mae:.4f}")
        print(f"  MAPE  : {mape:.2f}%")
        print(f"  R²    : {r2:.4f}")
        print("=" * 40)

    # ── VISUALISATIONS ───────────────────────────────────────────────────────────
    # Predicted vs Actual
    if save_artifacts:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5))
        plt.plot(test_index, y_test_actual, label='Actual Forward Return',    color='blue')
        plt.plot(test_index, predictions,   label='Predicted Forward Return', color='orange', linestyle='--')
        plt.title(f'LSTM – Predicted vs Actual {TARGET_HORIZON}-Day Return ({symbol_name})')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'lstm_predicted_vs_actual_{symbol_name}.png', dpi=150)
        plt.close()

        # Training loss curve
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses,   label='Val Loss',   color='orange')
        plt.axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.5, label='Best Epoch')
        plt.title(f'LSTM Training Loss ({symbol_name})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'lstm_loss_{symbol_name}.png', dpi=150)
        plt.close()

        # Residuals
        residuals = y_test_actual - predictions
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(test_index, residuals, color='red', alpha=0.7)
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
        plt.savefig(f'lstm_residuals_{symbol_name}.png', dpi=150)
        plt.close()

        torch.save(model.state_dict(), f'lstm_{symbol_name}_model.pt')

    # ── BUY / SELL SIGNALS ───────────────────────────────────────────────────────
    predicted_direction = np.sign(predictions)
    actual_direction    = np.sign(y_test_actual)
    direction_accuracy  = np.mean(predicted_direction == actual_direction) * 100

    if verbose:
        print(f"  Direction Accuracy: {direction_accuracy:.2f}%")

    latest_actual = df['CLOSE_PRICE'].iloc[-1]
    predicted_return = float(predictions[-1])
    latest_predicted = latest_actual * (1 + predicted_return)
    signal = 'BUY' if predicted_return > 0 else 'SELL'

    # Confidence Score — identical formula to XGBoost for fair comparison
    confidence_score = max(r2, 0) * abs(predicted_return) * (direction_accuracy / 100)

    return {
        'Symbol'              : symbol_name,
        'Last_Close (₦)'     : latest_actual,
        'Predicted_Price'     : round(latest_predicted, 2),
        'Predicted_Return (%)': round(predicted_return * 100, 2),
        'Signal'              : signal,
        'R2'                  : round(r2, 4),
        'RMSE'                : round(rmse, 4),
        'MAPE'                : round(mape, 2),
        'Direction_Acc (%)'   : round(direction_accuracy, 2),
        'Confidence_Score'    : round(confidence_score, 4),
        'Epochs_Run'          : best_epoch
    }


def rank_investments(results):
    """
    Splits LSTM results into BUY and SELL ranked tables.
    Same filters as XGBoost ranking for fair comparison.
    """
    df_all    = pd.DataFrame(results)
    qualified = df_all[
        (df_all['R2']               >= MIN_R2) &
        (df_all['Direction_Acc (%)'] >= MIN_DIRECTION_ACC)
    ].copy()
    buy_df  = qualified[qualified['Signal'] == 'BUY'].sort_values(
        'Confidence_Score', ascending=False).reset_index(drop=True)
    sell_df = qualified[qualified['Signal'] == 'SELL'].sort_values(
        'Confidence_Score', ascending=False).reset_index(drop=True)
    return buy_df, sell_df


def plot_combined_recommendations(buy_df, sell_df, top_n=10):
    """
    Dual-panel BUY/SELL chart — green for BUY, red for SELL.
    Same format as xgb_model.py for easy visual comparison.
    """
    top_buy  = buy_df.head(top_n).sort_values('Predicted_Return (%)', ascending=True)
    top_sell = sell_df.head(top_n).sort_values('Predicted_Return (%)', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, max(len(top_buy), len(top_sell)) * 0.6)))
    fig.suptitle('LSTM Investment Recommendations', fontsize=14, fontweight='bold')

    ax_sell = axes[0]
    if not top_sell.empty:
        bars = ax_sell.barh(top_sell['Symbol'], top_sell['Predicted_Return (%)'].abs(),
                            color='#d32f2f', alpha=0.85)
        ax_sell.set_xlabel('Predicted Return — Absolute (%)')
        ax_sell.set_title('⬇  SELL / AVOID', color='#d32f2f', fontweight='bold')
        ax_sell.invert_xaxis()
        for bar, val in zip(bars, top_sell['Predicted_Return (%)']):
            ax_sell.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{val:.2f}%', va='center', ha='right', fontsize=9, color='#d32f2f')
    else:
        ax_sell.text(0.5, 0.5, 'No SELL signals\nmet quality filters',
                     ha='center', va='center', transform=ax_sell.transAxes)
        ax_sell.set_title('⬇  SELL / AVOID', color='#d32f2f', fontweight='bold')

    ax_buy = axes[1]
    if not top_buy.empty:
        bars = ax_buy.barh(top_buy['Symbol'], top_buy['Predicted_Return (%)'],
                           color='#2e7d32', alpha=0.85)
        ax_buy.set_xlabel('Predicted Return (%)')
        ax_buy.set_title('⬆  BUY', color='#2e7d32', fontweight='bold')
        for bar, val in zip(bars, top_buy['Predicted_Return (%)']):
            ax_buy.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}%', va='center', fontsize=9, color='#2e7d32')
    else:
        ax_buy.text(0.5, 0.5, 'No BUY signals\nmet quality filters',
                    ha='center', va='center', transform=ax_buy.transAxes)
        ax_buy.set_title('⬆  BUY', color='#2e7d32', fontweight='bold')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    plt.savefig('lstm_investment_recommendations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ LSTM chart saved to lstm_investment_recommendations.png")


def build_comparison_report(xgb_results_path, lstm_buy, lstm_sell, lstm_all_results):
    """
    Builds combined XGBoost vs LSTM comparison report.
    Reads XGBoost results from investment_recommendations.csv,
    merges with LSTM results, and produces:
      - combined_recommendations.csv
      - model_comparison.csv
      - comparison_chart.png
    """
    lstm_df         = pd.DataFrame(lstm_all_results)
    xgb_buy_rows    = []
    xgb_sell_rows   = []
    current_section = None

    try:
        with open(xgb_results_path, 'r') as f:
            lines = f.readlines()

        header = None
        for line in lines:
            line = line.strip()
            if line == '=== BUY RECOMMENDATIONS ===':
                current_section = 'BUY';  header = None;  continue
            elif line == '=== SELL RECOMMENDATIONS ===':
                current_section = 'SELL'; header = None;  continue
            elif line == '':
                continue
            if header is None:
                header = line.split(',');  continue
            values = line.split(',')
            if len(values) == len(header):
                row = dict(zip(header, values))
                if current_section == 'BUY':
                    xgb_buy_rows.append(row)
                elif current_section == 'SELL':
                    xgb_sell_rows.append(row)

        xgb_all     = pd.DataFrame(xgb_buy_rows + xgb_sell_rows)
        xgb_compare = xgb_all[['Symbol', 'R2', 'RMSE', 'MAPE',
                                'Direction_Acc (%)', 'Predicted_Return (%)', 'Signal']].copy()
        xgb_compare = xgb_compare.rename(columns={
            'R2': 'XGB_R2', 'RMSE': 'XGB_RMSE', 'MAPE': 'XGB_MAPE',
            'Direction_Acc (%)': 'XGB_Direction_Acc',
            'Predicted_Return (%)': 'XGB_Predicted_Return', 'Signal': 'XGB_Signal'
        })

        lstm_compare = lstm_df[['Symbol', 'R2', 'RMSE', 'MAPE',
                                 'Direction_Acc (%)', 'Predicted_Return (%)', 'Signal']].copy()
        lstm_compare = lstm_compare.rename(columns={
            'R2': 'LSTM_R2', 'RMSE': 'LSTM_RMSE', 'MAPE': 'LSTM_MAPE',
            'Direction_Acc (%)': 'LSTM_Direction_Acc',
            'Predicted_Return (%)': 'LSTM_Predicted_Return', 'Signal': 'LSTM_Signal'
        })

        comparison = pd.merge(xgb_compare, lstm_compare, on='Symbol', how='inner')
        for col in ['XGB_R2', 'XGB_RMSE', 'XGB_MAPE', 'XGB_Direction_Acc',
                    'LSTM_R2', 'LSTM_RMSE', 'LSTM_MAPE', 'LSTM_Direction_Acc']:
            comparison[col] = pd.to_numeric(comparison[col], errors='coerce')

        comparison['Best_Model'] = np.where(
            comparison['LSTM_R2'] > comparison['XGB_R2'], 'LSTM', 'XGBoost'
        )
        xgb_wins  = (comparison['Best_Model'] == 'XGBoost').sum()
        lstm_wins = (comparison['Best_Model'] == 'LSTM').sum()

        comparison.to_csv('model_comparison.csv', index=False)
        print(f"\n✅ Model comparison saved to model_comparison.csv")
        print(f"   XGBoost wins : {xgb_wins} / {len(comparison)} symbols")
        print(f"   LSTM wins    : {lstm_wins} / {len(comparison)} symbols")

        # Combined recommendations CSV
        with open('combined_recommendations.csv', 'w') as f:
            f.write('=== XGBoost BUY RECOMMENDATIONS ===\n')
            pd.DataFrame(xgb_buy_rows).to_csv(f, index=False)
            f.write('\n')
            f.write('=== XGBoost SELL RECOMMENDATIONS ===\n')
            pd.DataFrame(xgb_sell_rows).to_csv(f, index=False)
            f.write('\n')
            f.write('=== LSTM BUY RECOMMENDATIONS ===\n')
            lstm_buy.to_csv(f, index=False)
            f.write('\n')
            f.write('=== LSTM SELL RECOMMENDATIONS ===\n')
            lstm_sell.to_csv(f, index=False)
            f.write('\n')
            f.write('=== SYMBOL-BY-SYMBOL MODEL COMPARISON ===\n')
            comparison.to_csv(f, index=False)
        print(f"✅ Combined recommendations saved to combined_recommendations.csv")

        # Comparison chart
        metrics = {
            'Avg R²'               : (comparison['XGB_R2'].mean(),           comparison['LSTM_R2'].mean()),
            'Avg Direction Acc (%)': (comparison['XGB_Direction_Acc'].mean(), comparison['LSTM_Direction_Acc'].mean()),
            'Avg MAPE (%)'         : (comparison['XGB_MAPE'].mean(),          comparison['LSTM_MAPE'].mean()),
        }
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle('XGBoost vs LSTM — Model Comparison', fontsize=14, fontweight='bold')
        colors = {'XGBoost': '#1565c0', 'LSTM': '#c62828'}
        for ax, (metric, (xgb_val, lstm_val)) in zip(axes, metrics.items()):
            bars = ax.bar(['XGBoost', 'LSTM'], [xgb_val, lstm_val],
                          color=[colors['XGBoost'], colors['LSTM']], alpha=0.85, width=0.5)
            ax.set_title(metric, fontweight='bold')
            ax.set_ylim(0, max(xgb_val, lstm_val) * 1.2)
            for bar, val in zip(bars, [xgb_val, lstm_val]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        winner = 'LSTM' if lstm_wins > xgb_wins else 'XGBoost'
        fig.text(0.5, -0.02,
                 f'Overall winner by R²: {winner}  '
                 f'(XGBoost: {xgb_wins} | LSTM: {lstm_wins})',
                 ha='center', fontsize=11, style='italic')
        plt.tight_layout()
        plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Comparison chart saved to comparison_chart.png")

    except FileNotFoundError:
        print(f"⚠️  XGBoost results not found at '{xgb_results_path}'.")
        print(f"   Run xgb_model.py first, then re-run lstm_model.py.")
        with open('lstm_recommendations.csv', 'w') as f:
            f.write('=== LSTM BUY RECOMMENDATIONS ===\n')
            lstm_buy.to_csv(f, index=False)
            f.write('\n')
            f.write('=== LSTM SELL RECOMMENDATIONS ===\n')
            lstm_sell.to_csv(f, index=False)
        print(f"✅ LSTM-only recommendations saved to lstm_recommendations.csv")


if __name__ == "__main__":
    df_ = load_training_data()
    symbol_cols   = [col for col in df_.columns if col.startswith('SYMBOL_')]
    df_['SYMBOL'] = df_[symbol_cols].idxmax(axis=1).str.replace('SYMBOL_', '', regex=False)

    all_symbols    = df_['SYMBOL'].unique()
    equity_symbols = [s for s in all_symbols if is_equity(s)]
    symbol_counts  = df_[df_['SYMBOL'].isin(equity_symbols)].groupby('SYMBOL').size()
    ranked_symbols = symbol_counts.sort_values(ascending=False).index.tolist()
    valid_symbols  = [s for s in ranked_symbols if symbol_counts[s] >= MIN_ROWS]

    print(f"Total symbols      : {len(all_symbols)}")
    print(f"Equity symbols     : {len(equity_symbols)}")
    print(f"Symbols ≥ {MIN_ROWS} rows : {len(valid_symbols)}")
    print(f"Lookback window    : {LOOKBACK} days")
    print(f"Device             : {DEVICE}\n")

    all_results = []
    trained     = 0

    for symbol in valid_symbols:
        symbol_df = df_[df_['SYMBOL'] == symbol].copy()
        trained  += 1
        print(f"\n[{trained}/{len(valid_symbols)}] {symbol}")
        try:
            result = lstm_train(symbol_df, symbol_name=symbol)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping {symbol} due to error: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("        LSTM INVESTMENT RECOMMENDATION RANKING")
    print(f"{'=' * 60}")

    if all_results:
        buy_df, sell_df = rank_investments(all_results)

        print(f"\n🟢 LSTM BUY RECOMMENDATIONS  ({len(buy_df)} stocks)\n")
        print(buy_df.to_string(index=False) if not buy_df.empty else "   No BUY signals met quality filters.")

        print(f"\n🔴 LSTM SELL RECOMMENDATIONS  ({len(sell_df)} stocks)\n")
        print(sell_df.to_string(index=False) if not sell_df.empty else "   No SELL signals met quality filters.")

        plot_combined_recommendations(buy_df, sell_df)

        print(f"\n{'=' * 60}")
        print("        BUILDING COMPARISON REPORT")
        print(f"{'=' * 60}")
        build_comparison_report(
            xgb_results_path='investment_recommendations.csv',
            lstm_buy=buy_df,
            lstm_sell=sell_df,
            lstm_all_results=all_results
        )
    else:
        print("⚠️  No results collected.")

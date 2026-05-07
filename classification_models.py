import os
import re

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import DEVICE, to_torch_tensor


TARGET_HORIZON = 20
LOOKBACK = 20
BUY_PROB_THRESHOLD = 0.50


def sanitize_column_names(df):
    df_clean = df.copy()
    df_clean.columns = [
        col if col == "SYMBOL" else re.sub(r"[\[\]<]", "_", col)
        for col in df_clean.columns
    ]
    return df_clean


def add_technical_indicators(df):
    df = df.reset_index(drop=True)
    close = df["CLOSE_PRICE"]

    for lag in [1, 3, 5, 10]:
        df[f"CLOSE_lag_{lag}"] = close.shift(lag)

    df["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_ind.macd()
    df["MACD_Signal"] = macd_ind.macd_signal()
    df["MACD_Hist"] = macd_ind.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Position"] = bb.bollinger_pband()

    df["EMA_9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["EMA_21"] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["EMA_9_21_cross"] = df["EMA_9"] - df["EMA_21"]

    df["SMA_10"] = ta.trend.SMAIndicator(close, window=10).sma_indicator()
    df["SMA_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Price_vs_SMA20"] = close / df["SMA_20"]

    for w in [7, 14, 30]:
        df[f"Rolling_Mean_{w}"] = close.rolling(window=w, min_periods=1).mean()
        df[f"Rolling_Std_{w}"] = close.rolling(window=w, min_periods=1).std()
        df[f"Rolling_Min_{w}"] = close.rolling(window=w, min_periods=1).min()
        df[f"Rolling_Max_{w}"] = close.rolling(window=w, min_periods=1).max()
        df[f"Rolling_Return_{w}"] = close.pct_change(periods=w)

    df["Price_Position_7"] = (
        (close.values - df["Rolling_Min_7"].values)
        / (df["Rolling_Max_7"].values - df["Rolling_Min_7"].values)
    )
    df["Price_Position_30"] = (
        (close.values - df["Rolling_Min_30"].values)
        / (df["Rolling_Max_30"].values - df["Rolling_Min_30"].values)
    )
    df["Volatility_Ratio"] = df["Rolling_Std_7"] / df["Rolling_Std_30"]

    dates = pd.to_datetime(df["TRANS_DATE"]) if "TRANS_DATE" in df.columns else pd.to_datetime(df.index)
    df["Day_of_Week"] = dates.dt.dayofweek
    df["Month"] = dates.dt.month
    df["Quarter"] = dates.dt.quarter
    df["Day_of_Month"] = dates.dt.day
    df["Week_of_Year"] = dates.dt.isocalendar().week.astype(int)
    df["Year"] = dates.dt.year
    df["Is_Monday"] = (df["Day_of_Week"] == 0).astype(int)
    df["Is_Friday"] = (df["Day_of_Week"] == 4).astype(int)
    df["Is_Month_Start"] = dates.dt.is_month_start.astype(int)
    df["Is_Month_End"] = dates.dt.is_month_end.astype(int)
    df["Is_Quarter_End"] = dates.dt.is_quarter_end.astype(int)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 5)
    df["DOW_cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 5)
    return df


def prepare_direction_data(df):
    df = sanitize_column_names(df)
    symbol_cols = [col for col in df.columns if col.startswith("SYMBOL_")]
    df = df.drop(columns=symbol_cols)
    df = df[(df["CLOSE_PRICE"] > 0) & (df["PREV_CLOSE"] > 0)].copy()
    df.dropna(subset=["CLOSE_PRICE"], inplace=True)

    df = add_technical_indicators(df)
    df["Target_Return"] = df["CLOSE_PRICE"].shift(-TARGET_HORIZON) / df["CLOSE_PRICE"] - 1
    df["Target_Direction"] = (df["Target_Return"] > 0).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        col for col in df.columns
        if col not in ["CLOSE_PRICE", "TRANS_DATE", "SYMBOL", "Target_Return", "Target_Direction"]
    ]
    X = df[feature_cols].astype("float64")
    y = df["Target_Direction"].astype(int)
    returns = df["Target_Return"].astype(float)
    return df, X, y, returns


def expected_return_from_probability(prob_buy, train_returns):
    magnitude = train_returns.abs().median()
    if pd.isna(magnitude) or magnitude == 0:
        magnitude = train_returns.abs().mean()
    if pd.isna(magnitude):
        magnitude = 0.0
    expected = (prob_buy - 0.5) * 2 * magnitude
    return float(np.clip(expected, -0.75, 0.75))


def make_classification_result(symbol_name, df, y_test, probabilities, train_returns):
    predictions = (probabilities >= BUY_PROB_THRESHOLD).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    precision_buy = precision_score(y_test, predictions, zero_division=0)
    recall_buy = recall_score(y_test, predictions, zero_division=0)
    f1_buy = f1_score(y_test, predictions, zero_division=0)

    latest_prob_buy = float(probabilities[-1])
    predicted_return = expected_return_from_probability(latest_prob_buy, train_returns)
    signal = "BUY" if latest_prob_buy >= BUY_PROB_THRESHOLD else "SELL"
    latest_actual = float(df["CLOSE_PRICE"].iloc[-1])
    latest_predicted = latest_actual * (1 + predicted_return)

    probability_edge = abs(latest_prob_buy - 0.5) * 2
    quality_score = max(balanced - 0.5, 0.0)
    confidence_score = probability_edge * max(balanced, 0.0) * max(f1_buy, 0.0)

    return {
        "Symbol": symbol_name,
        "Last_Close (₦)": latest_actual,
        "Predicted_Price": round(latest_predicted, 2),
        "Predicted_Return (%)": round(predicted_return * 100, 2),
        "Signal": signal,
        "R2": round(quality_score, 4),
        "RMSE": None,
        "MAPE": None,
        "Direction_Acc (%)": round(accuracy * 100, 2),
        "Balanced_Acc (%)": round(balanced * 100, 2),
        "Precision_BUY (%)": round(precision_buy * 100, 2),
        "Recall_BUY (%)": round(recall_buy * 100, 2),
        "F1_BUY (%)": round(f1_buy * 100, 2),
        "Buy_Probability (%)": round(latest_prob_buy * 100, 2),
        "Confidence_Score": round(confidence_score, 4),
        "Model_Type": "classification",
    }


def _split_and_scale(df):
    prepared_df, X, y, returns = prepare_direction_data(df)
    split_idx = int(len(prepared_df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    returns_train = returns.iloc[:split_idx]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Not enough usable rows after classification preprocessing.")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Direction target has only one class in train/test split.")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return prepared_df, X_train_scaled, X_test_scaled, y_train, y_test, returns_train


def xgboost_direction_train(df, symbol_name, save_artifacts=False, verbose=True, fast_mode=True):
    prepared_df, X_train, X_test, y_train, y_test, returns_train = _split_and_scale(df)
    if verbose:
        print(f"  Classification samples : train={len(X_train)} test={len(X_test)}")

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos if pos else 1.0
    model = xgb.XGBClassifier(
        n_estimators=120 if fast_mode else 250,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=8.0,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        nthread=-1,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    if save_artifacts:
        model.save_model(f"xgboost_direction_{symbol_name}_model.json")
    return make_classification_result(symbol_name, prepared_df, y_test, probabilities, returns_train)


def rf_direction_train(df, symbol_name, save_artifacts=False, verbose=True):
    prepared_df, X_train, X_test, y_train, y_test, returns_train = _split_and_scale(df)
    if verbose:
        print(f"  Classification samples : train={len(X_train)} test={len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    return make_classification_result(symbol_name, prepared_df, y_test, probabilities, returns_train)


def build_sequences(X_scaled, y, lookback=LOOKBACK):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


class DirectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def lstm_direction_train(
    df,
    symbol_name,
    save_artifacts=False,
    verbose=True,
    epochs_override=None,
    lookback_override=None,
    batch_size_override=None,
    patience_override=None,
    fast_mode=True,
):
    lookback = lookback_override or LOOKBACK
    epochs = epochs_override or (4 if fast_mode else 20)
    batch_size = batch_size_override or 256
    patience = patience_override or 2

    prepared_df, X_train, X_test, y_train, y_test, returns_train = _split_and_scale(df)
    X_train_seq, y_train_seq = build_sequences(X_train, y_train.values.astype(np.float32), lookback)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test.values.astype(np.float32), lookback)
    if len(X_train_seq) < 10 or len(X_test_seq) < 5:
        raise ValueError(f"Not enough LSTM classification sequences after LOOKBACK={lookback}.")

    val_split = int(len(X_train_seq) * 0.9)
    X_tr, X_val = X_train_seq[:val_split], X_train_seq[val_split:]
    y_tr, y_val = y_train_seq[:val_split], y_train_seq[val_split:]

    train_loader = DataLoader(TensorDataset(to_torch_tensor(X_tr), to_torch_tensor(y_tr)), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(to_torch_tensor(X_val), to_torch_tensor(y_val)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(to_torch_tensor(X_test_seq), to_torch_tensor(y_test_seq)), batch_size=batch_size)

    model = DirectionLSTM(input_size=X_train_seq.shape[2], hidden_size=16 if fast_mode else 32).to(DEVICE)
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / pos if pos else 1.0], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val = float("inf")
    best_weights = None
    wait = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= max(len(val_loader), 1)

        if verbose:
            print(
                f"  LSTM classifier {symbol_name} epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    probabilities = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch.to(DEVICE))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())

    if save_artifacts:
        torch.save(model.state_dict(), f"lstm_direction_{symbol_name}_model.pt")
    y_test_aligned = y_test.iloc[lookback:]
    return make_classification_result(
        symbol_name,
        prepared_df,
        y_test_aligned,
        np.array(probabilities, dtype=np.float32),
        returns_train,
    )

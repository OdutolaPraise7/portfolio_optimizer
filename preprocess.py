import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(filepath):
    # This is the shared entry point for the modeling pipeline: every model
    # starts from the same cleaned market dataset so their outputs are comparable.
    df = pd.read_csv(filepath, low_memory=False)
    columns = df.columns

    # convert dates
    df["TRANS_DATE"] = pd.to_datetime(df["TRANS_DATE"])

    start_date = dt.datetime.now() - dt.timedelta(days=365)

    # Drop audit/transaction identifiers that do not carry price signal.
    cols_to_drop = ['USERID', 'CHECKED', 'CHECKED_DATE', 'CHECKED_BY', 'SHAREID', 'TRANSID']
    df = df.drop(columns=[c for c in cols_to_drop if c in columns])

    # The downstream model files assume a fully numeric table.
    df = df.fillna(0)

    # A simple return feature gives the models short-term direction context.
    df['DAILY_RETURN'] = df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change().fillna(0)

    # One-hot encoding lets the models learn symbol-specific patterns.
    df_processed = pd.get_dummies(df, columns=['SYMBOL'], drop_first=True)

    # Any rows still containing NaNs after feature creation are unsafe to train on.
    df_processed = df_processed.dropna()

    features = [col for col in df_processed.columns if col not in ['CLOSE_PRICE', 'TRANS_DATE']]
    X = df_processed[features]
    y = df_processed['CLOSE_PRICE']

    # The split objects are not returned here, but leaving this step visible
    # documents that the project expects chronological train/test separation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return df_processed

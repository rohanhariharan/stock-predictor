"""Stock feature engineering, XGBoost training, and next-price forecasting.

Kept separate from the Streamlit UI so the ML logic is reusable and
easy to reason about. Pure pandas/numpy/sklearn/xgboost — no Streamlit
imports here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# Rolling windows that produce NaNs we don't want at the start of the history.
_MIN_TRAIN_ROWS = 80


def load_history(symbol: str) -> pd.DataFrame:
    """Fetch the maximum available daily price history for *symbol*.

    Raises ValueError for bad/missing symbols so the UI can show a friendly
    error instead of crashing.
    """
    import yfinance as yf

    if not symbol or not symbol.strip():
        raise ValueError("Please enter a stock symbol (e.g. AAPL).")

    ticker = yf.Ticker(symbol.strip().upper())
    df = ticker.history(period="max", auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(
            f"No price data found for '{symbol}'. Check the symbol spelling "
            "or that it trades on Yahoo Finance."
        )

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"'Close' prices are missing for '{symbol}'.")

    return df


def _feature_frame(df: pd.DataFrame, lags: int = 12) -> pd.DataFrame:
    """Build the full feature frame (returns, trends, volatility, lags).

    Computed for every row including the final one, so it can be reused
    both for training (rows with a target) and for the out-of-sample
    forecast (the final row, whose target is the not-yet-realized next close).
    """
    o = df["Close"]
    pcts = o.pct_change()

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = pcts
    feats["ret_5"] = o.pct_change(5)
    feats["ret_10"] = o.pct_change(10)
    feats["logret"] = np.log(o / o.shift(1))
    feats["ma_5"] = o.rolling(5).mean()
    feats["ma_10"] = o.rolling(10).mean()
    feats["ma_50"] = o.rolling(50).mean()
    feats["vol_5"] = pcts.rolling(5).std()
    feats["vol_10"] = pcts.rolling(10).std()

    hi = df["High"]
    lo = df["Low"]
    feats["range_1"] = (hi - lo) / o.shift(1)
    feats["range_5"] = (hi - lo).rolling(5).mean() / o.shift(1)

    if "Volume" in df.columns:
        vol = df["Volume"].replace(0, np.nan)
        feats["vol_ratio"] = vol / vol.rolling(10).mean()
    else:
        feats["vol_ratio"] = 0.0

    # Add lagged closes so the model can see price level, not just returns.
    for lag in range(1, lags + 1):
        feats[f"close_lag_{lag}"] = o.shift(lag)

    # Drop any row missing a feature value (start of history).
    feats = feats[feats.notna().all(axis=1)]
    return feats.astype(np.float64)


def build_features(df: pd.DataFrame, lags: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
    """Build sliding-window features and the next-step target column.

    Returns (features, target) aligned row-for-row, where target row *t* is
    the close on day *t+1*. Only rows whose target is already known (i.e.
    every row except the latest) are included by the caller that trains;
    this function returns the full aligned set and lets the train/forecast
    split happen downstream.
    """
    feats = _feature_frame(df, lags)
    o = df["Close"].reindex(feats.index)

    # Target: next-day simple return — scale-invariant, so a 46-year history
    # spanning prices from ~$0.10 to ~$330 stays one comparably-sized problem.
    target = o.shift(-1) / o - 1.0

    valid = target.notna()
    X = feats[valid]
    y = target[valid]

    if len(X) < _MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {len(X)} usable historical rows — not enough to train a model. "
            f"Need at least {_MIN_TRAIN_ROWS}."
        )

    return X, y


def predict_next(model: xgb.XGBRegressor, df: pd.DataFrame, lags: int = 12) -> float:
    """Forecast the next day's close after the last observed close.

    The final row's features are computable today, but its target (tomorrow's
    return) does not exist yet — so it is never part of training. We build that
    row directly, predict the return, and turn it back into a price.
    """
    feats = _feature_frame(df, lags)
    last_row = feats.iloc[[-1]]
    pred_return = float(model.predict(last_row)[0])
    price = df["Close"].iloc[-1] * (1.0 + pred_return)
    return max(float(price), 0.0)  # prices can't be negative


def forecast_recursive(
    model: xgb.XGBRegressor, df: pd.DataFrame, steps: int = 10, lags: int = 12
) -> pd.Series:
    """Recursively forecast the next *steps* closes, producing a forecast line.

    Each predicted return is turned into a close and fed back in as the newest
    observation, so the feature frame advances one trading day at a time. Future
    rows reuse the last known volume and set High/Low equal to the predicted
    close (intraday range is unknowable ahead of time), which only softens the
    ``range_*`` features.
    """
    steps = int(steps)
    if steps < 1:
        return pd.Series(dtype=float)

    work = df.copy()
    last_volume = float(work["Volume"].iloc[-1]) if "Volume" in work.columns else 0.0
    preds: list[float] = []
    dates: list[pd.Timestamp] = []

    for _ in range(steps):
        feats = _feature_frame(work, lags)
        if feats.empty:
            break
        pred_return = float(model.predict(feats.iloc[[-1]])[0])
        new_close = max(float(work["Close"].iloc[-1]) * (1.0 + pred_return), 0.0)
        preds.append(new_close)

        next_day = work.index[-1] + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        dates.append(next_day)

        new_row = {col: new_close for col in work.columns}
        if "Volume" in work.columns:
            new_row["Volume"] = last_volume
        work.loc[next_day] = new_row

    return pd.Series(preds, index=pd.DatetimeIndex(dates))


def forecast_arima(
    df: pd.DataFrame, order: Tuple[int, int, int] = (1, 1, 1), steps: int = 10
) -> pd.Series:
    """Forecast the next *steps* closes with ARIMA on log prices.

    Fitting happens on ``log(Close)`` so the multiplicative price scale is
    respected; predictions are exponentiated back to dollars. Dates mirror the
    trading-day snapping used by the recursive XGBoost forecast.
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    steps = int(steps)
    if steps < 1:
        return pd.Series(dtype=float)

    log_price = np.log(df["Close"].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fit on plain values: the yfinance index is tz-aware and has holiday
        # gaps, which statsmodels rejects as "no supported index".
        result = ARIMA(log_price.to_numpy(), order=tuple(order)).fit()
        mean = np.exp(np.asarray(result.forecast(steps=steps), dtype=float))

    dates: list[pd.Timestamp] = []
    cursor = df.index[-1]
    for _ in range(steps):
        cursor = cursor + pd.Timedelta(days=1)
        while cursor.weekday() >= 5:
            cursor += pd.Timedelta(days=1)
        dates.append(cursor)

    return pd.Series(mean, index=pd.DatetimeIndex(dates))


def atm_iv_by_expiry(
    symbol: str, expiry: str | None = None
) -> Tuple[pd.DataFrame, dict]:
    """Return at-the-money implied volatility per available expiration.

    Fetches the options chain from yfinance and, for each expiration, reads the
    at-the-money call's ``impliedVolatility`` (strike nearest the spot price).

    Returns a DataFrame indexed by expiration date (``pd.DatetimeIndex``) with an
    ``atm_iv`` column, plus a small info dict (list of all expiries and the spot).
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol.strip().upper())
    expiries = list(ticker.options)

    if not expiries:
        raise ValueError(
            f"No options chain found for '{symbol}'. It may not trade options on Yahoo."
        )

    spot = float(ticker.history(period="1d")["Close"].dropna().iloc[-1])

    rows = {}
    chosen = [expiry] if expiry else expiries
    for exp in chosen:
        if exp not in expiries:
            continue
        chain = ticker.option_chain(exp)
        calls = chain.calls
        if calls is None or calls.empty:
            continue
        calls = calls.dropna(subset=["strike", "impliedVolatility"])
        if calls.empty:
            continue
        nearest = calls.iloc[(calls["strike"] - spot).abs().argmin()]
        rows[exp] = nearest["impliedVolatility"]

    if not rows:
        raise ValueError(f"Could not compute implied volatility for '{symbol}'.")

    df = pd.DataFrame(
        {"atm_iv": [rows[e] for e in rows]},
        index=pd.DatetimeIndex([pd.Timestamp(e) for e in rows], name="expiry"),
    )
    df.index = df.index.normalize()
    df = df.sort_index()

    return df, {"expiries": expiries, "spot": spot}


def train_forecast_model(X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.XGBRegressor, dict]:
    """Train an XGBoost regressor and return model + training summary.

    Splits chronologically (80/20) so the validation set is what the model
    would actually face (most recent data), never shuffled into training.
    """
    split_idx = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
    X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, y_pred))

    return model, {
        "mae": mae,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "total_rows": len(X),
    }

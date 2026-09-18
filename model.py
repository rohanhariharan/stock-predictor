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


def _next_trading_day(day: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekday after *day*, preserving any timezone."""
    nxt = day + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def _future_trading_days(last_day: pd.Timestamp, steps: int) -> list[pd.Timestamp]:
    """Return the next *steps* trading days after *last_day*."""
    out: list[pd.Timestamp] = []
    cursor = last_day
    for _ in range(int(steps)):
        cursor = _next_trading_day(cursor)
        out.append(cursor)
    return out


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


def _return_pct(df: pd.DataFrame) -> pd.Series:
    """Close-to-close percent returns (in %), matching the GARCH input scale."""
    return df["Close"].astype(float).pct_change() * 100.0


def garch_conditional_vol(
    df: pd.DataFrame, p: int = 1, q: int = 1
) -> pd.Series:
    """One-step *conditional* volatility from a GARCH(p, q) fit (in % per day).

    Uses ``arch``'s in-sample ``conditional_volatility``, which at row *t* is a
    function of returns through *t* only — so it is a legitimate, non-leaky
    feature that can be joined onto the feature frame as of each day.
    """
    import warnings

    from arch import arch_model

    returns = _return_pct(df).dropna()
    if len(returns) < 50:
        raise ValueError("Need at least 50 daily returns to fit GARCH.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(
            returns.to_numpy(), vol="GARCH", p=int(p), q=int(q), dist="normal"
        ).fit(disp="off")
        vol = np.asarray(fit.conditional_volatility, dtype=float)

    return pd.Series(vol, index=returns.index, name="garch_vol")


def _feature_frame(
    df: pd.DataFrame, lags: int = 12, garch_vol: pd.Series | None = None
) -> pd.DataFrame:
    """Build the full feature frame (returns, trends, volatility, lags).

    Computed for every row including the final one, so it can be reused
    both for training (rows with a target) and for the out-of-sample
    forecast (the final row, whose target is the not-yet-realized next close).

    If *garch_vol* (a per-day conditional volatility series, see
    :func:`garch_conditional_vol`) is given, it is joined on by date along with
    a short rolling average and a level ratio, so the model can condition on the
    current volatility regime.
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

    if garch_vol is not None:
        gv = garch_vol.reindex(feats.index)
        feats["garch_vol"] = gv
        ma20 = gv.rolling(20).mean()
        feats["garch_vol_ratio"] = gv / ma20
        feats["garch_vol_chg"] = gv.diff()

    # Add lagged closes so the model can see price level, not just returns.
    for lag in range(1, lags + 1):
        feats[f"close_lag_{lag}"] = o.shift(lag)

    # Drop any row missing a feature value (start of history).
    feats = feats[feats.notna().all(axis=1)]
    return feats.astype(np.float64)


def build_features(
    df: pd.DataFrame, lags: int = 12, garch_vol: pd.Series | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build sliding-window features and the next-step target column.

    Returns (features, target) aligned row-for-row, where target row *t* is
    the close on day *t+1*. Only rows whose target is already known (i.e.
    every row except the latest) are included by the caller that trains;
    this function returns the full aligned set and lets the train/forecast
    split happen downstream.
    """
    feats = _feature_frame(df, lags, garch_vol)
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


def predict_return(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    lags: int = 12,
    garch_vol: pd.Series | None = None,
) -> float:
    """Predict the next-day simple return from the latest observed row.

    Separated from :func:`predict_next` so callers can anchor the predicted
    return to a live/intraday price instead of the last daily close.
    """
    feats = _feature_frame(df, lags, garch_vol)
    if feats.empty:
        raise ValueError("Not enough history to compute features.")
    return float(model.predict(feats.iloc[[-1]])[0])


def predict_next(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    lags: int = 12,
    garch_vol: pd.Series | None = None,
) -> float:
    """Forecast the next day's close after the last observed close.

    The final row's features are computable today, but its target (tomorrow's
    return) does not exist yet — so it is never part of training. We build that
    row directly, predict the return, and turn it back into a price.
    """
    pred_return = predict_return(model, df, lags, garch_vol)
    price = df["Close"].iloc[-1] * (1.0 + pred_return)
    return max(float(price), 0.0)  # prices can't be negative


def forecast_recursive(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    steps: int = 10,
    lags: int = 12,
    garch_vol: pd.Series | None = None,
) -> pd.Series:
    """Recursively forecast the next *steps* closes, producing a forecast line.

    Each predicted return is turned into a close and fed back in as the newest
    observation, so the feature frame advances one trading day at a time. Future
    rows reuse the last known volume and set High/Low equal to the predicted
    close (intraday range is unknowable ahead of time), which only softens the
    ``range_*`` features.

    When *garch_vol* is supplied, the last known conditional volatility is
    carried forward onto each future row so the model keeps seeing a vol regime
    during the recursion (future GARCH vol must itself be forecast, which this
    simple recursion does not do).
    """
    steps = int(steps)
    if steps < 1:
        return pd.Series(dtype=float)

    work = df.copy()
    last_volume = float(work["Volume"].iloc[-1]) if "Volume" in work.columns else 0.0
    work_garch = garch_vol.copy() if garch_vol is not None else None
    preds: list[float] = []
    dates: list[pd.Timestamp] = []

    for _ in range(steps):
        feats = _feature_frame(work, lags, work_garch)
        if feats.empty:
            break
        pred_return = float(model.predict(feats.iloc[[-1]])[0])
        new_close = max(float(work["Close"].iloc[-1]) * (1.0 + pred_return), 0.0)
        preds.append(new_close)

        next_day = _next_trading_day(work.index[-1])
        dates.append(next_day)

        new_row = {col: new_close for col in work.columns}
        if "Volume" in work.columns:
            new_row["Volume"] = last_volume
        work.loc[next_day] = new_row

        if work_garch is not None:
            work_garch.loc[next_day] = float(work_garch.iloc[-1])

    return pd.Series(preds, index=pd.DatetimeIndex(dates))


def forecast_arima(
    df: pd.DataFrame, order: Tuple[int, int, int] = (1, 1, 1), steps: int = 10
) -> pd.DataFrame:
    """Forecast the next *steps* closes with ARIMA on log prices.

    Fitting happens on ``log(Close)`` so the multiplicative price scale is
    respected; predictions and their prediction intervals are exponentiated back
    to dollars. Dates mirror the trading-day snapping used by the recursive
    XGBoost forecast.

    Returns a frame indexed by forecast date with columns ``mean``, ``lo80``,
    ``hi80``, ``lo95`` and ``hi95``. The bands widen with horizon, which for a
    near-random-walk price is the only genuinely informative part of the output.
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    steps = int(steps)
    if steps < 1:
        return pd.DataFrame(
            columns=["mean", "lo80", "hi80", "lo95", "hi95"], dtype=float
        )

    log_price = np.log(df["Close"].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fit on plain values: the yfinance index is tz-aware and has holiday
        # gaps, which statsmodels rejects as "no supported index".
        result = ARIMA(log_price.to_numpy(), order=tuple(order)).fit()
        fc = result.get_forecast(steps=steps)
        mean = np.exp(np.asarray(fc.predicted_mean, dtype=float))
        ci80 = np.exp(np.asarray(fc.conf_int(alpha=0.20), dtype=float))
        ci95 = np.exp(np.asarray(fc.conf_int(alpha=0.05), dtype=float))

    dates: list[pd.Timestamp] = []
    cursor = df.index[-1]
    for _ in range(steps):
        cursor = cursor + pd.Timedelta(days=1)
        while cursor.weekday() >= 5:
            cursor += pd.Timedelta(days=1)
        dates.append(cursor)

    return pd.DataFrame(
        {
            "mean": mean,
            "lo80": ci80[:, 0],
            "hi80": ci80[:, 1],
            "lo95": ci95[:, 0],
            "hi95": ci95[:, 1],
        },
        index=pd.DatetimeIndex(dates),
    )


def forecast_garch_volatility(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    steps: int = 10,
    window: int = 60,
) -> pd.DataFrame:
    """Fit GARCH(p, q) and return conditional volatility in percent per day.

    GARCH models the *variance* of returns, not their direction, so this is a
    volatility overlay rather than a price forecast. Returns an annualized-style
    daily vol (%) over recent history plus the forward forecast, as a frame with
    ``vol`` (in %), a ``kind`` column ('fitted' or 'forecast'), and the model's
    fitted parameters and persistence in ``.attrs``.
    """
    import warnings

    from arch import arch_model

    steps = int(steps)
    returns = (df["Close"].astype(float).pct_change().dropna() * 100.0)
    if len(returns) < 50:
        raise ValueError("Need at least 50 daily returns to fit GARCH.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fit = arch_model(
            returns.to_numpy(), vol="GARCH", p=int(p), q=int(q), dist="normal"
        ).fit(disp="off")
        fcast = model_fit.forecast(horizon=steps, reindex=False)
        future_var = np.asarray(fcast.variance.to_numpy()).ravel()[:steps]
        future_vol = np.sqrt(np.clip(future_var, 0.0, None))

    # Conditional vol over the tail of history for the fitted/observed line.
    fitted_vol = np.asarray(model_fit.conditional_volatility, dtype=float)
    fitted_tail = fitted_vol[-window:]
    hist_index = returns.index[-len(fitted_tail):]

    # Forward dates, snapped the same way as the other forecasts.
    future_dates: list[pd.Timestamp] = []
    cursor = df.index[-1]
    for _ in range(len(future_vol)):
        cursor = cursor + pd.Timedelta(days=1)
        while cursor.weekday() >= 5:
            cursor += pd.Timedelta(days=1)
        future_dates.append(cursor)

    out = pd.DataFrame(
        {
            "vol": np.concatenate([fitted_tail, future_vol]),
            "kind": ["fitted"] * len(fitted_tail) + ["forecast"] * len(future_vol),
        },
        index=pd.DatetimeIndex(list(hist_index) + future_dates),
    )

    params = dict(model_fit.params)
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    out.attrs["params"] = params
    out.attrs["persistence"] = alpha + beta
    out.attrs["order"] = (int(p), int(q))
    out.attrs["last_vol"] = float(fitted_vol[-1])
    return out


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

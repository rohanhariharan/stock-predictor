# 📈 Stock Price Predictor (XGBoost + Streamlit)

A web dashboard that shows a stock's **current value** alongside its **symbol**,
trains an **XGBoost** model on the **maximum available history**, and **overlays
the forecast** for the next day's close right next to the current value.

## What it does
- Fetches the **most history possible** per ticker via `yfinance` (`period="max"`).
- Engineers features: returns, log-returns, moving averages, volatility, daily
  range ratios, volume momentum, and lagged closes.
- Trains an `XGBRegressor` (chronological 80/20 split; validation on the most
  recent data).
- Predicts the **next day's close** after the last observed close.
- Renders a Plotly chart: recent price line + a green marker for the current
  value + a red dotted line/star to the **forecast** next point.

## Setup
```bash
cd /Users/rohanhariharan/xgboost
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: if `yfinance` fails to import on Python 3.14, upgrade it with
> `pip install --upgrade yfinance`.

## Run
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501. Enter a ticker (e.g. `AAPL`) or pick
a sample, then press **Run / refresh**.

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI: input, chart, metrics, forecast overlay |
| `model.py` | Feature engineering, XGBoost training, next-close forecast (UI-free) |

## Caveats
- Stock forecasting from daily bars alone is noisy; treat results as a demo,
  not investment advice. The validation MAE shown gives a sense of error size.
- Data lags Yahoo's published close; "current value" is the latest *daily* close,
  not a real-time intraday tick.
- Weekend/holiday forecast points are snapped to the next trading day on the chart.

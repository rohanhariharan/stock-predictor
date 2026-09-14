"""Stock price dashboard: current value + XGBoost next-close forecast overlay.

Run with:
    streamlit run app.py

Type a ticker (e.g. AAPL) and the app fetches the maximum available history,
trains an XGBoost model to forecast the next day's close, and overlays that
forecast right next to the current value on a Plotly chart.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import model

# Page config must be the first Streamlit call.
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Price Predictor")
st.caption("Train XGBoost on maximum history → forecast the next close → overlay on the current value.")


# ---- Implied volatility tab (options chains) --------------------------------

def get_chain(symbol: str) -> tuple[pd.DataFrame, dict] | None:
    """Fetch the at-the-money IV per expiration for *symbol* (live, uncached)."""
    try:
        return model.atm_iv_by_expiry(symbol)
    except ValueError:
        return None


def iv_tab() -> None:
    """Render the 'Implied Volatility' tab.

    Single symbol; the user picks which expirations to display. IV is fetched
    fresh on every rerun and accumulated in session state, so with auto-refresh
    on you get a live IV-over-time overlay that builds up during the session.
    """
    st.subheader("Realtime implied volatility (by expiration)")
    st.caption(
        "Pick one symbol and choose expirations. IV is polled on every refresh "
        "and an over-time series is accumulated while this tab is open."
    )

    with st.sidebar:
        st.markdown("")
        st.subheader("📉 Implied volatility")
        iv_symbol = st.text_input(
            "IV symbol", value="AAPL", max_chars=10
        ).strip().upper()

        avail_dates: list[str] = []
        result = get_chain(iv_symbol) if iv_symbol else None
        if result is not None:
            iv_df, iv_info = result
            avail_dates = [str(d.date()) for d in iv_df.index]
            chosen = st.multiselect(
                "Expirations to show",
                options=avail_dates,
                default=avail_dates[:2],
                help="Each selected expiration becomes its own IV line.",
            )
        else:
            chosen = []

    if not iv_symbol or result is None:
        if iv_symbol:
            st.warning(f"No options chain available for '{iv_symbol}'.")
        else:
            st.info("Enter a symbol on the left, then pick expirations.")
        return

    # Accumulate a live IV-over-time history per (symbol, expiry) across reruns.
    hist = st.session_state.setdefault("iv_hist", {})
    bucket = hist.setdefault(iv_symbol, {})
    now = dt.datetime.now()
    cap = 400
    for exp, iv_val in iv_df["atm_iv"].items():
        ey = str(exp.date())
        pts = bucket.setdefault(ey, [])
        pts.append((now, float(iv_val)))
        bucket[ey] = pts[-cap:]

    if not chosen:
        st.info("Pick at least one expiration on the left to plot IV over time.")
        return

    traces = [
        go.Scatter(
            x=[p[0] for p in bucket[ey]],
            y=[p[1] for p in bucket[ey]],
            mode="lines+markers",
            name=f"{ey} expiry",
            line=dict(width=2),
            marker=dict(size=4),
        )
        for ey in chosen
        if ey in bucket and len(bucket[ey]) > 1
    ]

    if not traces:
        st.info("Collecting IV points… keep the tab open through a refresh cycle. "
                "Two or more samples per expiration are needed to draw a line.")
        return

    latest = pd.Series(
        {ey: bucket[ey][-1][1] for ey in chosen if ey in bucket}
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"${iv_info['spot']:,.2f}")
    c2.metric("Avg IV (selected)", f"{latest.mean():.1%}")
    c3.metric("Latest sample", now.strftime("%H:%M:%S"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{iv_symbol} — ATM IV over time by expiration (points: {now:%Y-%m-%d %H:%M})",
        xaxis_title="Time",
        yaxis_title="Implied volatility",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=-0.2, x=0),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=60),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View accumulated samples", expanded=False):
        series = {
            ey: pd.Series([p[1] for p in bucket[ey]],
                          index=[p[0] for p in bucket[ey]])
            for ey in chosen
        }
        raw = pd.DataFrame(series).sort_index()
        raw.index.name = "Sample time"
        st.dataframe(raw.style.format("{:.1%}"), use_container_width=True)
    st.caption(
        f"{len(bucket)} expirations tracked · {len(chosen)} plotted · "
        f"history resets if you change the symbol."
    )


# Data + model are cached together (keyed by symbol). Auto-refresh clears this
# cache exactly once per chosen interval (see maybe_clear_cache), so a normal
# browser rerun is cheap and re-fetches/re-trains only when the interval elapses.
@st.cache_data(ttl=3600, show_spinner=False)
def get_dashboard(symbol: str) -> tuple[pd.DataFrame, dict]:
    """Fetch full history, train XGBoost, and forecast the next close."""
    df = model.load_history(symbol)
    X, y = model.build_features(df)
    xgb_model, info = model.train_forecast_model(X, y)
    fc = {
        "predicted_next": model.predict_next(xgb_model, df),
        "mae": info["mae"],
        "train_size": info["train_size"],
        "test_size": info["test_size"],
        "total_rows": info["total_rows"],
    }
    return df, fc


def maybe_clear_cache(interval_min: int, force: bool = False) -> None:
    """Drop the cached dashboard once per interval so auto-refresh pulls new data.

    Tracks the last clear time in session state; only triggers a real cache
    eviction (and therefore a fresh fetch + retrain) when the interval elapses.
    """
    now = dt.datetime.now()
    last = st.session_state.get("last_clear")
    stale = last is None or (now - last).total_seconds() >= interval_min * 60
    if force or stale:
        get_dashboard.clear()
        st.session_state["last_clear"] = now


def run(symbol: str, auto: bool) -> None:
    """Render the full dashboard for one symbol (data already fetched/trained)."""
    with st.spinner("Fetching data & training XGBoost…"):
        try:
            df, fc = get_dashboard(symbol)
        except ValueError as e:
            st.error(str(e))
            return

    if df.empty:
        st.error(f"No data returned for {symbol}.")
        return

    predicted_next = fc["predicted_next"]
    info = fc

    history = df.copy()
    latest_close = float(history["Close"].iloc[-1])
    latest_date = history.index[-1]
    next_date = latest_date + pd.Timedelta(days=1)
    # Skip week-ends so the forecast marker sits on a trading day.
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)

    pct_change = (predicted_next / latest_close - 1.0) * 100.0

    # ---- Top stats ----------------------------------------------------------
    st.subheader(f"{symbol.upper()} — {history.index[0].date()} → {latest_date.date()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", symbol.upper())
    c2.metric(
        "Current value",
        f"${latest_close:,.2f}",
        f"{latest_date.strftime('%b %d, %Y')}",
    )
    c3.metric(
        "Predicted next close",
        f"${predicted_next:,.2f}",
        f"{pct_change:+.2f}% vs current",
    )
    c4.metric(
        "Validation MAE",
        f"{info['mae']:.2%}",
        f"{info['train_size']:,} train / {info['test_size']:,} test rows",
    )

    st.caption(
        f"Trained on {info['total_rows']:,} contiguous days (the maximum usable "
        f"from {len(history):,} fetched). Horizon = 1 trading day ahead.  ·  "
        f"{'🔄 Auto-refresh on' if auto else 'Auto-refresh off'} · "
        f"updated {dt.datetime.now():%H:%M:%S}"
    )

    # ---- Chart: recent history + forecast overlay ---------------------------
    recent = history.tail(120)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # Overlay: current value (actual) and the forecast (next point).
    fig.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_close],
            mode="markers+text",
            name="Current value",
            text=["Current"],
            textposition="bottom center",
            marker=dict(color="#2ca02c", size=11, symbol="circle"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest_date, next_date],
            y=[latest_close, predicted_next],
            mode="lines+markers",
            name="Forecast → next close",
            line=dict(color="#d62728", width=2, dash="dot"),
            marker=dict(color="#d62728", size=12, symbol="star"),
            text=[None, "Prediction"],
            textposition="top center",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"{symbol.upper()} — close with next-close forecast overlay (last {len(recent)} days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Full-history mini chart -------------------------------------------
    st.subheader("Full history (maximum period fetched)")
    full_fig = go.Figure()
    full_fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4", width=1),
        )
    )
    full_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(full_fig, use_container_width=True)


# ---- Sidebar input + main flow ----------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_symbol = st.text_input("Stock symbol", value="AAPL", max_chars=10)
    sample = st.selectbox(
        "Or pick a sample",
        ["", "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META"],
        index=1,
    )
    symbol = (sample or default_symbol).strip().upper()

    if st.button("Run / refresh now", type="primary"):
        maybe_clear_cache(1, force=True)

    st.divider()
    st.subheader("🔄 Auto-refresh")
    enable_auto = st.checkbox(
        "Enable auto-refresh",
        value=True,
        help="Re-fetch + retrain on a schedule so the chart updates by itself.",
    )
    interval_min = st.selectbox(
        "Refresh every",
        [1, 5, 10, 15, 30, 60],
        index=1,
        help="Daily close data only changes once a day, so 5–15 min is plenty.",
    )
    st.caption(
        f"Next data refresh: "
        f"{'every ' + str(interval_min) + ' min' if enable_auto else 'only on demand'}"
    )

    st.divider()
    st.caption(
        "- Fetches **maximum** Yahoo Finance history\n"
        "- Trains XGBoost (200–300 trees)\n"
        "- Forecast horizon: 1 trading day ahead"
    )

if symbol:
    tab_predict, tab_iv = st.tabs(["📈 Price + Forecast", "📉 Implied Volatility"])
    # Schedule the auto-refresh first so it re-fires on every rerun.
    if enable_auto:
        st_autorefresh(
            interval=int(interval_min * 60 * 1000),
            key=f"stock-auto-refresh-{symbol}",
        )
        maybe_clear_cache(interval_min)
    with tab_predict:
        run(symbol, auto=enable_auto)
    with tab_iv:
        iv_tab()
else:
    st.info("Enter a stock symbol on the left to get started.")

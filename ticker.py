"""Live ticker dashboard for a single symbol.

Usage:
    python ticker.py AAPL
    python ticker.py AAPL --watch
    python ticker.py TSLA -w --refresh 2

Shows the freshest available quote (Yahoo's most recent price) plus the day's
stats and an intraday sparkline built from 1-minute bars — the finest
granularity Yahoo exposes. There is no 1-second feed.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Optional

import typer
import yfinance as yf
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# yfinance logs raw HTTP errors for bad symbols; keep our own output clean.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

app = typer.Typer(
    add_completion=False, help="Live price dashboard for a ticker.", no_args_is_help=True
)
console = Console()


def _set_prog_name() -> None:
    """Show 'ticker' rather than 'ticker.py' in usage/help output."""
    import sys

    sys.argv[0] = "ticker"


_set_prog_name()

_BLOCKS = "▁▂▃▄▅▆▇█"


def _fmt_price(value: Optional[float]) -> str:
    """Format a price with sensible precision depending on magnitude."""
    if value is None:
        return "—"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def _fmt_volume(value: Optional[float]) -> str:
    if value is None:
        return "—"
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"{value / cutoff:,.2f}{suffix}"
    return f"{value:,.0f}"


def _sparkline(values: list[float]) -> str:
    """Render a list of numbers as a unicode block sparkline."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _BLOCKS[3] * len(values)
    span = hi - lo
    return "".join(
        _BLOCKS[int((v - lo) / span * (len(_BLOCKS) - 1))] for v in values
    )


def fetch_quote(symbol: str) -> dict:
    """Fetch the latest quote and identifying info for *symbol*."""
    ticker = yf.Ticker(symbol)

    try:
        info = ticker.get_info()
    except Exception:
        info = {}

    try:
        fi = ticker.fast_info
        # Read attributes directly: fast_info.get()/keys() can trigger extra
        # network calls (and hangs) in some yfinance versions.
        last = getattr(fi, "last_price", None)
        prev = getattr(fi, "previous_close", None)
        open_ = getattr(fi, "open", None)
        day_high = getattr(fi, "day_high", None)
        day_low = getattr(fi, "day_low", None)
        volume = getattr(fi, "last_volume", None)
        currency = getattr(fi, "currency", None)
        exchange = getattr(fi, "exchange", None)
        timezone = getattr(fi, "timezone", None)
        market_cap = getattr(fi, "market_cap", None)
    except Exception as fi_err:
        raise ValueError(
            f"No live quote found for '{symbol}'. Check the symbol spelling "
            f"(e.g. AAPL, MSFT, BTC-USD)."
        ) from fi_err

    name = info.get("longName") or info.get("shortName") or symbol.upper()

    if last is None and not info:
        raise ValueError(
            f"No quote found for '{symbol}'. Check the symbol spelling."
        )

    return {
        "symbol": symbol.upper(),
        "name": name,
        "exchange": info.get("fullExchangeName") or exchange or "—",
        "quote_type": info.get("quoteType") or "—",
        "currency": currency or info.get("currency") or "",
        "last": last,
        "prev_close": prev,
        "open": open_,
        "day_high": day_high,
        "day_low": day_low,
        "volume": volume,
        "market_cap": info.get("marketCap") or market_cap,
        "timezone": timezone or "—",
    }


def fetch_intraday(symbol: str, interval: str = "1m") -> tuple[list[float], Optional[dt.datetime]]:
    """Fetch today's intraday closes at the given *interval*."""
    try:
        df = yf.Ticker(symbol).history(
            period="1d", interval=interval, prepost=False, auto_adjust=False
        )
    except Exception:
        return [], None

    if df is None or df.empty:
        return [], None

    closes = [float(v) for v in df["Close"].dropna().to_list()]
    stamp = df.index[-1].to_pydatetime() if len(df.index) else None
    return closes, stamp


@dataclass
class ModelState:
    """A trained XGBoost model plus the context needed to re-predict cheaply.

    Training (and the GARCH fit) takes seconds, so it is done once and refreshed
    only on a slow cadence; the live panel re-predicts from these in milliseconds
    by anchoring the predicted *return* to the current price.
    """

    model: object
    df: object
    garch_vol: object
    pred_return: float
    mae: float
    feature_count: int
    trained_at: dt.datetime
    uses_garch: bool


def fit_model(symbol: str, use_garch: bool = True, garch_p: int = 1, garch_q: int = 1) -> ModelState:
    """Fetch max history, fit GARCH (optional), and train XGBoost once."""
    import model as ml

    df = ml.load_history(symbol)

    garch_vol = None
    if use_garch:
        try:
            garch_vol = ml.garch_conditional_vol(df, p=garch_p, q=garch_q)
        except Exception:
            garch_vol = None

    X, y = ml.build_features(df, garch_vol=garch_vol)
    xgb_model, info = ml.train_forecast_model(X, y)
    pred_return = ml.predict_return(xgb_model, df, garch_vol=garch_vol)

    return ModelState(
        model=xgb_model,
        df=df,
        garch_vol=garch_vol,
        pred_return=pred_return,
        mae=info["mae"],
        feature_count=X.shape[1],
        trained_at=dt.datetime.now(),
        uses_garch=garch_vol is not None,
    )


def build_model_panel(state: Optional[ModelState], live_price: Optional[float]) -> Panel:
    """Build the XGBoost prediction panel, anchored to *live_price*."""
    if state is None:
        return Panel(
            Text("Model: fitting…", style="dim"),
            title="🤖 XGBoost prediction",
            border_style="grey50",
            expand=False,
            padding=(0, 2),
        )

    base = live_price if live_price is not None else float(state.df["Close"].iloc[-1])
    predicted = max(base * (1.0 + state.pred_return), 0.0)
    diff = predicted - base
    pct = state.pred_return * 100.0

    colour = "green" if diff > 0 else "red" if diff < 0 else "yellow"
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "•"

    pred_line = Text()
    pred_line.append(f"{_fmt_price(predicted)}", style=f"bold {colour}")
    pred_line.append(f"   {arrow} {pct:+.2f}% vs live", style=f"bold {colour}")

    anchor = "live" if live_price is not None else "last close"
    src = "with GARCH vol" if state.uses_garch else "no GARCH vol"
    sub = Text(
        f"next-close forecast · anchored to {anchor} · {src} · "
        f"{state.feature_count} features",
        style="dim",
    )

    stats = Table.grid(padding=(0, 2))
    stats.add_column(justify="right", style="dim")
    stats.add_column(justify="left")
    stats.add_column(justify="right", style="dim")
    stats.add_column(justify="left")
    stats.add_row(
        "Predicted",
        _fmt_price(predicted),
        "Change",
        f"{diff:+,.2f}",
    )
    stats.add_row(
        "Validation MAE",
        f"{state.mae:.2%}",
        "Trained",
        state.trained_at.strftime("%H:%M:%S"),
    )

    body = Group(pred_line, sub, Text(""), stats)

    return Panel(
        body,
        title="🤖 XGBoost prediction",
        subtitle=(
            "⚠️ near-random-walk target — treat as a demo"
            if abs(pct) < 0.01
            else "demo, not investment advice"
        ),
        border_style=colour,
        expand=False,
        padding=(0, 2),
    )


def build_dashboard(
    symbol: str, interval: str = "1m", quote: Optional[dict] = None
) -> Panel:
    """Build the rich price panel for *symbol* (quote, stats, sparkline).

    Pass *quote* to reuse an already-fetched quote and avoid a second network
    round-trip on each live refresh.
    """
    if quote is None:
        try:
            q = fetch_quote(symbol)
        except ValueError as e:
            return Panel(
                Text(str(e), style="bold red"), title="Error", border_style="red"
            )
    else:
        q = quote

    last = q["last"]
    prev = q["prev_close"]

    if last is None:
        return Panel(
            Text(f"No price available for '{q['symbol']}'.", style="bold yellow"),
            title=q["symbol"],
            border_style="yellow",
        )

    change = (last - prev) if prev else None
    pct = (change / prev * 100.0) if (change is not None and prev) else None

    if change is None:
        colour, arrow = "yellow", "•"
    elif change > 0:
        colour, arrow = "green", "▲"
    elif change < 0:
        colour, arrow = "red", "▼"
    else:
        colour, arrow = "yellow", "•"

    price_line = Text()
    price_line.append(f"{_fmt_price(last)}", style=f"bold {colour}")
    if q["currency"]:
        price_line.append(f" {q['currency']}", style="dim")
    if change is not None and pct is not None:
        price_line.append(
            f"   {arrow} {change:+,.2f} ({pct:+.2f}%)", style=f"bold {colour}"
        )

    closes, stamp = fetch_intraday(symbol, interval)

    if stamp is not None:
        try:
            stamp_txt = stamp.strftime("%H:%M %Z").strip()
        except Exception:
            stamp_txt = str(stamp)
    else:
        stamp_txt = "—"

    sub_line = Text(
        f"{q['exchange']} · {q['quote_type']} · as of {stamp_txt}",
        style="dim",
    )

    body: list = [price_line, sub_line]

    if closes:
        width = max(10, min(len(closes), console.width - 6))
        spark_vals = closes[-width:]
        spark = Text(_sparkline(spark_vals), style=colour)
        body.append(Text(""))
        body.append(spark)
        lo, hi = min(spark_vals), max(spark_vals)
        body.append(
            Text(
                f"{interval} bars · {len(spark_vals)} pts · low {_fmt_price(lo)} / high {_fmt_price(hi)}",
                style="dim",
            )
        )

    stats = Table.grid(padding=(0, 2))
    stats.add_column(justify="right", style="dim")
    stats.add_column(justify="left")
    stats.add_column(justify="right", style="dim")
    stats.add_column(justify="left")

    day_range = "—"
    if q["day_low"] is not None and q["day_high"] is not None:
        day_range = f"{_fmt_price(q['day_low'])} – {_fmt_price(q['day_high'])}"

    stats.add_row("Open", _fmt_price(q["open"]), "Prev close", _fmt_price(prev))
    stats.add_row("Day range", day_range, "Volume", _fmt_volume(q["volume"]))

    body.append(Text(""))
    body.append(stats)

    title = Text()
    title.append(q["symbol"], style="bold")
    title.append(f" · {q['name']}", style="none")

    return Panel(
        Group(*body),
        title=title,
        subtitle=f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        border_style=colour,
        expand=False,
        padding=(1, 2),
    )


def build_view(
    symbol: str,
    state: Optional[ModelState],
    interval: str = "1m",
) -> Group:
    """Build the full view: price panel + XGBoost prediction panel below it."""
    quote = None
    try:
        quote = fetch_quote(symbol)
    except Exception:
        quote = None

    price_panel = build_dashboard(symbol, interval, quote=quote)
    live_price = quote["last"] if quote else None
    return Group(price_panel, build_model_panel(state, live_price))


@app.command()
def main(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Continuously refresh the dashboard."
    ),
    refresh: float = typer.Option(
        5.0, "--refresh", "-r", help="Seconds between refreshes in watch mode."
    ),
    interval: str = typer.Option(
        "1m", "--interval", "-i", help="Intraday bar size (finest is 1m)."
    ),
    predict: bool = typer.Option(
        True, "--predict/--no-predict", help="Show the XGBoost prediction panel."
    ),
    model_refresh: float = typer.Option(
        900.0,
        "--model-refresh",
        help="Seconds between XGBoost retrains in watch mode (default 15 min).",
    ),
    no_garch: bool = typer.Option(
        False, "--no-garch", help="Train XGBoost without GARCH volatility features."
    ),
) -> None:
    """Show a live dashboard for SYMBOL, with an XGBoost forecast below it."""
    symbol = symbol.strip().upper()
    if not symbol:
        console.print("[bold red]Please provide a ticker symbol.[/bold red]")
        raise typer.Exit(code=1)

    use_garch = not no_garch

    if not watch:
        state = None
        if predict:
            console.print("[dim]Fitting XGBoost…[/dim]")
            try:
                state = fit_model(symbol, use_garch=use_garch)
            except ValueError as e:
                console.print(f"[yellow]Model skipped: {e}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Model skipped: {e}[/yellow]")
        console.print(build_view(symbol, state, interval))
        return

    refresh = max(1.0, refresh)
    model_refresh = max(refresh, model_refresh)

    state: Optional[ModelState] = None
    if predict:
        try:
            with console.status("[dim]Fitting XGBoost…[/dim]"):
                state = fit_model(symbol, use_garch=use_garch)
        except Exception as e:
            console.print(f"[yellow]Model skipped: {e}[/yellow]")

    try:
        with Live(
            build_view(symbol, state, interval),
            console=console,
            refresh_per_second=4,
            screen=True,
        ) as live:
            last_fit = time.monotonic()
            while True:
                time.sleep(refresh)
                if predict and (time.monotonic() - last_fit) >= model_refresh:
                    try:
                        state = fit_model(symbol, use_garch=use_garch)
                        last_fit = time.monotonic()
                    except Exception:
                        last_fit = time.monotonic()
                live.update(build_view(symbol, state, interval))
    except KeyboardInterrupt:
        console.print("[dim]Stopped.[/dim]")


if __name__ == "__main__":
    app()

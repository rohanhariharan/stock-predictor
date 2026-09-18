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


def build_dashboard(symbol: str, interval: str = "1m") -> Panel:
    """Build the rich dashboard panel for *symbol*."""
    try:
        q = fetch_quote(symbol)
    except ValueError as e:
        return Panel(Text(str(e), style="bold red"), title="Error", border_style="red")

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
        trend, arrow, colour = "flat", "•", "yellow"
    elif change > 0:
        trend, arrow, colour = "up", "▲", "green"
    elif change < 0:
        trend, arrow, colour = "down", "▼", "red"
    else:
        trend, arrow, colour = "flat", "•", "yellow"

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
) -> None:
    """Show a live dashboard for SYMBOL."""
    symbol = symbol.strip().upper()
    if not symbol:
        console.print("[bold red]Please provide a ticker symbol.[/bold red]")
        raise typer.Exit(code=1)

    if not watch:
        console.print(build_dashboard(symbol, interval))
        return

    refresh = max(1.0, refresh)
    try:
        with Live(
            build_dashboard(symbol, interval),
            console=console,
            refresh_per_second=4,
            screen=False,
        ) as live:
            while True:
                time.sleep(refresh)
                live.update(build_dashboard(symbol, interval))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


if __name__ == "__main__":
    app()

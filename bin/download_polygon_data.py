#!/usr/bin/env python3
"""
Download historical 5-minute OHLCV data from Polygon.io.

Downloads 2 years of 5-minute bars for a list of symbols and saves them
in the FoxML-expected parquet format:

    data/data_labeled_v2/interval=5m/symbol=AAPL/AAPL.parquet

Usage:
    export POLYGON_API_KEY=your_key_here

    # Download defaults (top 20 stocks, 2 years, 5-minute bars)
    python bin/download_polygon_data.py

    # Custom symbols and date range
    python bin/download_polygon_data.py --symbols AAPL MSFT GOOGL --years 1

    # Custom output directory
    python bin/download_polygon_data.py --output-dir data/my_data/interval=5m

    # Dry run (show what would be downloaded)
    python bin/download_polygon_data.py --dry-run

Requires:
    - POLYGON_API_KEY environment variable
    - requests, pandas, pyarrow packages

Rate Limits:
    - Free tier: 5 requests/minute (script auto-throttles)
    - Paid tier: set --rate-limit 0.1 for faster downloads
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Top 20 US stocks by market cap (as of early 2026)
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK.B", "AVGO", "JPM",
    "LLY", "V", "UNH", "MA", "XOM",
    "COST", "HD", "PG", "JNJ", "ABBV",
]

BASE_URL = "https://api.polygon.io"


def fetch_bars(
    symbol: str,
    start: str,
    end: str,
    api_key: str,
    multiplier: int = 5,
    timespan: str = "minute",
) -> list[dict]:
    """
    Fetch all bars for a symbol in a date range, handling pagination.

    Polygon returns max 50,000 results per request. For 5-min bars over
    2 years (~40,000 bars), one request usually suffices, but we paginate
    to be safe.
    """
    all_results = []
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}"
        f"/{start}/{end}"
    )
    params = {
        "apiKey": api_key,
        "limit": 50000,
        "sort": "asc",
    }

    while url:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("  Request failed for %s: %s", symbol, e)
            break

        if data.get("status") not in ("OK", "DELAYED"):
            logger.warning("  Unexpected status for %s: %s", symbol, data.get("status"))
            break

        results = data.get("results", [])
        all_results.extend(results)

        # Polygon uses "next_url" for pagination
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            # next_url already has params, but needs api key
            params = {"apiKey": api_key}
        else:
            url = None

    return all_results


def bars_to_dataframe(results: list[dict], symbol: str) -> pd.DataFrame:
    """Convert Polygon bar results to a FoxML-compatible DataFrame."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Polygon column mapping: o/h/l/c/v/t/vw/n
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "ts",
        "vw": "vwap",
        "n": "trades",
    })

    # Convert timestamp (milliseconds since epoch) to UTC datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    # Filter to regular trading hours (9:30-16:00 ET)
    # Convert to ET for filtering, keep UTC for storage
    ts_et = df["ts"].dt.tz_convert("America/New_York")
    market_open = ts_et.dt.time >= pd.Timestamp("09:30").time()
    market_close = ts_et.dt.time < pd.Timestamp("16:00").time()
    weekday = ts_et.dt.weekday < 5
    df = df[market_open & market_close & weekday].copy()

    if df.empty:
        return df

    # Add symbol column
    df["symbol"] = symbol

    # Sort by timestamp, drop duplicates
    df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)

    # Keep standard columns
    keep_cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    extra = [c for c in ["vwap", "trades"] if c in df.columns]
    return df[keep_cols + extra]


def download_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    api_key: str,
    rate_limit: float,
) -> bool:
    """Download and save data for a single symbol. Returns True on success."""
    logger.info("Downloading %s (%s to %s)...", symbol, start_date, end_date)

    results = fetch_bars(symbol, start_date, end_date, api_key)
    time.sleep(rate_limit)  # Rate limit between symbols

    if not results:
        logger.warning("  No data returned for %s", symbol)
        return False

    df = bars_to_dataframe(results, symbol)

    if df.empty:
        logger.warning("  No RTH bars for %s after filtering", symbol)
        return False

    # Save to FoxML directory structure
    symbol_dir = output_dir / f"symbol={symbol}"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    out_path = symbol_dir / f"{symbol}.parquet"

    df.to_parquet(out_path, index=False, engine="pyarrow")

    n_days = df["ts"].dt.date.nunique()
    logger.info(
        "  Saved %s: %d bars across %d trading days (%.1f MB)",
        out_path.name,
        len(df),
        n_days,
        out_path.stat().st_size / 1e6,
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download historical 5-min OHLCV data from Polygon.io",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Symbols to download (default: top 20 by market cap)",
    )
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Years of history to download (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/data_labeled_v2/interval=5m",
        help="Output directory (default: data/data_labeled_v2/interval=5m)",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=12.0,
        help="Seconds between API requests (default: 12 for free tier; use 0.1 for paid)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        sys.exit(1)

    # Date range
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=int(args.years * 365))).strftime("%Y-%m-%d")

    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Polygon.io Historical Data Download")
    logger.info("=" * 60)
    logger.info("Symbols:    %d (%s ... %s)", len(args.symbols), args.symbols[0], args.symbols[-1])
    logger.info("Date range: %s to %s", start_date, end_date)
    logger.info("Interval:   5-minute bars (RTH only)")
    logger.info("Output:     %s", output_dir)
    logger.info("Rate limit: %.1fs between requests", args.rate_limit)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN — no data will be downloaded")
        for sym in args.symbols:
            logger.info("  Would download: %s → %s/symbol=%s/%s.parquet", sym, output_dir, sym, sym)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []
    t0 = time.time()

    for i, symbol in enumerate(args.symbols, 1):
        logger.info("[%d/%d]", i, len(args.symbols))
        if download_symbol(symbol, start_date, end_date, output_dir, api_key, args.rate_limit):
            success += 1
        else:
            failed.append(symbol)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done in %.0fs — %d/%d symbols downloaded", elapsed, success, len(args.symbols))
    if failed:
        logger.warning("Failed: %s", ", ".join(failed))
    logger.info("Data saved to: %s", output_dir)


if __name__ == "__main__":
    main()

"""Update project-local Qlib CN daily data with AkShare/Sina daily bars."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import akshare as ak
import numpy as np
import pandas as pd


FIELDS = ("open", "close", "high", "low", "volume", "vwap", "change", "factor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="data/cn_data_rolling")
    parser.add_argument("--end-date", default="2026-07-01")
    parser.add_argument("--markets", nargs="+", default=["csi100", "csi300", "csi500"])
    parser.add_argument("--start-date", default=None, help="Override update start date, YYYY-MM-DD.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--socket-timeout", type=int, default=20)
    return parser.parse_args()


def qlib_symbol_to_ak(symbol: str) -> str:
    return symbol.lower()


def read_calendar(calendar_path: Path) -> List[pd.Timestamp]:
    return [pd.Timestamp(line.strip()) for line in calendar_path.read_text().splitlines() if line.strip()]


def write_calendar(calendar_path: Path, dates: Iterable[pd.Timestamp]) -> None:
    ordered = sorted({pd.Timestamp(d).normalize() for d in dates})
    calendar_path.write_text("\n".join(d.strftime("%Y-%m-%d") for d in ordered) + "\n")


def read_market_symbols(instruments_dir: Path, markets: List[str]) -> List[str]:
    symbols = []
    for market in markets:
        path = instruments_dir / f"{market}.txt"
        for line in path.read_text().splitlines():
            if line.strip():
                symbols.append(line.split("\t")[0].strip().upper())
    return sorted(set(symbols))


def update_instrument_end_dates(instruments_dir: Path, markets: List[str], end_date: str) -> None:
    for market in markets:
        path = instruments_dir / f"{market}.txt"
        lines = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                parts[2] = end_date
            lines.append("\t".join(parts))
        path.write_text("\n".join(lines) + "\n")


def load_bin(path: Path) -> Tuple[int, np.ndarray]:
    if not path.exists():
        return 0, np.array([], dtype=np.float32)
    arr = np.fromfile(path, dtype="<f")
    if arr.size == 0:
        return 0, np.array([], dtype=np.float32)
    return int(arr[0]), arr[1:].astype(np.float32, copy=False)


def write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.hstack([[float(start_index)], values.astype(np.float32)])
    arr.astype("<f").tofile(path)


def append_or_create_feature(path: Path, global_start: int, values: np.ndarray) -> None:
    old_start, old_values = load_bin(path)
    if old_values.size == 0:
        write_bin(path, global_start, values)
        return

    old_end = old_start + old_values.size - 1
    new_end = global_start + values.size - 1
    start = min(old_start, global_start)
    end = max(old_end, new_end)
    merged = np.full(end - start + 1, np.nan, dtype=np.float32)
    merged[old_start - start: old_start - start + old_values.size] = old_values
    merged[global_start - start: global_start - start + values.size] = values
    write_bin(path, start, merged)


def fetch_daily(symbol: str, start_date: str, end_date: str, retry: int) -> pd.DataFrame:
    ak_symbol = qlib_symbol_to_ak(symbol)
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    last_error = None
    for attempt in range(retry + 1):
        try:
            df = ak.stock_zh_a_daily(symbol=ak_symbol, start_date=start, end_date=end, adjust="")
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            return df
        except Exception as exc:  # noqa: BLE001 - keep updater resilient
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{symbol} fetch failed: {last_error}")


def build_feature_arrays(df: pd.DataFrame, dates: List[pd.Timestamp]) -> Dict[str, np.ndarray]:
    by_date = df.set_index("date").reindex(dates)
    amount = pd.to_numeric(by_date.get("amount"), errors="coerce")
    volume = pd.to_numeric(by_date.get("volume"), errors="coerce")
    vwap = amount / volume.replace(0, np.nan)

    close = pd.to_numeric(by_date.get("close"), errors="coerce")
    pct_change = close.pct_change(fill_method=None)
    if "turnover" in by_date.columns:
        # Keep qlib's `change` as a return-like field; existing factor code does not use it.
        change = pct_change
    else:
        change = pct_change

    arrays = {
        "open": pd.to_numeric(by_date.get("open"), errors="coerce").to_numpy(np.float32),
        "close": close.to_numpy(np.float32),
        "high": pd.to_numeric(by_date.get("high"), errors="coerce").to_numpy(np.float32),
        "low": pd.to_numeric(by_date.get("low"), errors="coerce").to_numpy(np.float32),
        "volume": volume.to_numpy(np.float32),
        "vwap": vwap.to_numpy(np.float32),
        "change": change.to_numpy(np.float32),
        "factor": np.ones(len(dates), dtype=np.float32),
    }
    return arrays


def update_one_symbol(
    provider_uri: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    append_dates: List[pd.Timestamp],
    append_start_index: int,
    retry: int,
) -> Dict[str, object]:
    df = fetch_daily(symbol, start_date, end_date, retry)
    feature_dir = provider_uri / "features" / symbol.lower()
    if df.empty:
        values = {field: np.full(len(append_dates), np.nan, dtype=np.float32) for field in FIELDS}
    else:
        values = build_feature_arrays(df, append_dates)

    for field in FIELDS:
        append_or_create_feature(feature_dir / f"{field}.day.bin", append_start_index, values[field])

    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "first": None if df.empty else str(df["date"].min().date()),
        "last": None if df.empty else str(df["date"].max().date()),
        "nan_close": int(np.isnan(values["close"]).sum()),
    }


def main() -> None:
    args = parse_args()
    socket.setdefaulttimeout(args.socket_timeout)

    provider_uri = Path(args.provider_uri).expanduser().resolve()
    calendar_path = provider_uri / "calendars" / "day.txt"
    instruments_dir = provider_uri / "instruments"
    log_dir = provider_uri / "update_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    old_calendar = read_calendar(calendar_path)
    old_last = old_calendar[-1]
    end_ts = pd.Timestamp(args.end_date).normalize()
    if end_ts <= old_last:
        print(json.dumps({"status": "already_up_to_date", "old_last": str(old_last.date())}, ensure_ascii=False))
        return

    trade_dates = ak.tool_trade_date_hist_sina()
    all_trade_dates = sorted(pd.to_datetime(trade_dates["trade_date"]).dt.normalize().tolist())
    new_dates = [d for d in all_trade_dates if old_last < d <= end_ts]
    if not new_dates:
        raise RuntimeError(f"No trade dates found between {old_last.date()} and {end_ts.date()}")

    append_start_index = len(old_calendar)
    write_calendar(calendar_path, old_calendar + new_dates)

    start_date = args.start_date or new_dates[0].strftime("%Y-%m-%d")
    symbols = read_market_symbols(instruments_dir, args.markets)
    print(f"provider={provider_uri}")
    print(f"calendar: {old_last.date()} -> {new_dates[-1].date()}, append_days={len(new_dates)}")
    print(f"symbols={len(symbols)}, markets={args.markets}, workers={args.workers}")

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                update_one_symbol,
                provider_uri,
                symbol,
                start_date,
                args.end_date,
                new_dates,
                append_start_index,
                args.retry,
            ): symbol
            for symbol in symbols
        }
        for idx, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                result = future.result()
                results.append(result)
                if idx % 50 == 0 or idx == len(futures):
                    print(f"updated {idx}/{len(futures)} latest={symbol}")
            except Exception as exc:  # noqa: BLE001
                failures.append({"symbol": symbol, "error": repr(exc)})
                feature_dir = provider_uri / "features" / symbol.lower()
                for field in FIELDS:
                    append_or_create_feature(
                        feature_dir / f"{field}.day.bin",
                        append_start_index,
                        np.full(len(new_dates), np.nan, dtype=np.float32),
                    )
                print(f"FAILED {symbol}: {exc}")

    update_instrument_end_dates(instruments_dir, args.markets, args.end_date)

    summary = {
        "provider_uri": str(provider_uri),
        "end_date": args.end_date,
        "append_days": len(new_dates),
        "append_start_index": append_start_index,
        "symbols": len(symbols),
        "success": len(results),
        "failures": len(failures),
        "markets": args.markets,
    }
    (log_dir / f"update_summary_{args.end_date}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (log_dir / f"update_results_{args.end_date}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    (log_dir / f"update_failures_{args.end_date}.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

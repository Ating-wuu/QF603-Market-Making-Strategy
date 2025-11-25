# every time data/resampled has new data, need to rerun this file
# it takes some time to complete (~4min for 7 days data for 1 trading pair)

import re
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/resampled")
OUT_DIR  = Path("data") / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)


PATTERN = re.compile(r'^(?P<symbol>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.csv\.gz$')

def read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )
    return df

def make_ts_column(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if "exchTimeMs" in df.columns:
        ms = df["exchTimeMs"]
    elif "timeMs" in df.columns:
        ms = df["timeMs"]
    else:
        raise ValueError(f"[{symbol}] missing both exchTimeMs and timeMs columns")

    ms_num = pd.to_numeric(ms, errors="coerce")
    ts = pd.to_datetime(ms_num, unit="ms", utc=True, errors="coerce")

    good = ts.notna()
    dropped = int((~good).sum())
    if dropped > 0:
        print(f"[WARN] {symbol}: drop {dropped} rows with bad timestamp")

    df = df.loc[good].copy()
    df["ts"] = ts.loc[good].values
    return df

def main():
    groups = {}
    for p in DATA_DIR.glob("*.csv.gz"):
        m = PATTERN.match(p.name)
        if not m:
            print(f"[SKIP] filename not matched: {p.name}")
            continue
        symbol = m.group("symbol")
        date_str = m.group("date")
        groups.setdefault(symbol, []).append((date_str, p))

    if not groups:
        print("No files found. Check your data/resampled directory.")
        return

    for symbol, items in groups.items():
        items.sort(key=lambda x: x[0])
        parts = []
        total_rows = 0

        print(f"\n=== Merging {symbol} ({len(items)} files) ===")
        for date_str, path in items:
            try:
                df = read_one(path)
                total_rows += len(df)
                df["file_date"] = date_str
                parts.append(df)
                print(f"  + {path.name}: {len(df):,} rows")
            except Exception as e:
                print(f"[ERROR] reading {path}: {e}")

        if not parts:
            print(f"[WARN] {symbol}: no valid parts, skipped.")
            continue

        merged = pd.concat(parts, ignore_index=True)

        try:
            merged = make_ts_column(merged, symbol)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e} -> skipped.")
            continue

        if merged.empty:
            print(f"[WARN] {symbol}: empty after cleaning, skipped.")
            continue

        merged = merged.sort_values(["ts"]).drop_duplicates(subset=["ts"]).reset_index(drop=True)

        out_path = OUT_DIR / f"{symbol}.csv.gz"
        merged.to_csv(out_path, index=False)
        print(f"[DONE] {symbol}: {len(merged):,} rows (from {total_rows:,}) -> {out_path}")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()

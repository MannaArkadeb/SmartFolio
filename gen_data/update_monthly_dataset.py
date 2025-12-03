"""Utilities for incrementally extending the default dataset month by month.

This module loads the latest raw snapshot stored under ``dataset_default``,
fetches the next month's worth of OHLCV data from yfinance, recomputes the
features and correlation matrices for that month, and persists the processed
samples as a compact "monthly shard". A lightweight manifest keeps track of
the last processed trading day, which monthly shard contains each date, and
which correlation matrices have already been generated so the script avoids
duplicating work on subsequent runs.

Example
-------
```
python -m gen_data.update_monthly_dataset --market us --tickers-file tickers.csv
```
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import MonthBegin, MonthEnd
from torch.autograd import Variable
from torch_geometric.data import Data

try:  # pragma: no cover - optional relative import when executed as module
    from .build_dataset_yf import (  # type: ignore
        DATASET_CORR_ROOT,
        DATASET_DEFAULT_ROOT,
        FEATURE_COLS,
        FEATURE_COLS_NORM,
        cal_rolling_mean_std,
        compute_monthly_corrs,
        fetch_ohlcv_yf,
        filter_code,
        gen_mats_by_threshold,
        get_label,
        group_and_norm,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from build_dataset_yf import (  # type: ignore
        DATASET_CORR_ROOT,
        DATASET_DEFAULT_ROOT,
        FEATURE_COLS,
        FEATURE_COLS_NORM,
        cal_rolling_mean_std,
        compute_monthly_corrs,
        fetch_ohlcv_yf,
        filter_code,
        gen_mats_by_threshold,
        get_label,
        group_and_norm,
    )


MANIFEST_NAME = "monthly_manifest.json"


def _load_manifest(path: str) -> Dict[str, object]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                raise RuntimeError(f"Manifest at {path} is corrupt. Please inspect or delete it.")
    return {
        "last_trading_day": None,
        "monthly_shards": {},
        "daily_index": {},
        "corr_matrices": [],
        "tickers": [],
        "raw_snapshot": None,
    }


def _dump_manifest(path: str, manifest: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _find_latest_raw_snapshot(root: str) -> str:
    """Locate the latest parquet or pickle snapshot under dataset_default."""

    candidates: List[Tuple[float, str]] = []

    search_dirs = [root]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if not fname.lower().endswith((".parquet", ".pkl")):
                continue
            full = os.path.join(directory, fname)
            candidates.append((os.path.getmtime(full), full))

    if not candidates:
        # Fallback: try converting a CSV snapshot (e.g., custom_org.csv) to parquet
        csv_candidates = []
        for directory in search_dirs:
            if not os.path.isdir(directory):
                continue
            for fname in os.listdir(directory):
                if fname.lower().endswith(".csv") and "org" in fname.lower():
                    csv_candidates.append(os.path.join(directory, fname))
        if not csv_candidates:
            raise FileNotFoundError(
                "Unable to locate a parquet/pickle snapshot under dataset_default."
            )
        csv_candidates.sort()
        csv_path = csv_candidates[-1]
        df_csv = pd.read_csv(csv_path)
        required_cols = {"kdcode", "dt", "close", "open", "high", "low", "prev_close", "volume"}
        missing = required_cols - set(df_csv.columns)
        if missing:
            raise ValueError(
                f"CSV snapshot at {csv_path} missing required columns: {', '.join(sorted(missing))}"
            )
        parquet_path = os.path.join(os.path.dirname(csv_path), "custom_latest.parquet")
        df_csv.to_parquet(parquet_path, index=False)
        print(f"Converted CSV snapshot {csv_path} to parquet {parquet_path}")
        return parquet_path

    candidates.sort()
    return candidates[-1][1]


def _load_snapshot(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        obj = pickle.load(open(path, "rb"))
        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, Mapping) and "data" in obj:
            df = pd.DataFrame(obj["data"])
        else:
            raise ValueError(
                "Unsupported pickle format for raw snapshot. Expected a pandas DataFrame "
                f"or a mapping with a 'data' key. Got type: {type(obj)!r}."
            )

    required_cols = {"kdcode", "dt", "close", "open", "high", "low", "prev_close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Snapshot is missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    return df


def _find_latest_existing_date(data_dir: str) -> Optional[pd.Timestamp]:
    """Inspect existing daily/monthly pkls to find the latest available date."""
    latest: Optional[pd.Timestamp] = None

    # Daily pkls in base_dir (YYYY-MM-DD.pkl)
    for fname in os.listdir(data_dir):
        if not fname.endswith(".pkl") or fname == MANIFEST_NAME:
            continue
        stem = os.path.splitext(fname)[0]
        try:
            dt = pd.to_datetime(stem)
        except Exception:
            continue
        if latest is None or dt > latest:
            latest = dt

    # Monthly pkls (legacy bundles)
    monthly_dir = os.path.join(data_dir, "monthly")
    if os.path.isdir(monthly_dir):
        for fname in os.listdir(monthly_dir):
            if not fname.endswith(".pkl"):
                continue
            shard_path = os.path.join(monthly_dir, fname)
            try:
                cache = pickle.load(open(shard_path, "rb"))
                dates = cache.get("dates") if isinstance(cache, dict) else None
            except Exception:
                continue
            if dates:
                try:
                    dt_max = pd.to_datetime(max(dates))
                except Exception:
                    continue
                if latest is None or dt_max > latest:
                    latest = dt_max
    return latest


def _determine_next_month(
    manifest: Mapping[str, object],
    df: pd.DataFrame,
    latest_existing: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last calendar day of the next month to process."""

    last_day_str = manifest.get("last_trading_day")
    if last_day_str:
        last_dt = pd.to_datetime(last_day_str)
    elif latest_existing is not None:
        last_dt = latest_existing
    else:
        last_dt = pd.to_datetime(df["dt"].max())

    next_month_start = (last_dt + MonthBegin(1)).normalize()
    next_month_end = next_month_start + MonthEnd(1)
    return next_month_start, next_month_end


def _ensure_corr(
    df: pd.DataFrame,
    market: str,
    lookback: int,
    corr_root: str,
    month_end_dt: pd.Timestamp,
) -> str:
    """Ensure correlation CSV for the month exists and return its path."""

    rel_dt = (month_end_dt + MonthEnd(0)).strftime("%Y-%m-%d")
    corr_csv = os.path.join(corr_root, market, f"{rel_dt}.csv")
    if os.path.exists(corr_csv):
        return corr_csv

    os.makedirs(os.path.dirname(corr_csv), exist_ok=True)
    # Restrict to a window that still covers the month but keeps runtime short
    window_start = month_end_dt - pd.Timedelta(days=lookback * 3)
    df_subset = df[df["dt"] >= window_start.strftime("%Y-%m-%d")]
    compute_monthly_corrs(df_subset, market=market, lookback_days=lookback, out_root=corr_root)
    return corr_csv


def _build_daily_sample(
    dt: str,
    df_all: pd.DataFrame,
    codes: Sequence[str],
    lookback: int,
    corr_csv: str,
    threshold: float,
    norm: bool = True,
    industry_matrix: Optional[np.ndarray] = None
) -> Optional[Dict[str, torch.Tensor]]:
    """Construct a single day's sample mirroring ``save_daily_graph``."""

    df_all = df_all.copy()
    stock_trade_dt_s_all = sorted(df_all["dt"].unique())
    if dt not in stock_trade_dt_s_all:
        return None

    idx = stock_trade_dt_s_all.index(dt)
    if idx < lookback - 1:
        return None

    ts_start = stock_trade_dt_s_all[idx - (lookback - 1)]
    df_ts = df_all[(df_all["dt"] >= ts_start) & (df_all["dt"] <= dt)].copy()

    if not os.path.exists(corr_csv):
        return None

    corr_df = pd.read_csv(corr_csv, index_col=0)
    corr_df = corr_df.reindex(index=codes, columns=codes).fillna(0)
    pos_adj, neg_adj = gen_mats_by_threshold(corr_df, threshold)

    corr = torch.from_numpy(corr_df.values.astype(np.float32))
    pos = torch.from_numpy(pos_adj.astype(np.float32))
    neg = torch.from_numpy(neg_adj.astype(np.float32))

    ts_features: List[np.ndarray] = []
    features: List[np.ndarray] = []
    labels: List[float] = []

    cols = FEATURE_COLS_NORM if norm else FEATURE_COLS
    for code in codes:
        df_code = df_ts[df_ts["kdcode"] == code].copy()
        df_code = df_code.sort_values("dt")
        # Force numeric dtype to avoid object arrays when stacking
        df_code[cols] = df_code[cols].apply(pd.to_numeric, errors="coerce")
        if df_code[cols].isnull().any().any():
            return None

        ts_array = df_code[cols].to_numpy(dtype=np.float32, copy=False)
        current = df_code[df_code["dt"] == dt]
        if ts_array.shape[0] != lookback or current.empty:
            return None

        feature_vec = current.iloc[0][cols].astype(np.float32).to_numpy(copy=False)

        ts_features.append(ts_array)
        features.append(feature_vec)
        labels.append(float(current.iloc[0]["label"]))

    ts_tensor = torch.from_numpy(np.stack(ts_features, axis=0)).float()
    feat_tensor = torch.from_numpy(np.stack(features, axis=0)).float()
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    ind = industry_matrix if industry_matrix is not None else np.zeros((len(codes), len(codes)), dtype=np.float32)
    # Create pyg_data
    edge_index = torch.triu_indices(ind.shape[0], ind.shape[0], offset=1)
    pyg_data = Data(x=features, edge_index=edge_index)
    pyg_data.edge_attr = ind[edge_index[0], edge_index[1]]

    result = {
        "corr": Variable(corr),
        "ts_features": Variable(ts_tensor),
        "features": Variable(feat_tensor),
        "industry_matrix": Variable(torch.from_numpy(ind.astype(np.float32))),
        "pos_matrix": Variable(pos),
        "neg_matrix": Variable(neg),
        "pyg_data" : pyg_data,
        "labels": Variable(label_tensor),
        "mask": [True] * len(labels),
    }

    for key, tensor in list(result.items()):
        if isinstance(tensor, torch.Tensor):
            result[key] = torch.nan_to_num(tensor, nan=0.0)

    return result


def _write_monthly_shard(
    out_dir: str,
    month: str,
    dates: Sequence[str],
    payloads: Sequence[Mapping[str, torch.Tensor]],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{month}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"dates": list(dates), "items": list(payloads)}, f)
    return out_path


def _concat_with_existing(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df_old, df_new], ignore_index=True)
    df = df.sort_values(["kdcode", "dt"]).drop_duplicates(subset=["kdcode", "dt"], keep="last")
    return df.reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root or DATASET_DEFAULT_ROOT
    corr_root = args.corr_root or DATASET_CORR_ROOT
    print(dataset_root, corr_root)

    data_dir = os.path.join(
        dataset_root,
        f"data_train_predict_{args.market}",
        f"{args.horizon}_{args.relation_type}",
    )
    os.makedirs(data_dir, exist_ok=True)

    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    manifest_raw = _load_manifest(manifest_path)
    manifest = cast(Dict[str, object], manifest_raw)

    raw_snapshot = cast(Optional[str], manifest.get("raw_snapshot"))
    raw_path = args.raw_path or raw_snapshot or _find_latest_raw_snapshot(dataset_root)
    print(f"Using raw snapshot at: {raw_path}")
    df_raw = _load_snapshot(raw_path)

    tickers: List[str]
    if args.tickers_file and os.path.exists(args.tickers_file):
        tickers = [line.strip() for line in open(args.tickers_file, "r", encoding="utf-8") if line.strip()]
    else:
        tickers = sorted(df_raw["kdcode"].unique().tolist())

    latest_existing = _find_latest_existing_date(data_dir)
    next_month_start, next_month_end = _determine_next_month(manifest, df_raw, latest_existing)
    print(f"Next month to process: {next_month_start.strftime('%Y-%m')}")
    month_tag = next_month_start.strftime("%Y-%m")

    monthly_shards = cast(Dict[str, str], manifest.setdefault("monthly_shards", {}))
    if month_tag in monthly_shards:
        print(f"Month {month_tag} already processed. Nothing to do.")
        return

    yf_start = (next_month_start - pd.Timedelta(days=args.lookback * 2)).strftime("%Y-%m-%d")
    yf_end = (next_month_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching OHLCV data via yfinance from {yf_start} to {yf_end} for {len(tickers)} tickers.")
    fetched = fetch_ohlcv_yf(tickers, start=yf_start, end=yf_end)
    if fetched.empty:
        print("No new data fetched; aborting update.")
        return

    df_combined = _concat_with_existing(df_raw, fetched)

    df_with_label = get_label(df_combined, horizon=args.horizon)
    df_stats = cal_rolling_mean_std(df_with_label, FEATURE_COLS, lookback=args.lookback)
    df_norm = group_and_norm(df_stats, FEATURE_COLS, n_clusters=args.n_clusters)

    mask_month = (df_norm["dt"] >= next_month_start.strftime("%Y-%m-%d")) & (
        df_norm["dt"] <= next_month_end.strftime("%Y-%m-%d")
    )
    df_month = df_norm[mask_month].copy()
    if df_month.empty:
        print(f"No rows available for month {month_tag} after preprocessing; aborting.")
        return

    month_dates = sorted(df_month["dt"].unique().tolist())
    month_end_trade = pd.to_datetime(month_dates[-1])
    relation_dt = (month_end_trade + MonthEnd(0)).strftime("%Y-%m-%d")
    corr_csv = _ensure_corr(df_norm, args.market, args.lookback, corr_root, month_end_trade)

    codes = filter_code(df_norm[df_norm["dt"] <= relation_dt])
    if not codes:
        codes = sorted(df_month["kdcode"].unique().tolist())

    payloads = []
    valid_dates = []
    ind = np.load(os.path.join(DATASET_DEFAULT_ROOT, args.market,"industry.npy"))
    for dt in month_dates:
        sample = _build_daily_sample(
            dt=dt,
            df_all=df_norm,
            codes=codes,
            lookback=args.lookback,
            corr_csv=corr_csv,
            threshold=args.threshold,
            norm=not args.disable_norm,
            industry_matrix=ind,
        )
        if sample is None:
            continue
        payloads.append(sample)
        valid_dates.append(dt)

    if not payloads:
        print(f"All samples for {month_tag} were filtered out; aborting.")
        return

    # Write individual daily pickle files (one per trading day)
    daily_index = cast(Dict[str, str], manifest.setdefault("daily_index", {}))
    for dt, payload in zip(valid_dates, payloads):
        daily_fname = f"{dt}.pkl"
        daily_path = os.path.join(data_dir, daily_fname)
        with open(daily_path, "wb") as f:
            pickle.dump(payload, f)
        daily_index[dt] = daily_fname

    # Record month metadata (dates, start/end) for downstream selection
    monthly_shards[month_tag] = {
        "month": month_tag,
        "dates": valid_dates,
        "month_start": valid_dates[0],
        "month_end": valid_dates[-1],
        "shard_type": "daily_files",
        "processed": False,
    }
    manifest["last_trading_day"] = valid_dates[-1]
    corr_matrices = cast(List[str], manifest.setdefault("corr_matrices", []))
    if relation_dt not in corr_matrices:
        corr_matrices.append(relation_dt)

    existing_tickers = cast(List[str], manifest.get("tickers", []))
    manifest["tickers"] = sorted(set(existing_tickers) | set(codes))
    manifest["raw_snapshot"] = raw_path
    # Persist the dataset base directory (and monthly subdir) for downstream consumers
    manifest["base_dir"] = data_dir
    manifest["monthly_dir"] = os.path.join(data_dir, "monthly")

    _dump_manifest(manifest_path, manifest)
    print(f"Saved daily files for {month_tag} with {len(valid_dates)} trading days into {data_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incrementally update monthly dataset shards.")
    parser.add_argument("--market", required=True, help="Market identifier used in dataset path.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon used for labels.")
    parser.add_argument("--relation-type", default="hy", help="Relation type directory suffix (default: hy).")
    parser.add_argument("--lookback", type=int, default=20, help="Rolling window size for features and correlations.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for positive/negative adjacency matrices.")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of KMeans clusters for normalization stage.")
    parser.add_argument("--disable-norm", action="store_true", help="Use raw OHLCV instead of normalized features.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset_default root directory.")
    parser.add_argument("--corr-root", default=None, help="Override correlation output directory.")
    parser.add_argument("--raw-path", default=None, help="Explicit path to the latest raw parquet/pickle snapshot.")
    parser.add_argument("--tickers-file", default="tickers.csv", help="Optional newline-delimited file with tickers to update.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI wrapper
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

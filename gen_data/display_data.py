import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optional dependency; this file is meant to be run as a standalone data builder
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


TREND_LOOKBACK_DAYS = 21  # ~1 trading month
FEATURE_COLS = [
    "close",
    "open",
    "high",
    "low",
    "prev_close",
    "volume",
    "daily_change",
    "trend_1m",
]
FEATURE_COLS_NORM = [f"{c}_normalized" for c in FEATURE_COLS]

# Output root for generated datasets
DISPLAY_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "display_data"))


def fetch_ohlcv_yf(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance and return tall dataframe with required columns.

    Output columns: kdcode, dt, close, open, high, low, prev_close, volume
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please `pip install yfinance` and try again.")

    # yfinance returns a wide MultiIndex df when multiple tickers
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)
    print(df.columns)

    # Normalize to a tall dataframe
    if isinstance(df.columns, pd.MultiIndex):
        parts = []
        for t in tickers:
            if (t, "Close") not in df.columns:
                # ticker may have failed; skip
                continue
            sub = df[(t,)].copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub["kdcode"] = t
            parts.append(sub.reset_index().rename(columns={"Date": "dt"}))
        tall = pd.concat(parts, ignore_index=True)
    else:
        # Single ticker case
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        d["kdcode"] = tickers[0]
        tall = d.reset_index().rename(columns={"Date": "dt"})

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(tall.columns)
    if missing:
        raise ValueError(f"Missing required columns from yfinance output: {missing}")

    # Compute prev_close per kdcode
    tall = tall.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    tall["prev_close"] = tall.groupby("kdcode")["close"].shift(1)

    # Drop first row per ticker where prev_close is NaN; keep consistent entries
    tall = tall.dropna(subset=["prev_close"]).copy()

    # Cast dt to string YYYY-MM-DD to match existing dataset files
    tall["dt"] = pd.to_datetime(tall["dt"]).dt.strftime("%Y-%m-%d")

    # Keep only needed columns
    tall = tall[["kdcode", "dt", "close", "open", "high", "low", "prev_close", "volume"]]
    return tall


def _trend_over_window(window: np.ndarray) -> float:
    """Percent change over the window to represent the month's line trend."""
    if len(window) == 0:
        return np.nan
    start = window[0]
    end = window[-1]
    return (end - start) / (start + 1e-8)


def add_engineered_features(df: pd.DataFrame, trend_lookback: int = TREND_LOOKBACK_DAYS) -> pd.DataFrame:
    """Add daily change % and 1M trend slope-like feature to the OHLCV frame."""
    df = df.copy()
    # Daily price change percentage using prev_close to match label definition
    df["daily_change"] = df["close"] / df["prev_close"] - 1
    # 1-month trend: percent change over the last ~21 trading days, per ticker
    df["trend_1m"] = (
        df.groupby("kdcode")["close"]
        .transform(lambda x: x.rolling(window=trend_lookback, min_periods=trend_lookback).apply(_trend_over_window, raw=True))
    )
    # Drop rows without full feature coverage
    df = df.dropna(subset=["daily_change", "trend_1m"]).reset_index(drop=True)
    return df


def get_label(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = df.copy()
    df.set_index("kdcode", inplace=True)
    for code, group in df.groupby("kdcode"):
        group = group.set_index("dt").sort_index()
        group["return"] = group["close"].shift(-horizon) / group["close"] - 1
        df.loc[code, "label"] = group["return"].values
    df = df.dropna().reset_index()
    return df


def cal_rolling_mean_std(
    df: pd.DataFrame, cal_cols: List[str], lookback: int = 5
) -> pd.DataFrame:
    """Calculate rolling mean and std using pandas."""
    df = df.sort_values(by=["kdcode", "dt"])  # sort by ticker, date
    for col in cal_cols:
        df[f"{col}_mean"] = df.groupby("kdcode")[col].transform(
            lambda x: x.rolling(window=lookback, min_periods=1).mean()
        )
        df[f"{col}_std"] = df.groupby("kdcode")[col].transform(
            lambda x: x.rolling(window=lookback, min_periods=1).std()
        )
    df = df.dropna().reset_index(drop=True)
    return df


def _zscore_safe(series: pd.Series) -> pd.Series:
    """Return z-score, guarding against zero or NaN std."""
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std < 1e-8:
        # If variance is zero, all values are identical; return zeros.
        return series.map(lambda _: 0.0)
    return (series - mean) / std


def group_and_norm(df: pd.DataFrame, base_cols: List[str], n_clusters: int) -> pd.DataFrame:
    result = []
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df = df.sort_values(by=["kdcode", "dt"])  # by ticker/date
    for date, group in df.groupby("dt"):
        group = group.copy()
        cluster_features = group[base_cols].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_features)
        group["cluster"] = kmeans.fit_predict(features_scaled)
        # Merge tiny clusters into nearest
        group_sizes = group["cluster"].value_counts()
        small_clusters = group_sizes[group_sizes < 2].index
        for cl in small_clusters:
            mask = group["cluster"] == cl
            cluster_data = group[mask]
            other_data = group[~mask]
            if other_data.empty or cluster_data.empty:
                continue
            distances = np.linalg.norm(other_data[base_cols].values[:, np.newaxis] - cluster_data[base_cols].values, axis=2)
            closest_cluster_indices = np.argmin(distances, axis=0)
            closest_clusters = other_data.iloc[closest_cluster_indices]["cluster"].values
            group.loc[mask, "cluster"] = closest_clusters
        # Z-score within cluster
        for f in FEATURE_COLS:
            group[f"{f}_normalized"] = group.groupby("cluster")[f].transform(_zscore_safe)
        result.append(group)
    return pd.concat(result)


def main():
    parser = argparse.ArgumentParser(description="Build SmartFolio-compatible dataset from yfinance")
    parser.add_argument("--market", default="custom", help="Market name tag to use in output paths (default: custom)")
    parser.add_argument("--tickers_file", default=None, help="CSV with a 'kdcode' or 'ticker' column listing symbols")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list if no file is provided")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--relation_type", default="hy")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--norm", dest="norm", action="store_true", help="Use normalized features (default)")
    parser.add_argument("--no-norm", dest="norm", action="store_false", help="Disable feature normalization")
    parser.set_defaults(norm=True)
    parser.add_argument("--industry_mode", default="identity", choices=["identity", "full", "sector"], help="How to build industry matrix for non-CN markets")
    args = parser.parse_args()

    # Resolve tickers
    if args.tickers_file:
        df_t = pd.read_csv(args.tickers_file)
        col = "kdcode" if "kdcode" in df_t.columns else ("ticker" if "ticker" in df_t.columns else None)
        if col is None:
            raise ValueError("tickers_file must have a 'kdcode' or 'ticker' column")
        tickers = sorted(df_t[col].dropna().astype(str).unique().tolist())
    elif args.tickers:
        tickers = sorted([t.strip() for t in args.tickers.split(",") if t.strip()])
    else:
        raise ValueError("Provide --tickers_file or --tickers")

    # 1) Download OHLCV
    print(f"Downloading OHLCV for {len(tickers)} tickers from {args.start} to {args.end}...")
    df_raw = fetch_ohlcv_yf(tickers, args.start, args.end)
    trend_window = max(args.lookback, TREND_LOOKBACK_DAYS) if args.lookback else TREND_LOOKBACK_DAYS
    df_raw = add_engineered_features(df_raw, trend_lookback=trend_window)
    if df_raw.empty:
        raise ValueError(
            f"No rows left after computing engineered features. Ensure the date range spans at least {trend_window} trading days."
        )

    # Save the 'org' CSV for reference (mirrors existing naming convention)
    org_out = os.path.join(DISPLAY_DATA_ROOT, f"{args.market}_org.csv")
    os.makedirs(DISPLAY_DATA_ROOT, exist_ok=True)
    df_raw.to_csv(org_out, index=False)

    # --- Create an index CSV (equal-weighted average daily return) for model_predict ---
    # model_predict expects ./display_data/index_data/{market}_index_2024.csv with columns ['datetime','daily_return']
    try:
        idx_dir = os.path.join(DISPLAY_DATA_ROOT, "index_data")
        os.makedirs(idx_dir, exist_ok=True)
        # Use prev_close from df_raw to compute per-ticker daily return
        df_idx = df_raw.copy()
        if "prev_close" in df_idx.columns:
            df_idx["daily_return"] = df_idx["close"] / df_idx["prev_close"] - 1
            df_idx_summary = df_idx.groupby("dt")["daily_return"].mean().reset_index()
            df_idx_summary = df_idx_summary.rename(columns={"dt": "datetime"})
            index_out = os.path.join(idx_dir, f"{args.market}_index.csv")
            df_idx_summary.to_csv(index_out, index=False)
        else:
            # If prev_close missing (shouldn't happen), skip index creation but warn
            print("Warning: prev_close missing in raw data; skipping index CSV creation.")
    except Exception as e:
        print(f"Warning: failed to create index CSV: {e}")

    # 2) Labels and preprocessing
    df_lbl = get_label(df_raw, horizon=args.horizon)
    df_roll = cal_rolling_mean_std(
        df_lbl,
        cal_cols=["close", "volume"],
        lookback=5,
    )
    df_norm = group_and_norm(
        df_roll,
        base_cols=["close_mean", "close_std", "volume_mean", "volume_std"],
        n_clusters=4,
    )

    # dates and codes
    df_all = df_norm.copy()
    df_all = df_all[(df_all["dt"] >= args.start) & (df_all["dt"] <= args.end)].copy()
    stock_trade_dt_s_all = sorted(df_norm["dt"].unique().tolist())
    stock_trade_dt_s = sorted(df_all["dt"].unique().tolist())

    
    print("Dataset build complete.")


if __name__ == "__main__":
    main()
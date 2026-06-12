"""
MLOps Batch Job - Rolling Mean Signal Pipeline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (stderr so stdout stays clean for JSON)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str, logger: logging.Logger) -> dict:
    logger.info(f"Loading config from: {config_path}")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is not a valid YAML mapping.")

    required_fields = {"seed", "window", "version"}
    missing = required_fields - config.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    # Type checks
    if not isinstance(config["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(config['seed'])}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str):
        raise ValueError(f"'version' must be a string, got: {type(config['version'])}")

    logger.info(
        f"Config validated — seed={config['seed']}, "
        f"window={config['window']}, version={config['version']}"
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric, drop rows that can't be parsed
    original_len = len(df)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    dropped = original_len - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with non-numeric 'close' values.")

    if df.empty:
        raise ValueError("No valid rows remain after cleaning 'close' column.")

    logger.info(f"Loaded {len(df)} rows successfully.")
    return df.reset_index(drop=True)


def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Computing rolling mean with window={window}.")
    # NaN for the first (window-1) rows; they are excluded from signal computation
    df["rolling_mean"] = df["close"].rolling(window=window).mean()
    valid = df["rolling_mean"].notna().sum()
    logger.info(
        f"Rolling mean computed. "
        f"Valid rows (window satisfied): {valid} | "
        f"NaN rows (excluded from signal): {window - 1}"
    )
    return df


def compute_signal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Generating binary signal (1 if close > rolling_mean, else 0).")
    # Only compute signal where rolling_mean is not NaN
    mask = df["rolling_mean"].notna()
    df.loc[mask, "signal"] = (df.loc[mask, "close"] > df.loc[mask, "rolling_mean"]).astype(int)
    df["signal"] = df["signal"].fillna(np.nan)  # keep NaN explicit for excluded rows

    signal_series = df.loc[mask, "signal"]
    signal_rate = float(signal_series.mean())
    logger.info(
        f"Signal generation complete. "
        f"Rows with signal: {mask.sum()} | "
        f"Signal rate: {signal_rate:.6f}"
    )
    return df, signal_rate


def write_metrics(output_path: str, payload: dict, logger: logging.Logger):
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Metrics written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MLOps Rolling Mean Signal Pipeline")
    parser.add_argument("--input",   required=True, help="Path to input CSV file")
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--output",  required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("MLOps Pipeline — Job START")
    logger.info("=" * 60)

    start_time = time.time()
    version = "unknown"

    try:
        # Step 1 — Config
        config = load_config(args.config, logger)
        version = config["version"]
        seed    = config["seed"]
        window  = config["window"]

        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        # Step 2 — Dataset
        df = load_dataset(args.input, logger)
        rows_loaded = len(df)

        # Step 3 — Rolling mean
        df = compute_rolling_mean(df, window, logger)

        # Step 4 — Signal
        df, signal_rate = compute_signal(df, logger)
        rows_processed = int(df["rolling_mean"].notna().sum())

        # Step 5 — Metrics
        latency_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        write_metrics(args.output, metrics, logger)

        logger.info("-" * 60)
        logger.info(f"rows_loaded    : {rows_loaded}")
        logger.info(f"rows_processed : {rows_processed}")
        logger.info(f"signal_rate    : {signal_rate:.6f}")
        logger.info(f"latency_ms     : {latency_ms}")
        logger.info(f"status         : success")
        logger.info("=" * 60)
        logger.info("MLOps Pipeline — Job END (success)")
        logger.info("=" * 60)

        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Pipeline failed: {exc}", exc_info=True)

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    latency_ms,
        }

        try:
            write_metrics(args.output, error_metrics, logger)
        except Exception as write_exc:
            logger.error(f"Could not write error metrics: {write_exc}")

        logger.info("=" * 60)
        logger.info("MLOps Pipeline — Job END (failure)")
        logger.info("=" * 60)

        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

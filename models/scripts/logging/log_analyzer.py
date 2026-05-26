"""
LLM inference server request log analyzer.

Reads the CSV request log produced by api_server.py, prints a human-readable
summary, and writes a resampled hourly CSV file suitable as training data for
an RNN / LSTM time-series model that predicts chat-service load.

Usage
-----
    # Activate the virtual environment first
    source scripts/llm_from_scratch/venv/bin/activate

    python scripts/logging/log_analyzer.py

    # With explicit paths:
    python scripts/logging/log_analyzer.py \\
        --csv  logs/requests.csv \\
        --output logs/hourly_load.csv

Input CSV columns (written by api_server.py)
--------------------------------------------
    timestamp, request_id, client_ip, model, messages_json, response,
    prompt_tokens, completion_tokens, latency_ms, status_code

Output hourly CSV columns
-------------------------
    hour                  — UTC hour bucket  (e.g. 2025-06-01T14:00:00Z)
    request_count         — total requests in this hour
    error_count           — requests with status_code >= 400
    avg_latency_ms        — mean end-to-end latency
    p95_latency_ms        — 95th-percentile latency
    avg_prompt_tokens     — mean input token count
    avg_completion_tokens — mean output token count
    total_tokens          — combined tokens (prompt + completion) in this hour

RNN / LSTM usage
----------------
    Load the hourly CSV with pandas, normalise the numeric columns with
    MinMaxScaler, and feed sliding windows of length W into an LSTM cell to
    predict the next hour's request_count.  The latency and token columns
    serve as auxiliary features that carry signal about model utilisation.

    Example (pandas):
        import pandas as pd
        df = pd.read_csv("logs/hourly_load.csv", parse_dates=["hour"])
        df = df.set_index("hour").sort_index()
        # df is now a time-indexed DataFrame ready for window extraction
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_csv(path: str) -> list[dict]:
    """Return all rows from the request log CSV as a list of dicts."""
    p = Path(path)
    if not p.exists():
        print(f"Error: CSV file not found — {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(records: list[dict]) -> None:
    """Print key aggregate statistics to stdout."""
    total = len(records)
    if total == 0:
        print("No records in the log file.")
        return

    errors = sum(
        1 for r in records if _int(r.get("status_code", 200)) >= 400
    )
    latencies = [
        _float(r["latency_ms"]) for r in records if r.get("latency_ms")
    ]
    avg_lat = _mean(latencies)
    p95_lat = _percentile(latencies, 0.95)
    p99_lat = _percentile(latencies, 0.99)

    total_prompt = sum(_int(r.get("prompt_tokens", 0)) for r in records)
    total_completion = sum(
        _int(r.get("completion_tokens", 0)) for r in records
    )

    print("=" * 54)
    print("  LLM Inference Server — Request Log Summary")
    print("=" * 54)
    print(f"  Total requests       : {total:,}")
    print(f"  Error responses      : {errors:,}  ({100 * errors / total:.1f} %)")
    print(f"  Avg latency          : {avg_lat:.1f} ms")
    print(f"  P95 latency          : {p95_lat:.1f} ms")
    print(f"  P99 latency          : {p99_lat:.1f} ms")
    print(f"  Total prompt tokens  : {total_prompt:,}")
    print(f"  Total output tokens  : {total_completion:,}")
    print("=" * 54)


# ---------------------------------------------------------------------------
# Hourly resampling
# ---------------------------------------------------------------------------


def resample_hourly(records: list[dict], output_path: str) -> None:
    """
    Aggregate request records by UTC hour and write a resampled CSV.

    Each row in the output represents one clock-hour and contains aggregate
    metrics that can be used as a feature vector for a time-series model.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for rec in records:
        ts_str = rec.get("timestamp", "")
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            hour_key = dt.strftime("%Y-%m-%dT%H:00:00Z")
        except ValueError:
            skipped += 1
            continue
        buckets[hour_key].append(rec)

    if skipped:
        print(
            f"Warning: skipped {skipped} record(s) with unparseable timestamps.",
            file=sys.stderr,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "hour",
        "request_count",
        "error_count",
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "total_tokens",
    ]

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for hour in sorted(buckets):
            recs = buckets[hour]
            count = len(recs)
            errors = sum(
                1 for r in recs if _int(r.get("status_code", 200)) >= 400
            )
            lats = [
                _float(r["latency_ms"]) for r in recs if r.get("latency_ms")
            ]
            prompt_tok = [
                _int(r["prompt_tokens"])
                for r in recs
                if r.get("prompt_tokens")
            ]
            comp_tok = [
                _int(r["completion_tokens"])
                for r in recs
                if r.get("completion_tokens")
            ]

            writer.writerow(
                {
                    "hour": hour,
                    "request_count": count,
                    "error_count": errors,
                    "avg_latency_ms": round(_mean(lats), 2),
                    "p95_latency_ms": round(_percentile(lats, 0.95), 2),
                    "avg_prompt_tokens": round(_mean(prompt_tok), 2),
                    "avg_completion_tokens": round(_mean(comp_tok), 2),
                    "total_tokens": sum(prompt_tok) + sum(comp_tok),
                }
            )

    hours = len(buckets)
    print(f"\nHourly resampled CSV written to: {out}")
    print(
        f"Rows: {hours} hour(s) — load this file into pandas for RNN / LSTM training."
    )


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(p * len(sorted_vals)) - 1)
    return sorted_vals[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and resample the LLM inference server request log."
    )
    parser.add_argument(
        "--csv",
        default="logs/requests.csv",
        help="Path to the raw request CSV log (default: logs/requests.csv)",
    )
    parser.add_argument(
        "--output",
        default="logs/hourly_load.csv",
        help="Output path for the hourly resampled CSV (default: logs/hourly_load.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = load_csv(args.csv)
    print_summary(records)
    resample_hourly(records, args.output)


if __name__ == "__main__":
    main()

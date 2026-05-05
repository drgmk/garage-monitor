from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import garage_monitor as gm


DEFAULT_COLUMNS = [
    "filename",
    "timestamp",
    "path",
    "band_mask_fraction",
    "door_line_resid_ssq",
    "car_poly_resid_ssq",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pc5",
    "global_luma_median",
    "general_pc1",
    "general_pc2",
    "general_pc3",
    "general_pc4",
    "general_pc5",
    "interesting_image_flag",
    "cluster_id",
    "cluster_case",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export candidate garage image filenames for manual annotation from the feature cache."
    )
    parser.add_argument("--config", default="garage_config.yml", help="Path to garage config YAML")
    parser.add_argument("--feature-cache", default=None, help="Optional override path to feature cache pickle")
    parser.add_argument("--table-pickle", default=None, help="Optional generic dataframe pickle to read instead of the feature cache")
    parser.add_argument("--output", required=True, help="Output path (.txt, .csv, or .json)")
    parser.add_argument("--query", default=None, help="Optional pandas query string applied before ranking")
    parser.add_argument("--metric", default=None, help="Optional metric column to sort by")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending")
    parser.add_argument("--top-n", type=int, default=None, help="Keep only the first N rows after filtering/sorting")
    parser.add_argument("--skip-annotated", default=None, help="Optional JSONL annotations file; skip filenames already present there")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Columns to include in CSV/JSON output. Defaults to a useful subset plus metric if needed.",
    )
    parser.add_argument("--show", type=int, default=20, help="Print the first N selected rows")
    return parser.parse_args()


def load_feature_rows(config: dict[str, Any], override: str | None, table_pickle: str | None) -> pd.DataFrame:
    source_path = None
    if table_pickle is not None:
        source_path = Path(table_pickle)
    else:
        source_path = (
            Path(override)
            if override is not None
            else gm.feature_cache_pickle_path(config["data_path"], config["bin_factor"], config["cache_dir_name"])
        )
    if not source_path.exists():
        raise FileNotFoundError(f"Input dataframe not found: {source_path}")
    df = pd.read_pickle(source_path).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "path" in df.columns:
        df["path"] = df["path"].map(str)
    return df


def load_annotated_filenames(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "filename" in record:
            out.add(str(record["filename"]))
    return out


def choose_columns(df: pd.DataFrame, requested: list[str] | None, metric: str | None) -> list[str]:
    cols = list(requested) if requested else list(DEFAULT_COLUMNS)
    if requested is None and not any(col in df.columns for col in ["general_pc1", "pc1"]):
        cols = [c for c in cols if not c.startswith("general_pc") and not (c.startswith("pc") and c[2:].isdigit())]
    if metric and metric not in cols:
        cols.append(metric)
    present = [c for c in cols if c in df.columns]
    if "filename" not in present and "filename" in df.columns:
        present.insert(0, "filename")
    return present


def write_output(df: pd.DataFrame, output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".txt":
        output_path.write_text("\n".join(df["filename"].astype(str).tolist()) + "\n")
        return
    out = df[columns].copy()
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].map(lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat())
    if suffix == ".csv":
        out.to_csv(output_path, index=False)
        return
    if suffix == ".json":
        output_path.write_text(json.dumps(out.to_dict(orient="records"), indent=2))
        return
    raise ValueError(f"Unsupported output format for {output_path}; use .txt, .csv, or .json")


def main() -> None:
    args = parse_args()
    config = gm.load_config(args.config)
    df = load_feature_rows(config, args.feature_cache, args.table_pickle)

    if args.query:
        df = df.query(args.query).copy()

    if args.skip_annotated:
        annotated = load_annotated_filenames(Path(args.skip_annotated))
        if annotated and "filename" in df.columns:
            df = df[~df["filename"].astype(str).isin(annotated)].copy()

    if args.metric:
        if args.metric not in df.columns:
            raise KeyError(f"Metric column not found: {args.metric}")
        df = df.sort_values(args.metric, ascending=args.ascending).copy()
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp").copy()

    if args.top_n is not None:
        df = df.head(args.top_n).copy()

    if df.empty:
        raise ValueError("No candidate rows remain after filtering")

    columns = choose_columns(df, args.columns, args.metric)
    write_output(df, Path(args.output), columns)

    print(f"Selected {len(df)} candidate rows")
    print(f"Wrote {args.output}")
    preview_cols = [c for c in columns if c in df.columns]
    if args.show and preview_cols:
        print(df[preview_cols].head(args.show).to_string(index=False))


if __name__ == "__main__":
    main()

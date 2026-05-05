from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import pandas as pd

import garage_monitor as gm

DEFAULT_LABELS = ["door_open", "car_absent", "garage_light_on"]
DEFAULT_OUTPUT = "garage_annotations.jsonl"


@dataclass
class Candidate:
    filename: str
    path: Path
    timestamp: pd.Timestamp | None
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate garage images with simple labels in a Matplotlib window.")
    parser.add_argument("--config", default="garage_config.yml", help="Path to garage config YAML")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSONL file to store annotations")
    parser.add_argument("--bin-factor", type=int, default=None, help="Override image bin factor for display")
    parser.add_argument("--filenames", nargs="*", default=None, help="Explicit image filenames to annotate")
    parser.add_argument("--filenames-file", default=None, help="Text/CSV/JSON file containing candidate filenames")
    parser.add_argument("--feature-cache", default=None, help="Optional feature cache pickle to resolve timestamps and paths")
    parser.add_argument("--all", action="store_true", help="Annotate all images from the feature cache or data_path")
    parser.add_argument("--include-annotated", action="store_true", help="Do not skip already annotated filenames")
    parser.add_argument("--start-at", default=None, help="Start from this filename if present in the candidate list")
    parser.add_argument("--title-note", default="", help="Optional note shown in the title for every image")
    return parser.parse_args()


def load_feature_rows(config: dict[str, Any], feature_cache_override: str | None) -> pd.DataFrame:
    feature_cache_path = (
        Path(feature_cache_override)
        if feature_cache_override is not None
        else gm.feature_cache_pickle_path(config["data_path"], config["bin_factor"], config["cache_dir_name"])
    )
    if feature_cache_path.exists():
        df = pd.read_pickle(feature_cache_path).copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "path" in df.columns:
            df["path"] = df["path"].map(Path)
        elif "filename" in df.columns:
            df["path"] = df["filename"].map(lambda name: Path(config["data_path"]) / str(name))
        return df

    images, _ = gm.discover_images(
        config["data_path"],
        bin_factor=config["bin_factor"],
        force_rescan=config.get("image_discovery", {}).get("force_rescan_images", False),
        cache_dir_name=config.get("cache_dir_name", gm.DEFAULT_CACHE_DIR_NAME),
    )
    if "timestamp" in images.columns:
        images["timestamp"] = pd.to_datetime(images["timestamp"])
    images["path"] = images["path"].map(Path)
    return images


def load_candidate_filenames(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".txt", ".lst"}:
        return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            out: list[str] = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and "filename" in item:
                    out.append(str(item["filename"]))
            return out
        raise ValueError(f"Unsupported JSON candidate structure in {path}")
    if suffix in {".csv", ".tsv"}:
        sep = "	" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "filename" not in df.columns:
            raise ValueError(f"Candidate table must contain a filename column: {path}")
        return [str(x) for x in df["filename"].dropna().tolist()]
    raise ValueError(f"Unsupported candidate file type: {path}")


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        filename = str(record["filename"])
        annotations[filename] = record
    return annotations


def write_annotations(path: Path, annotations: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for filename in sorted(annotations):
            handle.write(json.dumps(annotations[filename], sort_keys=True) + "\n")


def build_candidates(args: argparse.Namespace, config: dict[str, Any], feature_rows: pd.DataFrame) -> list[Candidate]:
    by_filename = {}
    for row in feature_rows.itertuples():
        filename = str(getattr(row, "filename", Path(row.path).name))
        by_filename[filename] = Candidate(
            filename=filename,
            path=Path(getattr(row, "path")),
            timestamp=getattr(row, "timestamp", None),
            note="",
        )

    ordered_filenames: list[str]
    if args.filenames:
        ordered_filenames = [str(name) for name in args.filenames]
    elif args.filenames_file:
        ordered_filenames = load_candidate_filenames(Path(args.filenames_file))
    elif args.all:
        if "timestamp" in feature_rows.columns:
            ordered_filenames = [str(name) for name in feature_rows.sort_values("timestamp")["filename"].tolist()]
        else:
            ordered_filenames = sorted(by_filename)
    else:
        raise ValueError("Provide --filenames, --filenames-file, or --all")

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for filename in ordered_filenames:
        if filename in seen:
            continue
        seen.add(filename)
        if filename in by_filename:
            candidates.append(by_filename[filename])
            continue
        path = Path(config["data_path"]) / filename
        if path.exists():
            candidates.append(Candidate(filename=filename, path=path, timestamp=None))
            continue
        print(f"Warning: candidate not found in feature cache or data_path: {filename}")
    return candidates


class AnnotatorApp:
    def __init__(
        self,
        candidates: list[Candidate],
        annotations: dict[str, dict[str, Any]],
        output_path: Path,
        config: dict[str, Any],
        *,
        bin_factor: int,
        title_note: str = "",
        start_at: str | None = None,
    ) -> None:
        self.candidates = candidates
        self.annotations = annotations
        self.output_path = output_path
        self.config = config
        self.bin_factor = bin_factor
        self.title_note = title_note
        self.index = 0
        if start_at is not None:
            for i, cand in enumerate(self.candidates):
                if cand.filename == start_at:
                    self.index = i
                    break

        self.current_states = {label: False for label in DEFAULT_LABELS}
        self.fig = plt.figure(figsize=(12, 7))
        self.ax_image = self.fig.add_axes([0.05, 0.08, 0.68, 0.84])
        self.ax_checks = self.fig.add_axes([0.78, 0.48, 0.18, 0.24])
        self.ax_info = self.fig.add_axes([0.76, 0.08, 0.22, 0.32])
        self.ax_info.axis("off")
        self.check = CheckButtons(self.ax_checks, DEFAULT_LABELS, [False] * len(DEFAULT_LABELS))
        self.check.on_clicked(self.on_check_clicked)

        self.ax_prev = self.fig.add_axes([0.76, 0.78, 0.08, 0.06])
        self.ax_next = self.fig.add_axes([0.86, 0.78, 0.08, 0.06])
        self.ax_save = self.fig.add_axes([0.76, 0.86, 0.18, 0.06])
        self.btn_prev = Button(self.ax_prev, "Prev")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_save = Button(self.ax_save, "Save")
        self.btn_prev.on_clicked(lambda event: self.step(-1))
        self.btn_next.on_clicked(lambda event: self.step(1))
        self.btn_save.on_clicked(lambda event: self.save_current())

        self.info_text = self.ax_info.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.refresh()

    def annotation_for(self, filename: str) -> dict[str, Any] | None:
        return self.annotations.get(filename)

    def set_states_from_record(self, record: dict[str, Any] | None) -> None:
        desired = {label: bool(record.get(label, False)) if record else False for label in DEFAULT_LABELS}
        current = self.current_states.copy()
        for idx, label in enumerate(DEFAULT_LABELS):
            if current[label] != desired[label]:
                self.check.set_active(idx)
        self.current_states = desired

    def on_check_clicked(self, label: str) -> None:
        self.current_states[label] = not self.current_states[label]
        self.update_info()

    def current_candidate(self) -> Candidate:
        return self.candidates[self.index]

    def save_current(self) -> None:
        cand = self.current_candidate()
        record = {
            "filename": cand.filename,
            "path": str(cand.path),
            "timestamp": None if cand.timestamp is None or pd.isna(cand.timestamp) else pd.Timestamp(cand.timestamp).isoformat(),
            "door_open": bool(self.current_states["door_open"]),
            "car_absent": bool(self.current_states["car_absent"]),
            "garage_light_on": bool(self.current_states["garage_light_on"]),
            "annotated_at": pd.Timestamp.now().isoformat(),
        }
        self.annotations[cand.filename] = record
        write_annotations(self.output_path, self.annotations)
        self.update_info(saved=True)

    def step(self, delta: int) -> None:
        self.index = max(0, min(len(self.candidates) - 1, self.index + delta))
        self.refresh()

    def on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key == "d":
            self.check.set_active(DEFAULT_LABELS.index("door_open"))
        elif key == "c":
            self.check.set_active(DEFAULT_LABELS.index("car_absent"))
        elif key == "l":
            self.check.set_active(DEFAULT_LABELS.index("garage_light_on"))
        elif key in {"right", "n", "enter", " ", "space"}:
            self.save_current()
            self.step(1)
        elif key in {"left", "p"}:
            self.step(-1)
        elif key == "s":
            self.save_current()
        elif key in {"backspace", "delete", "x"}:
            self.set_states_from_record({})
        elif key == "q":
            plt.close(self.fig)

    def update_info(self, saved: bool = False) -> None:
        cand = self.current_candidate()
        existing = self.annotation_for(cand.filename)
        lines = [
            f"image {self.index + 1}/{len(self.candidates)}",
            f"file: {cand.filename}",
            f"time: {cand.timestamp if cand.timestamp is not None else 'unknown'}",
            f"saved: {'yes' if existing else 'no'}",
            "",
            "keys:",
            "  d = door_open",
            "  c = car_absent",
            "  l = garage_light_on",
            "  s = save",
            "  n/enter/right = save+next",
            "  p/left = prev",
            "  x/backspace = clear",
            "  q = quit",
            "",
            "current:",
        ]
        for label in DEFAULT_LABELS:
            lines.append(f"  {label}: {self.current_states[label]}")
        if saved:
            lines.extend(["", f"saved -> {self.output_path}"])
        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def refresh(self) -> None:
        cand = self.current_candidate()
        self.ax_image.clear()
        gm.show_image(cand.path, bin_factor=self.bin_factor, ax=self.ax_image, cache_dir_name=self.config["cache_dir_name"])
        title = f"{cand.filename}"
        if cand.timestamp is not None and not pd.isna(cand.timestamp):
            title += f" | {pd.Timestamp(cand.timestamp):%Y-%m-%d %H:%M:%S}"
        if self.title_note:
            title += f" | {self.title_note}"
        self.ax_image.set_title(title)
        self.set_states_from_record(self.annotation_for(cand.filename))
        self.update_info()
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main() -> None:
    args = parse_args()
    config = gm.load_config(args.config)
    feature_rows = load_feature_rows(config, args.feature_cache)
    candidates = build_candidates(args, config, feature_rows)
    if not candidates:
        raise ValueError("No candidate images to annotate")

    output_path = Path(args.output)
    annotations = load_annotations(output_path)
    if not args.include_annotated:
        candidates = [cand for cand in candidates if cand.filename not in annotations]
    if not candidates:
        print("No unannotated candidate images remain.")
        return

    bin_factor = args.bin_factor if args.bin_factor is not None else int(config["bin_factor"])
    app = AnnotatorApp(
        candidates,
        annotations,
        output_path,
        config,
        bin_factor=bin_factor,
        title_note=args.title_note,
        start_at=args.start_at,
    )
    app.run()


if __name__ == "__main__":
    main()

"""Garage image feature extraction and threshold-based state reporting."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from PIL import ExifTags, Image
from scipy import ndimage


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CACHE_DIR_NAME = "binned"

EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
EXIF_DATETIME_TAGS = [
    EXIF_TAGS.get("DateTimeOriginal"),
    EXIF_TAGS.get("DateTimeDigitized"),
    EXIF_TAGS.get("DateTime"),
]
EXIF_DATETIME_TAGS = [tag for tag in EXIF_DATETIME_TAGS if tag is not None]

IMAGE_COLUMNS = [
    "path",
    "filename",
    "timestamp_exif",
    "timestamp_mtime",
    "timestamp",
    "timestamp_source",
]

ROI_STATS = ["mean", "median", "std", "p10", "p25", "p75", "p90", "p95"]

BAND_MASK_ROW_DELTA_ABS_FLOOR = 25.0
BAND_MASK_ROW_DELTA_ROBUST_Z = 8.0
BAND_MASK_EVIDENCE_WINDOW = 9
BAND_MASK_EVIDENCE_MIN_COUNT = 1
BAND_MASK_DILATION_RADIUS = 1
BAND_MASK_FILL_GAP = 13
BAND_MASK_MIN_RUN_LEN = 2
BAND_MASK_MIN_FRACTION_TO_APPLY = 0.05
BAND_MASK_LOCAL_WINDOW = 25

DOOR_PROFILE_ROTATE_CW_DEG = -1.0
DOOR_PROFILE_EDGE_DIVISOR = 6
DOOR_PROFILE_EDGE_MARGIN_DIVISOR = 12
DOOR_PROFILE_CENTER1 = (21, 30)
DOOR_PROFILE_CENTER2 = (18, 20)
DOOR_PROFILE_MAX_MASKED_FRACTION = 0.25
CAR_PROFILE_ROTATE_CW_DEG = 40.0
CAR_PROFILE_MAX_MASKED_FRACTION = 0.25
CAR_PROFILE_FIT_DEGREE = 2
FEATURE_CACHE_SCHEMA_VERSION = 6

BIN_CACHE_WARNINGS: list[dict[str, Any]] = []
BIN_CACHE_HITS = 0
BIN_CACHE_MISSES = 0


@dataclass(frozen=True)
class ThresholdRule:
    feature: str
    threshold: float
    higher_means_true: bool = True

    def evaluate(self, value: float) -> bool | None:
        if pd.isna(value):
            return None
        return bool(value > self.threshold) if self.higher_means_true else bool(value < self.threshold)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "data_path" not in config:
        raise ValueError("Config must define data_path")
    base_dir = config_path.resolve().parent
    config["data_path"] = Path(config["data_path"]).expanduser()
    config["bin_factor"] = config.get("bin_factor", 4)
    config["cache_dir_name"] = config.get("cache_dir_name", DEFAULT_CACHE_DIR_NAME)
    config["rois"] = {name: tuple(values) for name, values in config.get("rois", {}).items()}
    general_change = dict(config.get("general_change", {}) or {})
    if general_change.get("model_path"):
        model_path = Path(general_change["model_path"]).expanduser()
        general_change["model_path"] = model_path if model_path.is_absolute() else base_dir / model_path
    config["general_change"] = general_change
    return config

def cache_dir(data_path: str | Path, cache_dir_name: str = DEFAULT_CACHE_DIR_NAME) -> Path:
    return Path(data_path) / cache_dir_name


def is_cached_image_path(path: str | Path, root: str | Path, cache_dir_name: str = DEFAULT_CACHE_DIR_NAME) -> bool:
    path = Path(path)
    root = Path(root)
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return cache_dir_name in parts or re.search(r"_bin\d*$", path.stem) is not None


def feature_cache_pickle_path(
    data_path: str | Path,
    bin_factor: int | None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> Path:
    suffix = f"bin{bin_factor}" if bin_factor is not None else "unbinned"
    return cache_dir(data_path, cache_dir_name) / f"features_{suffix}.pkl"


def parse_timestamp_from_filename(filename: str | Path) -> datetime | None:
    match = re.search(r"(\d{8})[_-]?(\d{6})", Path(filename).stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_cached_image_records(
    data_path: str | Path,
    bin_factor: int | None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> tuple[pd.DataFrame, Path]:
    cache_path = feature_cache_pickle_path(data_path, bin_factor, cache_dir_name)
    if not cache_path.exists():
        return pd.DataFrame(columns=IMAGE_COLUMNS + ["image_key"]), cache_path

    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        return pd.DataFrame(columns=IMAGE_COLUMNS + ["image_key"]), cache_path

    if cached.empty or "filename" not in cached.columns:
        return pd.DataFrame(columns=IMAGE_COLUMNS + ["image_key"]), cache_path

    out = pd.DataFrame()
    out["filename"] = cached["filename"].astype(str)
    out["image_key"] = cached.get("image_key", out["filename"]).astype(str)

    if "path" in cached.columns:
        out["path"] = cached["path"].map(Path)
    else:
        out["path"] = out["filename"].map(lambda name: Path(data_path) / name)

    if "timestamp" in cached.columns:
        timestamps = pd.to_datetime(cached["timestamp"])
        out["timestamp"] = [None if pd.isna(ts) else ts.to_pydatetime() for ts in timestamps]
    else:
        out["timestamp"] = out["filename"].map(parse_timestamp_from_filename)

    out["timestamp_exif"] = cached["timestamp_exif"] if "timestamp_exif" in cached.columns else None
    out["timestamp_mtime"] = cached["timestamp_mtime"] if "timestamp_mtime" in cached.columns else pd.NaT
    out["timestamp_source"] = cached["timestamp_source"] if "timestamp_source" in cached.columns else "feature_cache"

    return out[IMAGE_COLUMNS + ["image_key"]], cache_path


def latest_filename_timestamp(records: pd.DataFrame) -> datetime | None:
    if records.empty:
        return None
    parsed = records["filename"].map(parse_timestamp_from_filename).dropna()
    if parsed.empty:
        return None
    return max(parsed)


def concat_if_needed(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].copy()
    return pd.concat(frames, ignore_index=True, sort=False)


def image_paths(
    root: str | Path,
    latest_known_ts: datetime | None = None,
    force_rescan: bool = False,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> tuple[list[Path], list[dict[str, Any]], int]:
    root = Path(root)
    paths = []
    skipped = []
    considered = 0

    iterator = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    for path in iterator:
        if is_cached_image_path(path, root, cache_dir_name):
            continue

        filename_ts = parse_timestamp_from_filename(path.name)
        if not force_rescan and latest_known_ts is not None and filename_ts is not None and filename_ts <= latest_known_ts:
            continue

        considered += 1
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append({"path": path, "reason": f"stat failed: {exc}"})
            continue

        if size == 0:
            skipped.append({"path": path, "reason": "zero-byte file"})
            continue

        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:
            skipped.append({"path": path, "reason": f"unreadable image: {exc}"})
            continue

        paths.append(path)

    return paths, skipped, considered


def exif_timestamp(path: str | Path) -> datetime | None:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            for tag in EXIF_DATETIME_TAGS:
                value = exif.get(tag)
                if value:
                    return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def mtime_timestamp(path: str | Path) -> datetime:
    return datetime.fromtimestamp(Path(path).stat().st_mtime)


def discover_images(
    data_path: str | Path,
    bin_factor: int | None = 4,
    force_rescan: bool = False,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached_image_records, early_feature_cache_path = load_cached_image_records(data_path, bin_factor, cache_dir_name)
    latest_known_ts = None if force_rescan else latest_filename_timestamp(cached_image_records)

    paths, skipped_files, considered_files = image_paths(
        data_path,
        latest_known_ts=latest_known_ts,
        force_rescan=force_rescan,
        cache_dir_name=cache_dir_name,
    )

    records = []
    for path in paths:
        ts_name = parse_timestamp_from_filename(path.name)
        ts_exif = exif_timestamp(path) if ts_name is None else None
        ts_mtime = mtime_timestamp(path)
        timestamp = ts_name or ts_exif or ts_mtime
        source = "filename" if ts_name is not None else "exif" if ts_exif is not None else "mtime"
        records.append(
            {
                "path": path,
                "filename": path.name,
                "timestamp_exif": ts_exif,
                "timestamp_mtime": ts_mtime,
                "timestamp": timestamp,
                "timestamp_source": source,
                "image_key": path.name,
            }
        )

    new_image_records = pd.DataFrame(records, columns=IMAGE_COLUMNS + ["image_key"])
    images = new_image_records if force_rescan else concat_if_needed([cached_image_records, new_image_records])

    if not images.empty:
        images["image_key"] = images.get("image_key", images["filename"]).astype(str)
        images = images.sort_values("timestamp").drop_duplicates("image_key", keep="last").reset_index(drop=True)

    info = {
        "feature_cache_used_for_discovery": early_feature_cache_path,
        "latest_cached_filename_timestamp": latest_known_ts,
        "top_level_source_files_checked": considered_files,
        "new_readable_source_images": len(new_image_records),
        "total_image_records": len(images),
        "skipped_files": skipped_files,
    }
    return images, info


def load_rgb(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Could not read image {path}: {exc}") from exc


def to_luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def median_bin(image: np.ndarray, factor: int | None = 4) -> np.ndarray:
    image = np.asarray(image)
    if factor is None or factor <= 1:
        return image
    h, w = image.shape[:2]
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    cropped = image[:h2, :w2]
    if image.ndim == 2:
        return np.median(cropped.reshape(h2 // factor, factor, w2 // factor, factor), axis=(1, 3))
    return np.median(cropped.reshape(h2 // factor, factor, w2 // factor, factor, image.shape[2]), axis=(1, 3))


def sample_indices(n: int, k: int = 12) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int)))


def binned_image_path(
    path: str | Path,
    bin_factor: int = 4,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> Path:
    path = Path(path)
    return path.parent / cache_dir_name / f"{path.stem}_bin{bin_factor}.png"


def read_cached_luma(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def quantize_luma_for_cache(luma: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(luma), 0, 255).astype(np.uint8)


def write_cached_luma(path: str | Path, luma: np.ndarray) -> np.ndarray:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = quantize_luma_for_cache(luma)
    Image.fromarray(arr, mode="L").save(path)
    return arr.astype(np.float32)


def load_luma(
    path: str | Path,
    bin_factor: int | None = 4,
    use_cache: bool = True,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> np.ndarray:
    global BIN_CACHE_HITS, BIN_CACHE_MISSES

    if bin_factor is None or bin_factor <= 1 or not use_cache:
        return median_bin(to_luminance(load_rgb(path)), factor=bin_factor)

    cached_path = binned_image_path(path, bin_factor=bin_factor, cache_dir_name=cache_dir_name)
    if cached_path.exists() and cached_path.stat().st_size > 0:
        try:
            BIN_CACHE_HITS += 1
            return read_cached_luma(cached_path)
        except Exception as exc:
            BIN_CACHE_WARNINGS.append({"path": cached_path, "reason": f"cache read failed: {exc}"})

    BIN_CACHE_MISSES += 1
    luma = median_bin(to_luminance(load_rgb(path)), factor=bin_factor)
    try:
        return write_cached_luma(cached_path, luma)
    except Exception as exc:
        BIN_CACHE_WARNINGS.append({"path": cached_path, "reason": f"cache write failed: {exc}"})
        return quantize_luma_for_cache(luma).astype(np.float32)


def normalize_roi(roi: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in roi]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def rolling_count(mask: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(np.asarray(mask, dtype=float), np.ones(window), mode="same")


def row_metrics_luma(image: np.ndarray, local_window: int = BAND_MASK_LOCAL_WINDOW) -> pd.DataFrame:
    image = np.asarray(image, dtype=np.float32)
    row_lum_median = np.median(image, axis=1)
    local_baseline = (
        pd.Series(row_lum_median).rolling(local_window, center=True, min_periods=1).median().to_numpy()
    )
    return pd.DataFrame(
        {
            "y": np.arange(image.shape[0]),
            "lum_median": row_lum_median,
            "lum_iqr": np.percentile(image, 75, axis=1) - np.percentile(image, 25, axis=1),
            "black_fraction": (image < 8).mean(axis=1),
            "white_fraction": (image > 220).mean(axis=1),
            "row_delta": np.r_[0, np.abs(np.diff(row_lum_median))],
            "abs_local_residual": np.abs(row_lum_median - local_baseline),
        }
    )


def dilate_row_mask(mask: np.ndarray, radius: int = BAND_MASK_DILATION_RADIUS) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[:-shift] |= mask[shift:]
        out[shift:] |= mask[:-shift]
    return out


def fill_small_gaps(mask: np.ndarray, max_gap: int = BAND_MASK_FILL_GAP) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        start = i
        while i < len(out) and not out[i]:
            i += 1
        if start > 0 and i < len(out) and i - start <= max_gap:
            out[start:i] = True
    return out


def remove_small_runs(mask: np.ndarray, min_len: int = BAND_MASK_MIN_RUN_LEN) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    i = 0
    while i < len(out):
        if not out[i]:
            i += 1
            continue
        start = i
        while i < len(out) and out[i]:
            i += 1
        if i - start < min_len:
            out[start:i] = False
    return out


def robust_binned_row_band_mask(image: np.ndarray) -> dict[str, Any]:
    df = row_metrics_luma(image)
    row_delta = df["row_delta"].to_numpy()
    median = float(np.median(row_delta))
    mad = float(np.median(np.abs(row_delta - median)))
    scale = 1.4826 * mad if mad > 0 else max(float(np.std(row_delta)), 1.0)

    edge_seed = (row_delta > BAND_MASK_ROW_DELTA_ABS_FLOOR) & (
        (row_delta - median) / (scale + 1e-6) > BAND_MASK_ROW_DELTA_ROBUST_Z
    )
    flat_seed = (
        (df["lum_iqr"] < 8) & ((df["black_fraction"] > 0.5) | (df["white_fraction"] > 0.5))
    ).to_numpy()
    seeds = edge_seed | flat_seed
    evidence = rolling_count(seeds, BAND_MASK_EVIDENCE_WINDOW) >= BAND_MASK_EVIDENCE_MIN_COUNT
    band = dilate_row_mask(evidence, radius=BAND_MASK_DILATION_RADIUS)
    band = fill_small_gaps(band, max_gap=BAND_MASK_FILL_GAP)
    band = remove_small_runs(band, min_len=BAND_MASK_MIN_RUN_LEN)

    return {
        "edge_seed": edge_seed,
        "flat_seed": flat_seed,
        "seeds": seeds,
        "evidence": evidence,
        "band": band,
        "band_fraction": float(np.mean(band)),
        "row_delta_max": float(np.max(row_delta)),
        "row_delta_median": median,
        "row_delta_mad": mad,
        "row_delta_scale": scale,
    }


def mask_banded_rows(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    band_info = robust_binned_row_band_mask(image)
    masked = np.asarray(image, dtype=np.float32).copy()
    apply_mask = band_info["band_fraction"] >= BAND_MASK_MIN_FRACTION_TO_APPLY
    if apply_mask:
        masked[band_info["band"], :] = np.nan
    band_info["applied"] = bool(apply_mask)
    return masked, band_info


def gaussian_blur_nan(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return np.asarray(image, dtype=np.float32)

    data = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(data)
    if not valid.any():
        return np.full_like(data, np.nan, dtype=np.float32)

    filled = np.where(valid, data, 0.0)
    weights = valid.astype(np.float32)
    blurred = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
    weight_blur = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")

    out = np.full_like(data, np.nan, dtype=np.float32)
    good = weight_blur > 1e-6
    out[good] = blurred[good] / weight_blur[good]
    return out


def preprocess_structure_image(
    image: np.ndarray,
    mode: str = "none",
    sigma: float = 2.0,
    amount: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if mode in (None, "none"):
        return arr

    smooth = gaussian_blur_nan(arr, sigma=sigma)
    if mode == "highpass":
        out = arr - smooth
    elif mode == "unsharp":
        out = arr + amount * (arr - smooth)
    else:
        raise ValueError(f"Unknown preprocess_mode={mode!r}")

    if clip and np.isfinite(arr).any():
        out = np.clip(out, np.nanmin(arr), np.nanmax(arr))
    return out.astype(np.float32)


def compact_image_mean(image: np.ndarray, factor: int = 8) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if factor is None or factor <= 1:
        return arr
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    arr = arr[:h2, :w2]
    if h2 == 0 or w2 == 0:
        raise ValueError("Downsample factor too large for image shape")
    return arr.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def vectorize_image_for_general_change(
    path: str | Path,
    *,
    bin_factor: int = 4,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
    spatial_downsample: int = 8,
    preprocess_mode: str = "none",
    preprocess_sigma: float = 2.0,
    preprocess_amount: float = 1.0,
    clip_preprocessed: bool = True,
) -> np.ndarray:
    img = load_luma(path, bin_factor=bin_factor, cache_dir_name=cache_dir_name)
    img, _ = mask_banded_rows(img)
    img = preprocess_structure_image(
        img,
        mode=preprocess_mode,
        sigma=preprocess_sigma,
        amount=preprocess_amount,
        clip=clip_preprocessed,
    )
    compact = compact_image_mean(img, factor=spatial_downsample)
    if np.isfinite(compact).any():
        fill_value = np.nanmedian(compact)
    else:
        fill_value = 0.0
    if not np.isfinite(fill_value):
        fill_value = 0.0
    compact = np.where(np.isfinite(compact), compact, fill_value)
    return compact.astype(np.float32)


def vectorize_image_for_general_change_job(args: tuple[Any, ...]) -> np.ndarray:
    (
        path,
        bin_factor,
        cache_dir_name,
        spatial_downsample,
        preprocess_mode,
        preprocess_sigma,
        preprocess_amount,
        clip_preprocessed,
    ) = args
    return vectorize_image_for_general_change(
        path,
        bin_factor=bin_factor,
        cache_dir_name=cache_dir_name,
        spatial_downsample=spatial_downsample,
        preprocess_mode=preprocess_mode,
        preprocess_sigma=preprocess_sigma,
        preprocess_amount=preprocess_amount,
        clip_preprocessed=clip_preprocessed,
    )


def general_change_model_signature(model_path: str | Path | None) -> dict[str, Any] | None:
    if not model_path:
        return None
    path = Path(model_path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def save_general_change_model(path: str | Path, model: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(dict(model), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def load_general_change_model(model_path: str | Path | None) -> dict[str, Any] | None:
    if not model_path:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        model = pickle.load(handle)
    required = {"scaler", "pca", "vector_bin_factor", "spatial_downsample", "preprocess_mode", "preprocess_sigma", "preprocess_amount", "clip_preprocessed"}
    missing = required.difference(model)
    if missing:
        raise ValueError(f"General-change PCA model missing keys: {sorted(missing)}")
    return dict(model)


def project_general_change_rows(
    rows: pd.DataFrame,
    model: Mapping[str, Any] | None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> pd.DataFrame:
    if model is None or rows.empty:
        return pd.DataFrame()

    scaler = model["scaler"]
    pca = model["pca"]
    expected_size = model.get("vector_size")
    projection_rows: list[dict[str, Any]] = []

    for row in rows.itertuples():
        compact = vectorize_image_for_general_change(
            row.path,
            bin_factor=int(model.get("vector_bin_factor", 4)),
            cache_dir_name=cache_dir_name,
            spatial_downsample=int(model.get("spatial_downsample", 8)),
            preprocess_mode=str(model.get("preprocess_mode", "none")),
            preprocess_sigma=float(model.get("preprocess_sigma", 2.0)),
            preprocess_amount=float(model.get("preprocess_amount", 1.0)),
            clip_preprocessed=bool(model.get("clip_preprocessed", True)),
        )
        x_raw = compact.reshape(1, -1)
        if expected_size is not None and x_raw.shape[1] != int(expected_size):
            raise ValueError(
                f"Vector size mismatch for {row.path}: got {x_raw.shape[1]}, expected {expected_size}"
            )
        row_median = np.median(x_raw, axis=1, keepdims=True)
        x_centered = x_raw - row_median
        x_scaled = scaler.transform(x_centered)
        z = pca.transform(x_scaled)[0]
        recon = pca.inverse_transform(z.reshape(1, -1))[0]
        residual = float(np.sqrt(np.mean((x_scaled[0] - recon) ** 2)))

        record = {
            "image_key": str(row.image_key),
            "general_change_residual_projected": residual,
            "general_vector_luma_median": float(row_median[0, 0]),
        }
        for idx, value in enumerate(z, start=1):
            record[f"general_pc{idx}"] = float(value)
        projection_rows.append(record)

    return pd.DataFrame(projection_rows)


def show_unavailable_image(path: str | Path, reason: Exception, ax: Any = None, title: str | None = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    path = Path(path)
    ax.text(
        0.5,
        0.55,
        "Image unavailable",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.45,
        f"{path.name}\n{reason}",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    ax.set_title(title or f"{path.name} unavailable")
    return ax


def show_image(path: str | Path, bin_factor: int | None = 4, ax: Any = None, title: str | None = None, cache_dir_name: str = DEFAULT_CACHE_DIR_NAME) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    try:
        img = load_luma(path, bin_factor=bin_factor, cache_dir_name=cache_dir_name)
    except Exception as exc:
        return show_unavailable_image(path, exc, ax=ax, title=title)
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax


def show_grid(
    path: str | Path,
    bin_factor: int | None = 4,
    step: int = 25,
    major_every: int = 4,
    ax: Any = None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    try:
        img = load_luma(path, bin_factor=bin_factor, cache_dir_name=cache_dir_name)
    except Exception as exc:
        return show_unavailable_image(path, exc, ax=ax)
    ax.imshow(img, cmap="gray")
    h, w = img.shape
    for x in range(0, w, step):
        is_major = (x // step) % major_every == 0
        ax.axvline(x, color="yellow", lw=0.8 if is_major else 0.35, alpha=0.75 if is_major else 0.35)
    for y in range(0, h, step):
        is_major = (y // step) % major_every == 0
        ax.axhline(y, color="yellow", lw=0.8 if is_major else 0.35, alpha=0.75 if is_major else 0.35)
    ax.set_xticks(range(0, w, step * major_every))
    ax.set_yticks(range(0, h, step * major_every))
    ax.tick_params(colors="yellow", labelsize=9)
    ax.set_title(f"Grid coordinates after {bin_factor}x median binning: width={w}, height={h}")
    return ax


def draw_rois(
    path: str | Path,
    rois: Mapping[str, tuple[int, int, int, int] | list[int]],
    bin_factor: int | None = 4,
    ax: Any = None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> Any:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    try:
        img = load_luma(path, bin_factor=bin_factor, cache_dir_name=cache_dir_name)
    except Exception as exc:
        return show_unavailable_image(path, exc, ax=ax)
    ax.imshow(img, cmap="gray")
    for name, roi in rois.items():
        x0, y0, x1, y1 = normalize_roi(roi)
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="tab:red", lw=2)
        ax.add_patch(rect)
        ax.text(x0, y0 - 3, name, color="tab:red", fontsize=11, weight="bold", va="bottom")
    ax.set_title(f"ROIs on {Path(path).name}")
    ax.set_axis_off()
    return ax


def roi_values(image: np.ndarray, roi: tuple[int, int, int, int] | list[int]) -> np.ndarray:
    x0, y0, x1, y1 = normalize_roi(roi)
    h, w = image.shape[:2]
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=float)
    vals = image[y0:y1, x0:x1].ravel()
    return vals[np.isfinite(vals)]


def roi_crop(image: np.ndarray, roi: tuple[int, int, int, int] | list[int]) -> np.ndarray:
    x0, y0, x1, y1 = normalize_roi(roi)
    h, w = image.shape[:2]
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float32)


def rotated_roi_crop(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | list[int],
    rotate_cw_deg: float = DOOR_PROFILE_ROTATE_CW_DEG,
) -> np.ndarray:
    crop = roi_crop(image, roi)
    if crop.size == 0 or rotate_cw_deg == 0:
        return crop
    return ndimage.rotate(crop, angle=rotate_cw_deg, reshape=False, order=1, mode="constant", cval=np.nan)


def door_line_residual_metrics(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | list[int],
    rotate_cw_deg: float = DOOR_PROFILE_ROTATE_CW_DEG,
    max_masked_fraction: float = DOOR_PROFILE_MAX_MASKED_FRACTION,
) -> dict[str, float]:
    raw_crop = roi_crop(image, roi)
    raw_valid_fraction = float(np.mean(np.isfinite(raw_crop))) if raw_crop.size else np.nan
    raw_masked_fraction = 1.0 - raw_valid_fraction if pd.notna(raw_valid_fraction) else np.nan
    trusted = bool(raw_masked_fraction <= max_masked_fraction) if pd.notna(raw_masked_fraction) else False

    empty = {
        "door_line_resid_sum": np.nan,
        "door_line_resid_ssq": np.nan,
        "door_line_resid_sum_raw": np.nan,
        "door_line_resid_ssq_raw": np.nan,
        "door_profile_valid_fraction": raw_valid_fraction,
        "door_profile_masked_fraction": raw_masked_fraction,
        "door_profile_valid_fraction_rotated": np.nan,
        "door_profile_masked_fraction_rotated": np.nan,
        "door_metric_trusted": trusted,
    }
    crop = rotated_roi_crop(image, roi, rotate_cw_deg=rotate_cw_deg)
    if crop.size == 0 or not np.isfinite(crop).any():
        return empty

    valid_cols = np.isfinite(crop).any(axis=0)
    crop = crop[:, valid_cols]
    rotated_valid_fraction = float(np.mean(np.isfinite(crop))) if crop.size else np.nan
    rotated_masked_fraction = 1.0 - rotated_valid_fraction if pd.notna(rotated_valid_fraction) else np.nan
    if crop.shape[1] < 30:
        empty["door_profile_valid_fraction_rotated"] = rotated_valid_fraction
        empty["door_profile_masked_fraction_rotated"] = rotated_masked_fraction
        return empty

    profile = np.nanmean(crop, axis=0)
    n = profile.size
    edge_n = max(3, n // DOOR_PROFILE_EDGE_DIVISOR)
    edge_margin = max(2, n // DOOR_PROFILE_EDGE_MARGIN_DIVISOR)
    left_start = edge_margin
    left_end = min(n, edge_margin + edge_n)
    right_end = max(0, n - edge_margin)
    right_start = max(0, right_end - edge_n)
    if right_start <= left_end:
        empty["door_profile_valid_fraction_rotated"] = rotated_valid_fraction
        empty["door_profile_masked_fraction_rotated"] = rotated_masked_fraction
        return empty

    fit_segment = profile[left_end:right_start]
    if fit_segment.size < 3:
        empty["door_profile_valid_fraction_rotated"] = rotated_valid_fraction
        empty["door_profile_masked_fraction_rotated"] = rotated_masked_fraction
        return empty

    x_fit = np.arange(left_end, right_start, dtype=np.float32)
    y_fit = fit_segment - float(np.nanmean(fit_segment))
    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    fitted = slope * x_fit + intercept
    residual = y_fit - fitted
    sumabs = float(np.nansum(np.abs(residual)))
    ssq = float(np.nansum(residual ** 2))

    return {
        "door_line_resid_sum": sumabs if trusted else np.nan,
        "door_line_resid_ssq": ssq if trusted else np.nan,
        "door_line_resid_sum_raw": sumabs,
        "door_line_resid_ssq_raw": ssq,
        "door_profile_valid_fraction": raw_valid_fraction,
        "door_profile_masked_fraction": raw_masked_fraction,
        "door_profile_valid_fraction_rotated": rotated_valid_fraction,
        "door_profile_masked_fraction_rotated": rotated_masked_fraction,
        "door_metric_trusted": trusted,
    }


def car_profile_residual_metrics(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | list[int],
    rotate_cw_deg: float = CAR_PROFILE_ROTATE_CW_DEG,
    max_masked_fraction: float = CAR_PROFILE_MAX_MASKED_FRACTION,
    fit_degree: int = CAR_PROFILE_FIT_DEGREE,
) -> dict[str, float]:
    raw_crop = roi_crop(image, roi)
    raw_valid_fraction = float(np.mean(np.isfinite(raw_crop))) if raw_crop.size else np.nan
    raw_masked_fraction = 1.0 - raw_valid_fraction if pd.notna(raw_valid_fraction) else np.nan

    empty = {
        "car_poly_resid_sumabs": np.nan,
        "car_poly_resid_ssq": np.nan,
        "car_poly_resid_sumabs_raw": np.nan,
        "car_poly_resid_ssq_raw": np.nan,
        "car_profile_valid_fraction": raw_valid_fraction,
        "car_profile_masked_fraction": raw_masked_fraction,
        "car_profile_valid_fraction_rotated": np.nan,
        "car_profile_masked_fraction_rotated": np.nan,
        "car_metric_trusted": False,
        "car_profile_fit_degree": int(fit_degree),
    }

    crop = rotated_roi_crop(image, roi, rotate_cw_deg=rotate_cw_deg)
    if crop.size == 0 or not np.isfinite(crop).any():
        return empty

    valid_cols = np.isfinite(crop).any(axis=0)
    crop = crop[:, valid_cols]
    rotated_valid_fraction = float(np.mean(np.isfinite(crop))) if crop.size else np.nan
    rotated_masked_fraction = 1.0 - rotated_valid_fraction if pd.notna(rotated_valid_fraction) else np.nan
    trusted = bool(raw_masked_fraction <= max_masked_fraction) if pd.notna(raw_masked_fraction) else False
    min_points = max(20, int(fit_degree) + 2)
    if crop.shape[1] < min_points:
        empty["car_profile_valid_fraction_rotated"] = rotated_valid_fraction
        empty["car_profile_masked_fraction_rotated"] = rotated_masked_fraction
        empty["car_metric_trusted"] = trusted
        return empty

    profile = np.nanmedian(crop, axis=0)
    finite = np.isfinite(profile)
    if finite.sum() < min_points:
        empty["car_profile_valid_fraction_rotated"] = rotated_valid_fraction
        empty["car_profile_masked_fraction_rotated"] = rotated_masked_fraction
        empty["car_metric_trusted"] = trusted
        return empty

    x = np.arange(profile.size, dtype=float)
    y = profile - float(np.nanmean(profile))
    finite_x = x[finite]
    finite_y = y[finite]
    coeffs = np.polyfit(finite_x, finite_y, deg=int(fit_degree))
    fit_curve = np.polyval(coeffs, finite_x)
    residual = finite_y - fit_curve
    sumabs = float(np.nansum(np.abs(residual)))
    ssq = float(np.nansum(residual ** 2))

    return {
        "car_poly_resid_sumabs": sumabs if trusted else np.nan,
        "car_poly_resid_ssq": ssq if trusted else np.nan,
        "car_poly_resid_sumabs_raw": sumabs,
        "car_poly_resid_ssq_raw": ssq,
        "car_profile_valid_fraction": raw_valid_fraction,
        "car_profile_masked_fraction": raw_masked_fraction,
        "car_profile_valid_fraction_rotated": rotated_valid_fraction,
        "car_profile_masked_fraction_rotated": rotated_masked_fraction,
        "car_metric_trusted": trusted,
        "car_profile_fit_degree": int(fit_degree),
    }

def roi_stats_for_image(
    path: str | Path,
    rois: Mapping[str, tuple[int, int, int, int] | list[int]],
    bin_factor: int | None = 4,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
    door_profile_rotate_cw_deg: float = DOOR_PROFILE_ROTATE_CW_DEG,
    door_profile_max_masked_fraction: float = DOOR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_rotate_cw_deg: float = CAR_PROFILE_ROTATE_CW_DEG,
    car_profile_max_masked_fraction: float = CAR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_fit_degree: int = CAR_PROFILE_FIT_DEGREE,
) -> dict[str, float | int]:
    img = load_luma(path, bin_factor=bin_factor, cache_dir_name=cache_dir_name)
    img, band_info = mask_banded_rows(img)
    out = {}
    out["band_mask_rows"] = int(np.sum(band_info["band"]))
    out["band_mask_fraction"] = float(band_info["band_fraction"])
    out["band_mask_applied"] = bool(band_info["applied"])
    out["band_mask_row_delta_max"] = float(band_info["row_delta_max"])
    for name, roi in rois.items():
        vals = roi_values(img, roi)
        if vals.size == 0:
            out[f"{name}_n"] = 0
            for stat in ROI_STATS:
                out[f"{name}_{stat}"] = np.nan
            continue
        out[f"{name}_n"] = vals.size
        out[f"{name}_mean"] = float(np.nanmean(vals))
        out[f"{name}_median"] = float(np.nanmedian(vals))
        out[f"{name}_std"] = float(np.nanstd(vals))
        for q in [10, 25, 75, 90, 95]:
            out[f"{name}_p{q}"] = float(np.nanpercentile(vals, q))

    if "door" in rois:
        out.update(
            door_line_residual_metrics(
                img,
                rois["door"],
                rotate_cw_deg=door_profile_rotate_cw_deg,
                max_masked_fraction=door_profile_max_masked_fraction,
            )
        )
    if "car_door" in rois:
        out.update(
            car_profile_residual_metrics(
                img,
                rois["car_door"],
                rotate_cw_deg=car_profile_rotate_cw_deg,
                max_masked_fraction=car_profile_max_masked_fraction,
                fit_degree=car_profile_fit_degree,
            )
        )
    return out


def add_ratio_feature(out: pd.DataFrame, numerator: str, denominator: str, name: str | None = None, eps: float = 1e-6) -> None:
    if {numerator, denominator}.issubset(out.columns):
        feature_name = name or f"{numerator}_to_{denominator}"
        out[feature_name] = out[numerator] / (out[denominator] + eps)


def add_difference_feature(out: pd.DataFrame, left: str, right: str, name: str | None = None) -> None:
    if {left, right}.issubset(out.columns):
        feature_name = name or f"{left}_minus_{right}"
        out[feature_name] = out[left] - out[right]


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    add_ratio_feature(out, "door_median", "wall_median", "door_to_wall_median")
    add_difference_feature(out, "door_median", "wall_median", "door_minus_wall_median")

    add_ratio_feature(out, "gap_or_outside_median", "wall_median", "gap_to_wall_median")

    return out


def add_interesting_image_feature(
    df: pd.DataFrame,
    interesting_pc4_threshold: float | None,
    interesting_pc5_threshold: float | None,
) -> pd.DataFrame:
    out = df.copy()
    if interesting_pc4_threshold is None or interesting_pc5_threshold is None:
        return out
    if {"general_pc4", "general_pc5"}.issubset(out.columns):
        uninteresting = (out["general_pc4"] > float(interesting_pc4_threshold)) & (
            out["general_pc5"] > float(interesting_pc5_threshold)
        )
        out["interesting_image_flag"] = (~uninteresting).astype(float)
    elif "interesting_image_flag" not in out.columns:
        out["interesting_image_flag"] = np.nan
    return out

def normalized_rois_for_cache(rois: Mapping[str, tuple[int, int, int, int] | list[int]]) -> dict[str, list[int]]:
    return {name: [int(v) for v in normalize_roi(roi)] for name, roi in rois.items()}


def feature_cache_metadata(
    rois: Mapping[str, tuple[int, int, int, int] | list[int]],
    bin_factor: int | None,
    door_profile_rotate_cw_deg: float = DOOR_PROFILE_ROTATE_CW_DEG,
    door_profile_max_masked_fraction: float = DOOR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_rotate_cw_deg: float = CAR_PROFILE_ROTATE_CW_DEG,
    car_profile_max_masked_fraction: float = CAR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_fit_degree: int = CAR_PROFILE_FIT_DEGREE,
    general_change_model_path: str | Path | None = None,
    interesting_pc4_threshold: float | None = None,
    interesting_pc5_threshold: float | None = None,
) -> dict[str, Any]:
    return {
        "bin_factor": int(bin_factor) if bin_factor is not None else None,
        "rois": normalized_rois_for_cache(rois),
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "door_profile_rotate_cw_deg": float(door_profile_rotate_cw_deg),
        "door_profile_max_masked_fraction": float(door_profile_max_masked_fraction),
        "car_profile_rotate_cw_deg": float(car_profile_rotate_cw_deg),
        "car_profile_max_masked_fraction": float(car_profile_max_masked_fraction),
        "car_profile_fit_degree": int(car_profile_fit_degree),
        "general_change_model": general_change_model_signature(general_change_model_path),
        "interesting_pc4_threshold": None if interesting_pc4_threshold is None else float(interesting_pc4_threshold),
        "interesting_pc5_threshold": None if interesting_pc5_threshold is None else float(interesting_pc5_threshold),
    }

def feature_cache_paths(
    data_path: str | Path,
    bin_factor: int | None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
) -> tuple[Path, Path]:
    suffix = f"bin{bin_factor}" if bin_factor is not None else "unbinned"
    cache_path = cache_dir(data_path, cache_dir_name) / f"features_{suffix}.pkl"
    return cache_path, cache_path.with_suffix(".meta.json")


def load_feature_cache(
    data_path: str | Path,
    rois: Mapping[str, tuple[int, int, int, int] | list[int]],
    bin_factor: int | None,
    force: bool = False,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
    door_profile_rotate_cw_deg: float = DOOR_PROFILE_ROTATE_CW_DEG,
    door_profile_max_masked_fraction: float = DOOR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_rotate_cw_deg: float = CAR_PROFILE_ROTATE_CW_DEG,
    car_profile_max_masked_fraction: float = CAR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_fit_degree: int = CAR_PROFILE_FIT_DEGREE,
    general_change_model_path: str | Path | None = None,
    interesting_pc4_threshold: float | None = None,
    interesting_pc5_threshold: float | None = None,
) -> tuple[pd.DataFrame, Path, Path, dict[str, Any], str]:
    cache_path, meta_path = feature_cache_paths(data_path, bin_factor, cache_dir_name)
    expected_meta = feature_cache_metadata(
        rois,
        bin_factor,
        door_profile_rotate_cw_deg=door_profile_rotate_cw_deg,
        door_profile_max_masked_fraction=door_profile_max_masked_fraction,
        car_profile_rotate_cw_deg=car_profile_rotate_cw_deg,
        car_profile_max_masked_fraction=car_profile_max_masked_fraction,
        car_profile_fit_degree=car_profile_fit_degree,
        general_change_model_path=general_change_model_path,
        interesting_pc4_threshold=interesting_pc4_threshold,
        interesting_pc5_threshold=interesting_pc5_threshold,
    )

    if force or not cache_path.exists():
        return pd.DataFrame(), cache_path, meta_path, expected_meta, "none"

    if meta_path.exists():
        try:
            observed_meta = json.loads(meta_path.read_text())
        except Exception:
            return pd.DataFrame(), cache_path, meta_path, expected_meta, "metadata unreadable"
        comparable_observed = {key: observed_meta.get(key) for key in expected_meta}
        if comparable_observed != expected_meta:
            return pd.DataFrame(), cache_path, meta_path, expected_meta, "metadata mismatch"
    else:
        return pd.DataFrame(), cache_path, meta_path, expected_meta, "metadata missing"

    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        return pd.DataFrame(), cache_path, meta_path, expected_meta, "cache unreadable"

    if "image_key" not in cached.columns:
        if "filename" in cached.columns:
            cached = cached.copy()
            cached["image_key"] = cached["filename"].astype(str)
        else:
            return pd.DataFrame(), cache_path, meta_path, expected_meta, "missing key"

    if "path" in cached.columns:
        cached = cached.copy()
        cached["path"] = cached["path"].map(Path)

    return cached, cache_path, meta_path, expected_meta, "loaded"

def save_feature_cache(features: pd.DataFrame, cache_path: Path, meta_path: Path, metadata: Mapping[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(cache_path)
    meta = dict(metadata)
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta["rows"] = int(len(features))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def build_features(
    images: pd.DataFrame,
    rois: Mapping[str, tuple[int, int, int, int] | list[int]],
    data_path: str | Path,
    bin_factor: int | None = 4,
    force_recompute: bool = False,
    max_images: int | None = None,
    cache_dir_name: str = DEFAULT_CACHE_DIR_NAME,
    door_profile_rotate_cw_deg: float = DOOR_PROFILE_ROTATE_CW_DEG,
    door_profile_max_masked_fraction: float = DOOR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_rotate_cw_deg: float = CAR_PROFILE_ROTATE_CW_DEG,
    car_profile_max_masked_fraction: float = CAR_PROFILE_MAX_MASKED_FRACTION,
    car_profile_fit_degree: int = CAR_PROFILE_FIT_DEGREE,
    general_change_model_path: str | Path | None = None,
    interesting_pc4_threshold: float | None = None,
    interesting_pc5_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global BIN_CACHE_HITS, BIN_CACHE_MISSES, BIN_CACHE_WARNINGS

    if images.empty:
        raise ValueError("No image records available")

    work = images if max_images is None else images.iloc[:max_images]
    cached_features, feature_cache_path, feature_meta_path, feature_meta, feature_cache_status = load_feature_cache(
        data_path,
        rois,
        bin_factor,
        force=force_recompute,
        cache_dir_name=cache_dir_name,
        door_profile_rotate_cw_deg=door_profile_rotate_cw_deg,
        door_profile_max_masked_fraction=door_profile_max_masked_fraction,
        car_profile_rotate_cw_deg=car_profile_rotate_cw_deg,
        car_profile_max_masked_fraction=car_profile_max_masked_fraction,
        car_profile_fit_degree=car_profile_fit_degree,
        general_change_model_path=general_change_model_path,
        interesting_pc4_threshold=interesting_pc4_threshold,
        interesting_pc5_threshold=interesting_pc5_threshold,
    )

    work = work.copy()
    work["image_key"] = work["filename"].astype(str)
    cached_keys = set(cached_features["image_key"].astype(str)) if not cached_features.empty else set()
    new_work = work[~work["image_key"].isin(cached_keys)].copy()

    BIN_CACHE_HITS = 0
    BIN_CACHE_MISSES = 0
    BIN_CACHE_WARNINGS = []

    feature_rows = []
    feature_failures = []
    for _, row in new_work.iterrows():
        try:
            stats = roi_stats_for_image(
                row.path,
                rois,
                bin_factor=bin_factor,
                cache_dir_name=cache_dir_name,
                door_profile_rotate_cw_deg=door_profile_rotate_cw_deg,
                door_profile_max_masked_fraction=door_profile_max_masked_fraction,
                car_profile_rotate_cw_deg=car_profile_rotate_cw_deg,
                car_profile_max_masked_fraction=car_profile_max_masked_fraction,
                car_profile_fit_degree=car_profile_fit_degree,
            )
        except Exception as exc:
            feature_failures.append({"path": row.path, "filename": row.filename, "timestamp": row.timestamp, "reason": str(exc)})
            continue

        stats.update(
            {
                "image_key": row.image_key,
                "path": row.path,
                "filename": row.filename,
                "timestamp": row.timestamp,
            }
        )
        feature_rows.append(stats)

    new_features = pd.DataFrame(feature_rows)
    general_change_model = load_general_change_model(general_change_model_path)
    if general_change_model is not None and not new_features.empty:
        projected = project_general_change_rows(
            new_features[["image_key", "path"]],
            general_change_model,
            cache_dir_name=cache_dir_name,
        )
        if not projected.empty:
            new_features = new_features.merge(projected, on="image_key", how="left")

    if not new_features.empty:
        new_features = add_candidate_features(new_features)
        new_features = add_interesting_image_feature(
            new_features,
            interesting_pc4_threshold=interesting_pc4_threshold,
            interesting_pc5_threshold=interesting_pc5_threshold,
        )

    features = concat_if_needed([cached_features, new_features])
    if not features.empty:
        features["image_key"] = features["image_key"].astype(str)
        features = features.sort_values("timestamp").drop_duplicates("image_key", keep="last").reset_index(drop=True)
        features = add_candidate_features(features)
        features = add_interesting_image_feature(
            features,
            interesting_pc4_threshold=interesting_pc4_threshold,
            interesting_pc5_threshold=interesting_pc5_threshold,
        )

    if features.empty:
        raise ValueError("No features were available. Check image readability, feature cache, and ROI definitions.")

    save_feature_cache(features, feature_cache_path, feature_meta_path, feature_meta)
    info = {
        "feature_cache_status": feature_cache_status,
        "feature_cache_path": feature_cache_path,
        "cached_feature_rows": len(cached_features),
        "images_needing_feature_extraction": len(new_work),
        "extracted_new_features": len(new_features),
        "total_feature_rows": len(features),
        "binned_cache_hits": BIN_CACHE_HITS,
        "binned_cache_misses": BIN_CACHE_MISSES,
        "binned_cache_warnings": BIN_CACHE_WARNINGS,
        "feature_failures": feature_failures,
        "general_change_model_path": None if general_change_model_path is None else str(general_change_model_path),
        "general_change_model_loaded": bool(general_change_model is not None),
    }
    return features, info

def threshold_rules_from_config(config: Mapping[str, Any]) -> dict[str, ThresholdRule]:
    rules = {}
    for name, values in config.get("conclusions", {}).items():
        rules[name] = ThresholdRule(
            feature=str(values["feature"]),
            threshold=float(values["threshold"]),
            higher_means_true=bool(values.get("higher_means_true", True)),
        )
    return rules


def latest_feature_row(features: pd.DataFrame) -> pd.Series:
    if features.empty:
        raise ValueError("No feature rows available")
    return features.sort_values("timestamp").iloc[-1]


def evaluate_row(row: pd.Series, rules: Mapping[str, ThresholdRule]) -> dict[str, Any]:
    conclusions = {}
    for name, rule in rules.items():
        value = row.get(rule.feature, np.nan)
        conclusions[name] = {
            "value": None if pd.isna(value) else float(value),
            "threshold": rule.threshold,
            "higher_means_true": rule.higher_means_true,
            "result": rule.evaluate(value),
            "feature": rule.feature,
        }
    return conclusions


def state_from_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    data_path = config["data_path"]
    bin_factor = config.get("bin_factor", 4)
    cache_dir_name = config.get("cache_dir_name", DEFAULT_CACHE_DIR_NAME)
    discovery_config = config.get("image_discovery", {})
    feature_config = config.get("features", {})
    general_change_config = config.get("general_change", {}) or {}

    images, discovery_info = discover_images(
        data_path,
        bin_factor=bin_factor,
        force_rescan=bool(discovery_config.get("force_rescan_images", False)),
        cache_dir_name=cache_dir_name,
    )
    features, feature_info = build_features(
        images,
        config["rois"],
        data_path,
        bin_factor=bin_factor,
        force_recompute=bool(feature_config.get("force_recompute_features", False)),
        max_images=feature_config.get("max_images"),
        cache_dir_name=cache_dir_name,
        door_profile_rotate_cw_deg=float(feature_config.get("door_profile_rotate_cw_deg", DOOR_PROFILE_ROTATE_CW_DEG)),
        door_profile_max_masked_fraction=float(feature_config.get("door_profile_max_masked_fraction", DOOR_PROFILE_MAX_MASKED_FRACTION)),
        car_profile_rotate_cw_deg=float(feature_config.get("car_profile_rotate_cw_deg", CAR_PROFILE_ROTATE_CW_DEG)),
        car_profile_max_masked_fraction=float(feature_config.get("car_profile_max_masked_fraction", CAR_PROFILE_MAX_MASKED_FRACTION)),
        car_profile_fit_degree=int(feature_config.get("car_profile_fit_degree", CAR_PROFILE_FIT_DEGREE)),
        general_change_model_path=general_change_config.get("model_path"),
        interesting_pc4_threshold=general_change_config.get("interesting_pc4_threshold"),
        interesting_pc5_threshold=general_change_config.get("interesting_pc5_threshold"),
    )
    row = latest_feature_row(features)
    rules = threshold_rules_from_config(config)
    conclusions = evaluate_row(row, rules)

    state = {
        "timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
        "filename": str(row["filename"]),
        "path": str(row["path"]),
        "image_key": str(row["image_key"]),
        "garage_door_open": conclusions.get("garage_door_open", {}).get("result"),
        "car_present": conclusions.get("car_present", {}).get("result"),
        "interesting_image": conclusions.get("interesting_image", {}).get("result"),
        "conclusions": conclusions,
    }
    info = {"discovery": discovery_info, "features": feature_info}
    return state, features, info

def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def format_env(state: Mapping[str, Any]) -> str:
    values = {
        "GARAGE_IMAGE_TIMESTAMP": state["timestamp"],
        "GARAGE_IMAGE_FILENAME": state["filename"],
        "GARAGE_DOOR_OPEN": state["garage_door_open"],
        "GARAGE_CAR_PRESENT": state["car_present"],
        "GARAGE_INTERESTING_IMAGE": state.get("interesting_image"),
    }
    return "\n".join(f"{key}={json.dumps(value)}" for key, value in values.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report garage door and car state from the latest camera image.")
    parser.add_argument("--config", default=Path(__file__).with_name("garage_config.yml"), help="Path to YAML config")
    parser.add_argument("--format", choices=["json", "env", "text"], default="json", help="Output format")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    state, _, _ = state_from_config(config)

    if args.format == "json":
        print(json.dumps(json_ready(state), indent=2 if args.pretty else None, sort_keys=True))
    elif args.format == "env":
        print(format_env(state))
    else:
        print(f"{state['timestamp']} {state['filename']}")
        print(f"garage_door_open={state['garage_door_open']}")
        print(f"car_present={state['car_present']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

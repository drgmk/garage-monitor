from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import garage_monitor as gm
import garage_state as gs


def build_state_from_row(row: pd.Series, rules: Mapping[str, gm.ThresholdRule]) -> dict[str, Any]:
    conclusions = gm.evaluate_row(row, rules)
    return {
        "timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
        "filename": str(row["filename"]),
        "path": str(row["path"]),
        "image_key": str(row.get("image_key", row["filename"])),
        "garage_door_open": conclusions.get("garage_door_open", {}).get("result"),
        "car_present": conclusions.get("car_present", {}).get("result"),
        "conclusions": conclusions,
    }


def rebuild_timeline(config_path: Path, mode: str = "all") -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = gm.load_config(config_path)
    auto = gs.automation_config(config)
    discovery_config = config.get("image_discovery", {})
    feature_config = config.get("features", {})

    images, discovery_info = gm.discover_images(
        config["data_path"],
        bin_factor=config.get("bin_factor", 4),
        force_rescan=bool(discovery_config.get("force_rescan_images", False)),
        cache_dir_name=config.get("cache_dir_name", gm.DEFAULT_CACHE_DIR_NAME),
    )
    features, feature_info = gm.build_features(
        images,
        config["rois"],
        config["data_path"],
        bin_factor=config.get("bin_factor", 4),
        force_recompute=bool(feature_config.get("force_recompute_features", False)),
        max_images=feature_config.get("max_images"),
        cache_dir_name=config.get("cache_dir_name", gm.DEFAULT_CACHE_DIR_NAME),
        door_profile_rotate_cw_deg=float(feature_config.get("door_profile_rotate_cw_deg", gm.DOOR_PROFILE_ROTATE_CW_DEG)),
        car_profile_rotate_cw_deg=float(feature_config.get("car_profile_rotate_cw_deg", gm.CAR_PROFILE_ROTATE_CW_DEG)),
        car_profile_max_masked_fraction=float(feature_config.get("car_profile_max_masked_fraction", gm.CAR_PROFILE_MAX_MASKED_FRACTION)),
    )
    features = features.sort_values("timestamp").reset_index(drop=True)
    rules = gm.threshold_rules_from_config(config)

    previous_state: Mapping[str, Any] | None = None
    sent_checks: dict[str, str] = {}
    recent_events: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    latest_state: dict[str, Any] | None = None
    recent_events_limit = int(auto.get("recent_events_limit", 25))

    for _, row in features.iterrows():
        current_state = build_state_from_row(row, rules)
        current_state = gs.resolve_indeterminate_state(previous_state, current_state)
        now = pd.to_datetime(row["timestamp"]).to_pydatetime()
        latest_state = current_state

        emitted: list[dict[str, Any]] = []
        if mode in {"all", "transitions"}:
            emitted.extend(gs.transition_events(previous_state, current_state, auto.get("transitions", {}), now))

        if mode in {"all", "scheduled"}:
            scheduled, sent_checks = gs.scheduled_events(
                {"scheduled_checks": {"sent": sent_checks}},
                current_state,
                auto.get("scheduled_checks", []),
                now,
            )
            emitted.extend(scheduled)

        all_events.extend(emitted)
        recent_events = (recent_events + emitted)[-recent_events_limit:]
        previous_state = current_state

    if latest_state is None:
        raise ValueError("No states available from feature history")

    latest_binned = gs.latest_binned_image_path(config, latest_state)
    latest_image = None
    if latest_binned is not None:
        latest_image = {
            "path": str(latest_binned),
            "target": str(latest_binned.resolve()),
        }

    status_doc = {
        "updated_at": latest_state["timestamp"],
        "state": latest_state,
        "latest_binned_image": latest_image,
        "scheduled_checks": {"sent": sent_checks},
        "recent_events": recent_events,
    }
    info = {
        "discovery": discovery_info,
        "features": feature_info,
        "feature_rows": len(features),
        "event_rows": len(all_events),
        "mode": mode,
    }
    return status_doc, all_events, info


def write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(gm.json_ready(record), sort_keys=True) + "\n")


def default_output_paths(config_path: Path) -> tuple[Path, Path, Path]:
    config = gm.load_config(config_path)
    auto = gs.automation_config(config)
    base_dir = config_path.parent
    return (
        gs.resolve_path(auto["event_log_path"], base_dir),
        gs.resolve_path(auto["state_path"], base_dir),
        gs.resolve_path(auto["html_status_path"], base_dir),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild garage event timeline from feature history.")
    parser.add_argument("--config", default=Path(__file__).with_name("garage_config.yml"), help="Path to YAML config")
    parser.add_argument("--mode", choices=["all", "transitions", "scheduled"], default="all")
    parser.add_argument("--events-out", help="Path to rebuilt event log JSONL")
    parser.add_argument("--status-out", help="Path to rebuilt status JSON")
    parser.add_argument("--html-out", help="Path to rebuilt status HTML")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser()
    events_out, status_out, html_out = default_output_paths(config_path)
    if args.events_out:
        events_out = Path(args.events_out).expanduser()
    if args.status_out:
        status_out = Path(args.status_out).expanduser()
    if args.html_out:
        html_out = Path(args.html_out).expanduser()

    status_doc, events, info = rebuild_timeline(config_path, mode=args.mode)
    latest_image = status_doc.get("latest_binned_image")
    if latest_image is not None:
        latest_image["url"] = gs.relative_url(Path(latest_image["path"]), html_out)
    write_jsonl(events_out, events)
    gs.write_json_atomic(status_out, status_doc)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(gs.render_status_html(status_doc), encoding="utf-8")

    print(json.dumps(gm.json_ready({
        "events_out": events_out,
        "status_out": status_out,
        "html_out": html_out,
        "event_rows": len(events),
        "latest_state": status_doc["state"],
        "info": info,
    }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

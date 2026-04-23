"""State, event, and status-page layer for garage_monitor."""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import garage_monitor as gm


DEFAULT_AUTOMATION = {
    "state_path": "garage_status.json",
    "event_log_path": "garage_events.jsonl",
    "html_status_path": "garage_status.html",
    "latest_binned_image_path": "garage_latest_binned.png",
    "notification_log_path": "garage_notifications.jsonl",
    "notifications": {
        "enabled": False,
        "provider": "ntfy",
        "url": None,
        "url_env": "GARAGE_NTFY_URL",
        "timeout_seconds": 10,
        "title": "Garage",
        "tags": ["garage"],
    },
    "transitions": {},
    "scheduled_checks": [],
}


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else base_dir / path


def automation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULT_AUTOMATION)
    out.update(config.get("automation", {}) or {})
    notifications = dict(DEFAULT_AUTOMATION["notifications"])
    notifications.update(out.get("notifications") or {})
    out["notifications"] = notifications
    return out


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(gm.json_ready(data), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(gm.json_ready(record), sort_keys=True) + "\n")


def update_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = link_path.with_name(f".{link_path.name}.tmp")
    if tmp_path.exists() or tmp_path.is_symlink():
        tmp_path.unlink()
    tmp_path.symlink_to(target_path)
    os.replace(tmp_path, link_path)


def latest_binned_image_path(config: Mapping[str, Any], state: Mapping[str, Any]) -> Path | None:
    source_path = state.get("path")
    if not source_path:
        return None

    binned_path = gm.binned_image_path(
        source_path,
        bin_factor=config.get("bin_factor", 4),
        cache_dir_name=config.get("cache_dir_name", gm.DEFAULT_CACHE_DIR_NAME),
    )
    return binned_path if binned_path.exists() else None


def relative_url(path: Path, base_path: Path) -> str:
    return Path(os.path.relpath(path, start=base_path.parent)).as_posix()


def bool_key(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def state_label(key: str, value: Any) -> str:
    if key == "garage_door_open":
        return "open" if value is True else "closed" if value is False else "unknown"
    if key == "car_present":
        return "present" if value is True else "absent" if value is False else "unknown"
    return bool_key(value)


def format_heading_time(value: Any) -> str:
    try:
        observed_at = datetime.fromisoformat(str(value))
    except ValueError:
        return "unknown time"
    return observed_at.strftime("%H:%M on %a %-d %B")


def transition_events(
    previous_state: Mapping[str, Any] | None,
    current_state: Mapping[str, Any],
    rules: Mapping[str, Any],
    now: datetime,
) -> list[dict[str, Any]]:
    if not previous_state:
        return []

    events = []
    for key, key_rules in rules.items():
        before = previous_state.get(key)
        after = current_state.get(key)
        if before is after or before == after:
            continue

        transition_key = f"{bool_key(before)}_to_{bool_key(after)}"
        rule = (key_rules or {}).get(transition_key, {})
        if not rule or not bool(rule.get("enabled", False)):
            continue

        events.append(
            {
                "created_at": now.isoformat(timespec="seconds"),
                "kind": "transition",
                "name": f"{key}.{transition_key}",
                "entity": key,
                "from": before,
                "to": after,
                "message": rule.get("message", f"{key} changed from {before} to {after}."),
                "image_timestamp": current_state.get("timestamp"),
                "filename": current_state.get("filename"),
            }
        )
    return events


def condition_matches(state: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    return all(state.get(key) == expected for key, expected in condition.items())


def scheduled_events(
    status_doc: Mapping[str, Any],
    current_state: Mapping[str, Any],
    checks: list[Mapping[str, Any]],
    now: datetime,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    emitted = []
    sent = dict(status_doc.get("scheduled_checks", {}).get("sent", {}))
    today = now.date().isoformat()
    current_hhmm = now.strftime("%H:%M")

    for check in checks:
        if not bool(check.get("enabled", True)):
            continue

        name = str(check["name"])
        check_time = str(check["at"])
        if current_hhmm < check_time:
            continue
        if sent.get(name) == today:
            continue
        if not condition_matches(current_state, check.get("condition", {})):
            continue

        emitted.append(
            {
                "created_at": now.isoformat(timespec="seconds"),
                "kind": "scheduled_check",
                "name": name,
                "message": check.get("message", f"Scheduled garage check {name} matched."),
                "condition": check.get("condition", {}),
                "image_timestamp": current_state.get("timestamp"),
                "filename": current_state.get("filename"),
            }
        )
        sent[name] = today

    return emitted, sent


def notification_url(notification_config: Mapping[str, Any]) -> str | None:
    url = notification_config.get("url")
    if url:
        return str(url)
    url_env = notification_config.get("url_env")
    if url_env:
        return os.environ.get(str(url_env))
    return None


def ntfy_message(event: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    lines = [
        str(event.get("message", "Garage status changed.")),
        f"Door: {state_label('garage_door_open', state.get('garage_door_open'))}",
        f"Car: {state_label('car_present', state.get('car_present'))}",
    ]
    image_timestamp = event.get("image_timestamp")
    if image_timestamp:
        lines.append(f"Image time: {image_timestamp}")
    return "\n".join(lines)


def send_ntfy_notification(
    event: Mapping[str, Any],
    state: Mapping[str, Any],
    notification_config: Mapping[str, Any],
) -> dict[str, Any]:
    url = notification_url(notification_config)
    if not url:
        return {"notification_status": "skipped", "reason": "missing ntfy url"}

    title = str(notification_config.get("title") or "Garage")
    tags = notification_config.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    timeout = float(notification_config.get("timeout_seconds", 10))

    headers = {
        "Title": title,
        "Tags": ",".join(str(tag) for tag in tags),
    }
    request = urllib.request.Request(
        url,
        data=ntfy_message(event, state).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
        return {
            "notification_status": "sent",
            "provider": "ntfy",
            "status_code": response.status,
            "response": response_body,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "notification_status": "failed",
            "provider": "ntfy",
            "reason": str(exc),
        }


def send_notification(
    event: Mapping[str, Any],
    state: Mapping[str, Any],
    notification_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(notification_config.get("enabled", False)):
        return {"notification_status": "skipped", "reason": "notifications disabled"}

    provider = str(notification_config.get("provider", "ntfy")).lower()
    if provider != "ntfy":
        return {"notification_status": "skipped", "reason": f"unsupported provider: {provider}"}

    return send_ntfy_notification(event, state, notification_config)


def render_status_html(status_doc: Mapping[str, Any]) -> str:
    state = status_doc["state"]
    events = status_doc.get("recent_events", [])
    latest_image = status_doc.get("latest_binned_image")

    door = state_label("garage_door_open", state.get("garage_door_open"))
    car = state_label("car_present", state.get("car_present"))
    filename = html.escape(str(state.get("filename", "unknown")))
    heading_time = html.escape(format_heading_time(state.get("timestamp")))
    door_class = "is-bad" if state.get("garage_door_open") is True else "is-good" if state.get("garage_door_open") is False else ""
    car_class = "is-good" if state.get("car_present") is True else "is-bad" if state.get("car_present") is False else ""

    rows = []
    for event in events[-10:][::-1]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('created_at', '')))}</td>"
            f"<td>{html.escape(str(event.get('kind', '')))}</td>"
            f"<td>{html.escape(str(event.get('message', '')))}</td>"
            "</tr>"
        )

    event_rows = "\n".join(rows) or '<tr><td colspan="3">No events recorded yet.</td></tr>'
    image_section = ""
    if latest_image:
        image_src = html.escape(str(latest_image["url"]))
        image_section = f"""
    <section class="latest-image">
      <h2>Last Image ({filename})</h2>
      <img src="{image_src}" alt="Latest binned garage camera image">
    </section>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>Garage Status</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #1f2933;
    }}
    main {{
      max-width: 760px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 20px;
      line-height: 1.2;
    }}
    .status {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .tile {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 16px;
    }}
    .tile.is-good {{
      background: #dff3e6;
      border-color: #a8d8b9;
    }}
    .tile.is-bad {{
      background: #f8dddd;
      border-color: #e4a6a6;
    }}
    .label {{
      color: #52606d;
      font-size: 14px;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 30px;
      font-weight: 700;
      text-transform: capitalize;
    }}
    table {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      padding: 0;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #d9e2ec;
      padding: 10px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: #52606d;
      font-weight: 600;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .latest-image {{
      margin: 20px 0;
    }}
    .latest-image img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Status at {heading_time}</h1>
    <section class="status">
      <div class="tile {door_class}">
        <div class="label">Garage door</div>
        <div class="value">{html.escape(door)}</div>
      </div>
      <div class="tile {car_class}">
        <div class="label">Car</div>
        <div class="value">{html.escape(car)}</div>
      </div>
    </section>
    {image_section}
    <h2>Recent Events</h2>
    <table>
      <thead>
        <tr><th>Time</th><th>Type</th><th>Message</th></tr>
      </thead>
      <tbody>
        {event_rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def update_status(config_path: Path, mode: str = "all") -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    config = gm.load_config(config_path)
    base_dir = config_path.parent
    auto = automation_config(config)
    paths = {
        "state": resolve_path(auto["state_path"], base_dir),
        "event_log": resolve_path(auto["event_log_path"], base_dir),
        "html_status": resolve_path(auto["html_status_path"], base_dir),
        "latest_binned_image": resolve_path(auto["latest_binned_image_path"], base_dir),
        "notification_log": resolve_path(auto["notification_log_path"], base_dir),
    }

    now = datetime.now()
    previous_doc = read_json(paths["state"], default={})
    previous_state = previous_doc.get("state")
    current_state, _, _ = gm.state_from_config(config)

    events = []
    if mode in {"all", "transitions"}:
        events.extend(transition_events(previous_state, current_state, auto.get("transitions", {}), now))

    sent_checks = dict(previous_doc.get("scheduled_checks", {}).get("sent", {}))
    if mode in {"all", "scheduled"}:
        scheduled, sent_checks = scheduled_events(previous_doc, current_state, auto.get("scheduled_checks", []), now)
        events.extend(scheduled)

    latest_binned = latest_binned_image_path(config, current_state)
    latest_image = None
    if latest_binned is not None:
        update_symlink(paths["latest_binned_image"], latest_binned)
        latest_image = {
            "path": paths["latest_binned_image"],
            "target": latest_binned,
            "url": relative_url(paths["latest_binned_image"], paths["html_status"]),
        }

    previous_events = previous_doc.get("recent_events", [])
    recent_events = (previous_events + events)[-25:]
    status_doc = {
        "updated_at": now.isoformat(timespec="seconds"),
        "state": current_state,
        "latest_binned_image": latest_image,
        "scheduled_checks": {"sent": sent_checks},
        "recent_events": recent_events,
    }

    write_json_atomic(paths["state"], status_doc)
    paths["html_status"].parent.mkdir(parents=True, exist_ok=True)
    paths["html_status"].write_text(render_status_html(status_doc), encoding="utf-8")

    notification_config = auto.get("notifications", {})
    for event in events:
        append_jsonl(paths["event_log"], event)
        notification_result = send_notification(event, current_state, notification_config)
        append_jsonl(paths["notification_log"], {**event, **notification_result})

    return status_doc, events, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update garage status files and record notification-worthy events.")
    parser.add_argument("--config", default=Path(__file__).with_name("garage_config.yml"), help="Path to YAML config")
    parser.add_argument("--mode", choices=["all", "transitions", "scheduled"], default="all")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)

    status_doc, events, paths = update_status(Path(args.config).expanduser(), mode=args.mode)

    if args.format == "json":
        print(json.dumps(gm.json_ready({"status": status_doc, "events": events, "paths": paths}), indent=2, sort_keys=True))
        return 0

    state = status_doc["state"]
    print(f"status: garage_door={state_label('garage_door_open', state.get('garage_door_open'))}, car={state_label('car_present', state.get('car_present'))}")
    print(f"status_json: {paths['state']}")
    print(f"status_html: {paths['html_status']}")
    if events:
        print("notification candidates:")
        for event in events:
            print(f"- [{event['kind']}] {event['message']}")
    else:
        print("notification candidates: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

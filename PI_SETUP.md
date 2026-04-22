# Garage Monitor Pi Setup

This directory is intended to be copied to the Raspberry Pi that can see the
camera image directory. `garage_state.py` can send notification-worthy events to
ntfy and records each attempted notification in JSONL.

## Files

- `garage_monitor.py`: image detection and threshold evaluation.
- `garage_state.py`: status file, HTML page, transition detection, scheduled checks.
- `garage_config.yml`: paths, ROIs, thresholds, and automation rules.
- `garage_status.json`: generated current status for scripts.
- `garage_status.html`: generated phone-friendly status page.
- `garage_latest_binned.png`: generated symlink to the latest cached binned image.
- `garage_events.jsonl`: generated event history.
- `garage_notifications.jsonl`: generated pending notification records.

## Pi Configuration

Edit `garage_config.yml` after copying:

```yaml
data_path: /path/to/camera/images

automation:
  state_path: garage_status.json
  event_log_path: garage_events.jsonl
  html_status_path: garage_status.html
  notification_log_path: garage_notifications.jsonl
```

Relative automation paths are resolved relative to `garage_config.yml`, which
keeps the generated files beside the scripts.

## Manual Test

Run this from the copied directory:

```bash
python garage_state.py --config garage_config.yml
```

Expected output looks like:

```text
status: garage_door=closed, car=present
status_json: /home/pi/garage/garage_status.json
status_html: /home/pi/garage/garage_status.html
notification candidates: none
```

For machine-readable output:

```bash
python garage_state.py --config garage_config.yml --format json
```

## Cron

Run the detector frequently to keep current state fresh and catch transitions:

```cron
GARAGE_NTFY_URL=https://ntfy.sh/your-private-topic
*/5 * * * * cd /home/pi/garage && /usr/bin/python3 garage_state.py --config garage_config.yml >> garage_cron.log 2>&1
```

If you prefer the evening open-door check to run only from a specific cron entry:

```cron
0 21 * * * cd /home/pi/garage && /usr/bin/python3 garage_state.py --config garage_config.yml --mode scheduled >> garage_cron.log 2>&1
```

The default `--mode all` also evaluates scheduled checks. Each scheduled check is
recorded at most once per local day.

## Phone Lookup Over Tailscale

Once Tailscale is running on the Pi, the simplest lookup page is a small HTTP
server bound to the Tailscale interface or localhost. For a first test:

```bash
cd /home/pi/garage
python3 -m http.server 8765
```

Then open this from your phone while connected to Tailscale:

```text
http://<pi-tailscale-name-or-ip>:8765/garage_status.html
```

For a persistent setup, run the HTTP server with systemd, nginx, Caddy, or any
other small web server you already use. No public port forwarding is needed if
you access it through Tailscale.

## Notifications

`garage_notifications.jsonl` receives one JSON object per notification-worthy
event with `notification_status` set to `sent`, `failed`, or `skipped`.

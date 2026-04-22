# Garage Monitor

Threshold-based garage status monitor for Raspberry Pi camera images.

The detector reads the latest camera image, extracts brightness features from
configured regions of interest, and reports:

- whether the garage door appears open
- whether the car appears present

The state layer writes current status files, records transition events, and
generates a small HTML page that can be viewed from a phone over Tailscale.
Actual notification delivery is intentionally left as a later hook.

## Files

- `garage_monitor.py`: image discovery, ROI features, thresholds, and CLI output.
- `garage_state.py`: persistent status, event detection, scheduled checks, HTML output.
- `garage_config.yml`: data path, ROIs, thresholds, and automation rules.
- `explore.ipynb`: interactive notebook for reviewing ROIs, features, and thresholds.
- `PI_SETUP.md`: Raspberry Pi setup notes.
- `crontab.example`: example cron entries for the Pi.

## Quick Test

```bash
python garage_monitor.py --config garage_config.yml --pretty
python garage_state.py --config garage_config.yml
```

`garage_state.py` generates:

- `garage_status.json`
- `garage_status.html`
- `garage_latest_binned.png`
- `garage_events.jsonl`
- `garage_notifications.jsonl`

`garage_latest_binned.png` is a symlink to the latest cached binned image so the
status page can show the same image that was scored. These generated files are
ignored by git.

## Phone Status Page

On the Pi, while connected to Tailscale:

```bash
cd /home/pi/garage
python3 -m http.server 8765
```

Then open:

```text
http://<pi-tailscale-name-or-ip>:8765/garage_status.html
```

For a persistent setup, run a real web server or a systemd-managed HTTP server.

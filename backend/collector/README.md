# Beverage solution collector

Runs as its own service (`systemctl status score-collector`), separate from the labeler web app.

Flow: chain poll (SN44 commitments, turbovision venv) -> wait for public repo (<= 1h from commit) ->
download pinned revision (weights.onnx <= 12 MB total, miner.py, small configs) -> sandboxed run
(Docker: no network, read-only, nobody, 2 CPU / 2 GB) on the latest 50 validated images ->
score s (same formula as the site) -> gate (s >= 0.70, p95 <= 150 ms, load-aware) -> store.

- Validated image = labeled (Done) AND all reference miners plausible (3/3). See `gtset.py`.
- Registry: `review_datasets/collector/collector.sqlite3`; passed packs in `.../packs/<id>/`.
- Web UI: `/collected`. Commands (refresh, add repo, re-measure, retry, push) are rows in `commands`.
- Push: optional deep post-process (sandboxed), optional ONNX metadata hash change, re-verified in the
  sandbox, then uploaded to YOUR private HF repo. The HF token goes through a 0600 tmpfs file only.

Settings (environment of the service): COLLECTOR_BACKFILL_DAYS, COLLECTOR_GATE_MIN_S, COLLECTOR_GATE_N,
COLLECTOR_GATE_MAX_P95_MS, COLLECTOR_MAX_ONNX_BYTES, COLLECTOR_ELEMENT (default "beverage"), ...
See `config.py`. Sandbox image: `docker/Dockerfile` (rebuilt automatically if missing).

## Execution environment

- Runtime: Python 3.10+ on the Raspberry Pi 5 (Raspberry Pi OS Bookworm 64-bit) or
  Linux/macOS desktops. Inference is **ONNX-only** — no PyTorch at runtime.
  `.pt` weights are exported to `.onnx` via `export_models.py` (this happens
  automatically in the Docker builder stage).
- Camera: Picamera2 for the CSI camera on the Pi 5 (OpenCV cannot read CSI
  directly); OpenCV/V4L2 for USB webcams on desktop.
- NumPy/OpenCV are installed from apt (`python3-numpy`, `python3-opencv`), not
  pip, to avoid conflicts with `python3-picamera2`. Only `onnxruntime` comes
  from pip in the runtime image.
- Preferred way to run: Docker via the Makefile — `make build`, `make up` /
  `make up-d`, `make logs`, `make shell`, `make down`. Bare-metal Pi and
  desktop-dev paths are documented in the README Quick Start.
- Container engine: the Makefile auto-detects — Docker is preferred (it's the
  documented Pi deployment target), and falls back to `podman compose` when
  `docker` isn't on PATH (e.g. dev machines that only have Podman installed).
  Both are invoked uniformly as `<engine> compose ...`.
- Never install packages directly on a development host (no `pip install`,
  `apt install`, etc. outside a container) when developing/testing here — use
  `make build && make shell` (or `make up`) instead. This is about the dev
  workflow only; it doesn't contradict the README's bare-metal Pi/desktop
  paths, which are separate, documented deployment options on an end user's
  own hardware.
- The container starts as root (`docker-entrypoint.sh`) to fix ownership on
  mounted volumes (`/app/logs`, `/tmp/libcamera`, `/tmp/Ultralytics`), then
  drops to the `salte` user before running the main command.
- Target: ~30 FPS on a Raspberry Pi 5 with no GPU — keep the hot path
  (`main.py` loop, `detectors/`) lean; avoid adding per-frame allocations or
  heavyweight abstractions there.

## Code style

- Functions: 4-20 lines. Split if longer.
- Files: under 500 lines. Split by responsibility.
- One thing per function, one responsibility per module (SRP).
- Names: specific and unique. Avoid `data`, `handler`, `Manager`.
  Prefer names that return <5 grep hits in the codebase.
- Types: explicit type hints on function signatures. Avoid bare/untyped
  `dict` or `Any` — use concrete types (`dataclass`, `TypedDict`, etc.).
- No code duplication. Extract shared logic into a function/module.
- Early returns over nested ifs. Max 2 levels of indentation.
- Exception messages must include the offending value and expected shape.

## Comments

- Keep your own comments. Don't strip them on refactor — they carry
  intent and provenance.
- Write WHY, not WHAT. Skip `# increment counter` above `i += 1`.
- Docstrings on public functions: intent + one usage example.
- Reference issue numbers / commit SHAs when a line exists because
  of a specific bug or upstream constraint.

## Tests

- No automated test suite exists yet. If you add one, use `pytest` and wire
  it up so it runs with a single command (document that command here once
  it exists).
- Every new function gets a test. Bug fixes get a regression test.
- Mock external I/O (camera, ONNX runtime, filesystem) with named fake
  classes, not inline stubs.
- Tests must be F.I.R.S.T: fast, independent, repeatable,
  self-validating, timely.

## Dependencies

- Inject dependencies through constructor/parameter, not global/import.
- Wrap third-party libs behind a thin interface owned by this project
  (e.g. `detectors/yolo_onnx.py` wraps `onnxruntime` — follow that pattern
  for new third-party integrations).

## Structure

- Follow the existing layout: `main.py` (entry point + camera init),
  `detectors/` (model inference wrappers), `behaviors/` (per-behavior state
  machines), `utils/` (shared helpers), `config.py` (global thresholds).
- Prefer small focused modules over god files.
- New detected behaviors go in `behaviors/`, new model wrappers in
  `detectors/`, cross-cutting helpers in `utils/`.

## Formatting

- Use `black`. Don't discuss style beyond that.

## Logging

- Runtime/event logging follows the existing CSV convention (see
  `eventos.csv` written from `main.py`) — keep new event logs in that
  format rather than introducing a second logging scheme.
- Plain text only for user-facing CLI output.

## Git

- Never add a "Co-Authored-By" line or any AI/tool attribution to commit
  messages in this project.
- One commit per finished functionality/change — don't bundle unrelated
  changes into a single commit.
- Never push to the remote without an explicit user request each time;
  committing locally is the normal flow, pushing is a separate action that
  needs authorization every time.
- Commit message subject: short and concise — a few words, dash-separated,
  as `type: short-dash-case-summary` (e.g. `add: retry-sync-backup`,
  `fix: timeout-ap-configuravel`). No long prose subject lines.

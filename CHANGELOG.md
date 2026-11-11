# Changelog

## 1.0.0 — 2026-11-12

First public release. The original artefact was a single Kaggle notebook that
ran five milestones in one kernel; this release turns it into an importable
package without changing the logic.

### Added
- `betacard/` package: `config`, `m1_design`, `m2_render`, `m3_compose`,
  `m4_service`, `m5_ui`.
- `tests/test_m1_design.py` — 11 tests for the CPU-only design engine.
- `scripts/preflight.py` and `scripts/bootstrap.py` lifted out of the notebook's
  environment cells.
- `LICENSE` (MIT), requirements files, `.gitignore`.

### Changed
- **Portable paths.** `BETACARD_BASE` defaults to `/kaggle/working/betacard`
  only when that directory exists, otherwise `~/betacard`. The hard-coded
  Kaggle work dir is now derived from that base.
- **The M1 guard is an import.** The notebook checked `globals()` for M1 names
  and told the user to paste a cell. The package imports M1 instead, so the
  failure is a normal `ImportError` at import time.
- **`from __future__ import annotations` hoisted** in `m5_ui`, where it had
  ended up below other imports and made the module a syntax error.
- The `CONFIG` dict moved out of the M2 imports cell into `config.py`, so every
  milestone reads one source of truth.

### Unchanged, deliberately
- Each milestone still runs its self-check on import. The checks are the
  specification and they were written to sit next to the code.
- The watermarker stays **on** by default. Turning it off is a legal and
  ethical decision, not a technical one, and remains an explicit flag.

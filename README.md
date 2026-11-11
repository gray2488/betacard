# Beta Card

**A greeting-card pipeline that designs twelve cards, renders them, and never
bakes the text into the image.**

Give it a name and a date of birth. It produces a twelve-card catalog — each card
with its own palette, layout, motif, texture, typography and written message —
renders the backgrounds on a GPU, composes the text on top with contrast-safe
placement, and serves the result through a Flask API.

---

## Why it is built the way it is

Text rendered by a diffusion model is unreliable: letters melt, kerning drifts,
and a name comes out misspelled often enough to embarrass the sender. So the
pipeline splits the job. The model produces **only the background art**, from a
prompt that explicitly forbids lettering. The text is drawn afterwards by a
compositor that knows the font, the colour contrast and the safe region.

Everything else follows from that decision:

- **Designs are deterministic.** `build_catalog_plan(session_id, person)` is a
  pure function of its inputs. Kill the kernel halfway through a render and the
  same twelve designs come back, which is what makes a run resumable.
- **Contrast is enforced, not hoped for.** Every palette is checked against
  WCAG AA (4.5:1) before it is allowed into the library, and the compositor
  re-checks the ratio it actually achieves.
- **Cards cannot look alike.** A collision resolver guarantees twelve unique
  (palette, layout, motif) triples in one catalog and re-rolls when two collide.

---

## The five milestones

| Module | Lines | What it does |
|---|---:|---|
| `m1_design.py` | 1,166 | Design token library, personalisation, template bank, catalog planner. No GPU, no network, no I/O. |
| `m2_render.py` | 1,131 | SDXL background rendering, content-addressed cache, manifest, VRAM watchdog. |
| `m3_compose.py` | 1,377 | Font resolution per script, message polish, text compositing with contrast checks. |
| `m4_service.py` | 1,889 | Storage and catalog adapters (local or Supabase), job queue, Flask API, tunnel. |
| `m5_ui.py` | 1,490 | Kivy client. |

Each milestone runs its own self-check on import, exactly as the notebook did —
thirteen sections in M1 alone, from contrast ratios to ordinals to collision
resolution. That is deliberate: the checks are the specification.

---

## Quickstart

M1 needs nothing but the standard library:

```bash
pip install -r requirements.txt
python -m betacard.m1_design
```

```
========================================================================
M1 SELF-CHECK (REVISION 2) - design token library and template bank
========================================================================
[1] Palette contrast (text on background, WCAG AA >= 4.5)
  PASS  midnight_gold        ratio 12.04
  ...
[13] AGA findings 2-4 verified as resolved
  PASS  finding 2: pick_text_colour exists (reserved for M3)
  ...
RESULT: ALL CHECKS PASSED
```

The full pipeline runs without a GPU in mock mode — the renderer is replaced by
a gradient generator, everything else is real:

```bash
export BETACARD_MOCK=1 BETACARD_LOCAL=1
python -c "from betacard import m4_service; print(m4_service.app)"
```

A real run needs a GPU with SDXL and the text model:

```bash
pip install -r requirements-gpu.txt
python scripts/preflight.py     # are the weights and secrets in place?
python scripts/bootstrap.py     # install only what is missing
```

---

## What a catalog looks like

Twelve cards, generated from `sess_abc123` for *Aayush*, turning 24:

| # | Palette | Layout | Motif | Texture | Tone |
|---|---|---|---|---|---|
| 0 | Dusty Rose | Lower Band | Bokeh Lights | Linen Weave | poetic |
| 1 | Ocean Teal | Top Banner | Blossom Branch | Watercolour | formal |
| 2 | Warm Amber | Left Panel | Geometric | Matte Film | sentimental |
| 3 | Charcoal Neon | Bottom Caption | Botanical | Paper Grain | celebratory |
| 4 | Sage Terracotta | Centre Card | Stars and Moon | Soft Gradient | playful |
| 5 | Blush Cream | Right Panel | Cake | Risograph | warm |
| 6 | Cotton Candy | Lower Band | Floral Wreath | Linen Weave | funny |
| 7 | Lavender Mist | Top Banner | Confetti | Watercolour | reflective |
| 8 | Royal Plum | Left Panel | Paper Cut | Matte Film | poetic |
| 9 | Forest Moss | Bottom Caption | Balloons | Paper Grain | formal |
| 10 | Ivory Ink | Centre Card | Art Deco | Soft Gradient | sentimental |
| 11 | Midnight Gold | Right Panel | Gift and Ribbon | Risograph | celebratory |

Two sample messages from that run:

> *poetic, short* — "The years arrive like tide, Aayush. You remain unbothered."

> *funny, medium* — "Aayush, the candles are a fire hazard and the cake is a lie.
> Happy 24th anyway."

Every card carries a `cache_key` derived from its design, so a background
rendered once is never rendered twice.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BETACARD_BASE` | `/kaggle/working/betacard` or `~/betacard` | Working directory for cache, cards and manifests |
| `BETACARD_HF_CACHE` | sibling of the base dir | Hugging Face cache |
| `BETACARD_MOCK` | `0` | `1` replaces the renderer with a gradient generator |
| `BETACARD_LOCAL` | unset | `1` uses local storage instead of Supabase |
| `BETACARD_SDXL_PATH` | Hub id | Local SDXL checkpoint |
| `BETACARD_VAE_PATH` | Hub id | Local VAE |
| `BETACARD_TEXT_MODEL_PATH` | Hub id | Local text model |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | unset | Only for the Supabase adapters |
| `API_BEARER_TOKEN` | unset | Required by the Flask API outside local mode |
| `NGROK_AUTHTOKEN` | unset | Only for the public tunnel |

---

## Project layout

```
betacard/
├── betacard/
│   ├── config.py       portable paths, render ladder, VRAM thresholds
│   ├── m1_design.py    tokens, personalisation, catalog planner
│   ├── m2_render.py    SDXL rendering, cache, manifest, watchdog
│   ├── m3_compose.py   fonts, text, compositing
│   ├── m4_service.py   storage, jobs, Flask API
│   └── m5_ui.py        Kivy client
├── scripts/
│   ├── preflight.py    are the weights and secrets present?
│   └── bootstrap.py    install only what is missing
└── tests/
    └── test_m1_design.py
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 11 passed
```

The suite covers the parts that can be tested without a GPU: ordinals, script
detection, DOB parsing and its rejection cases, WCAG contrast arithmetic,
catalog size, determinism across identical session ids, divergence across
different ones, twelve unique design triples, the name appearing in every
message, and JSON-safety of the exported records.

---

## Known limitations

- **The Kivy client was never executed.** M5 is written and self-checks its
  pure logic, but `kivy` was not installed in the environment where this was
  prepared, so the UI has not been run.
- **No real GPU render is included here.** Every check above ran in mock mode.
  Real SDXL output requires a CUDA machine with ~7 GB of weights.
- **The watermark flag is an open decision.** Stability AI attaches an invisible
  watermark to SDXL output. Turning it off is a legal and ethical choice rather
  than a technical one, so the default is **on** and the flag is left for an
  explicit decision.
- **Milestones execute on import.** The notebook ran top-level cells; the
  modules keep that behaviour so the self-checks stay with the code. Importing
  `m4_service` therefore creates directories and runs its checks.
- **Kaggle-era paths.** Defaults still point at `/kaggle/working` when that
  directory exists. Everything is overridable by environment variable.
- **Supabase is optional and untested here.** The local adapters are the ones
  exercised by the checks.

---

## Licence

MIT. See [LICENSE](LICENSE).

Built by **Group of AI (GoA)**.

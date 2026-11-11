"""Runtime configuration.

Paths are portable: ``BETACARD_BASE`` is set to ``/kaggle/working/betacard``
when that directory exists, otherwise to ``~/betacard``. Every other path
and secret is an environment variable with a safe default.
"""

# [env-fix] RUN MODE + portable workspace.
# Kaggle uses /kaggle/working; anywhere else (laptop/Colab) a local folder.
# Pre-set BETACARD_BASE to override.
import os

if not os.environ.get('BETACARD_BASE'):
    if os.path.isdir('/kaggle/working'):
        _BASE = '/kaggle/working/betacard'
    else:
        _BASE = os.path.join(os.path.expanduser('~'), 'betacard')
    os.environ['BETACARD_BASE'] = _BASE
os.environ.setdefault('BETACARD_HF_CACHE',
                      os.path.join(os.path.dirname(
                          os.environ['BETACARD_BASE']), 'hf_cache'))
print('[env] workspace:', os.environ['BETACARD_BASE'])

# [env-fix] Heavy models as Kaggle Datasets (read-only /kaggle/input).
# A dataset needs no download, so nothing lands in the 20 GB working
# quota and startup is fast. Rename to match your datasets, or set the
# env var directly, e.g. os.environ['BETACARD_VAE_PATH']='/kaggle/input/my-vae'.
# SDXL is NOT read from a dataset: its dataset holds only 609 B, so we
# always download SDXL from the Hub into the 20 GiB working dir (~6.5 GiB).
for _key, _default in (
        ('BETACARD_VAE_PATH', '/kaggle/input/betacard-vae'),
        ('BETACARD_TEXT_MODEL_PATH', '/kaggle/input/betacard-qwen')):
    os.environ.setdefault(_key, _default)
    if os.path.isdir(os.environ[_key]):
        print('[env] %s -> local dataset %s' % (_key, os.environ[_key]))
    else:
        os.environ[_key] = ''
        print('[env] %s not mounted; will use the Hugging Face id' % _key)

# Force a local test run (no Supabase/catalog cloud): uncomment next line.
# os.environ['BETACARD_LOCAL'] = '1'


CONFIG = {
    # [env-fix] Prefer a locally-mounted Kaggle Dataset (read-only /kaggle/input)
    # over a Hugging Face download, so nothing lands in the 20 GB working quota.
    # Set BETACARD_SDXL_PATH / BETACARD_VAE_PATH in the top cell to point here.
    "model_id": (os.environ.get("BETACARD_SDXL_PATH")
                 or "stabilityai/stable-diffusion-xl-base-1.0"),
    "vae_id": (os.environ.get("BETACARD_VAE_PATH")
               or "madebyollin/sdxl-vae-fp16-fix"),
    # [env-fix] portable base: Kaggle's /kaggle/working, else a local dir
    # (the notebook's top "RUN MODE" cell sets BETACARD_BASE).
    "work_dir": os.environ["BETACARD_BASE"],
    "hf_cache_dir": (os.environ.get("BETACARD_HF_CACHE")
                     or "/kaggle/working/hf_cache"),

    # [F9] FLAGGED FOR DECISION.
    # Stability AI attaches an invisible watermark to SDXL output. Turning it
    # off is a legal/ethical choice, not a technical one, and the directive
    # does not mention it. The safe default is ON; flip this only with an
    # explicit order.
    "disable_watermarker": False,

    # [F10] Mock render delay, seconds. Configurable so it can be set to 0
    # in a tight test loop.
    "mock_render_delay_s": 0.02,

    # Two-stage rendering. Draft = 8 steps; final = 30.
    "draft": {"steps": 8,  "size": 1024, "guidance": 5.0},
    "final": {"steps": 30, "size": 1024, "guidance": 7.0},

    # Watchdog is now an EMERGENCY BRAKE, not the primary cleanup path.
    # Cleanup runs after every render regardless (F1 fix).
    "watchdog_threshold_gb": 13.0,
    "watchdog_hard_limit_gb": 15.0,
    "kernel_restart_every_n_jobs": 24,

    # VRAM optimization ladder (directive section 7).
    "vram_ladder": {
        "fp16_weights":          True,   # Rung 1 mandatory
        "sdpa_attention":        True,   # Rung 2 mandatory
        "vae_slicing":           True,   # Rung 3 mandatory
        "fixed_fp16_vae":        True,   # Rung 4 mandatory
        "vae_tiling":            False,  # Rung 5 conditional
        "model_cpu_offload":     False,  # Rung 6 conditional
        "resolution_cap":        1024,   # Rung 7 mandatory
        "batch_size":            1,      # Rung 8 mandatory
        "sequential_cpu_offload": False, # Rung 9 last resort
    },
}

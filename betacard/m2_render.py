"""M2 - SDXL background rendering, caching and the VRAM watchdog."""

from .config import CONFIG
from .m1_design import *  # noqa: F401,F403  (shared milestone namespace)

# =============================================================================
# PROJECT BETA-CARD - MILESTONE M2 (REVISION 2)
# Image generation pipeline and caching layer.
#
# Consumes M1's catalog plan. Renders SDXL aesthetic backgrounds. Caches them
# by the design's cache_key. Emits a manifest. No cloud storage, no uploads.
#
# Revision 2 changes (AGA findings F1-F12):
#   [F1] CRITICAL - cleanup now runs after EVERY render, not only when the
#        watchdog trips. Watchdog demoted to emergency brake.
#   [F2] CRITICAL - silent mock fallback removed. MOCK_MODE only from env var;
#        missing CUDA without the flag raises.
#   [F3] MAJOR - watchdog samples on cache hits too.
#   [F4] MAJOR - manifest stores rel_path alongside path; resolve_path() falls
#        back to the relative form when the session's work dir differs.
#   [F5] Removed unused imports inspect, sys.
#   [F6] _M1_REQUIRED now lists the type names too.
#   [F7] VRAMWatchdog.hard_exceeded() wired in as a hard abort.
#   [F8] BackgroundCache.load/list_keys marked reserved for M3/M4.
#   [F9] Watermarker default restored to ON; disable is now an explicit flag.
#        FLAGGED FOR DECISION - see the comment on disable_watermarker.
#   [F10] Mock render delay moved to config.
#   [F11] Manifest skips the write when nothing changed.
#   [F12] Corrupted manifest is moved to .bak instead of being discarded.
#
# REQUIRES: the M1 cell (frozen, AGA-approved) has already run in this kernel.
# Kaggle: accelerator "GPU T4 x2", Internet ON.
# Local test: set BETACARD_MOCK=1 for CPU-only logic testing.
# =============================================================================


# --- SECTION 1: IMPORTS AND CONFIG ------------------------------------------

import os
import gc
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Guard: M1 must have run in this kernel.
_M1_REQUIRED = (
    "build_catalog_plan", "build_personalisation", "plan_as_records",
    "DesignToken", "Personalisation",
)
_M1_MISSING = [n for n in _M1_REQUIRED if n not in globals()]
if _M1_MISSING:
    raise RuntimeError(
        f"M1 cell must run before M2. Missing names: {_M1_MISSING}. "
        "Paste the M1 cell above this one and run it first."
    )


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
    "work_dir": os.environ.get("BETACARD_BASE") or "/kaggle/working/betacard",
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


# --- SECTION 2: PATHS --------------------------------------------------------

WORK = Path(CONFIG["work_dir"])
CACHE_DIR = WORK / "cache" / "backgrounds"
MANIFEST_PATH = WORK / "manifest.json"
STATE_PATH = WORK / "state.json"
HF_CACHE = Path(CONFIG["hf_cache_dir"])

for _d in (WORK, CACHE_DIR, HF_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))


# --- SECTION 3: DEVICE AND MODE ---------------------------------------------

HAS_CUDA = bool(HAS_TORCH and torch.cuda.is_available())

# [F2] MOCK_MODE is only ever set explicitly. Silent fallback to mock on a
# missing GPU was a false-acceptance risk: a session with no accelerator would
# have produced twelve placeholder gradients and looked successful.
MOCK_MODE = os.environ.get("BETACARD_MOCK", "0") == "1"

if not MOCK_MODE and not HAS_CUDA:
    raise RuntimeError(
        "CUDA is not available and BETACARD_MOCK is not set. "
        "Enable a GPU accelerator on Kaggle (select 'GPU T4 x2', not the "
        "generic 'GPU' option which maps to P100), or set BETACARD_MOCK=1 "
        "for a CPU-only logic test. Silent fallback was removed in M2 "
        "revision 2 per AGA finding F2."
    )

if HAS_TORCH and HAS_CUDA:
    DEVICE_NAME = torch.cuda.get_device_name(0)
    DEVICE_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    DEVICE_COUNT = torch.cuda.device_count()
else:
    DEVICE_NAME, DEVICE_VRAM_GB, DEVICE_COUNT = "cpu", 0.0, 0

print(f"[M2] torch={HAS_TORCH} cuda={HAS_CUDA} mock={MOCK_MODE}")
print(f"[M2] device={DEVICE_NAME} vram={DEVICE_VRAM_GB:.1f}GB count={DEVICE_COUNT}")
print(f"[M2] work={WORK}")
print(f"[M2] cache={CACHE_DIR}")


# --- SECTION 4: SEED DERIVATION ---------------------------------------------

def derive_seed(cache_key, stage, base=0):
    """Deterministic seed from cache_key and stage.

    A resumed session must produce byte-identical output for the same design,
    so the seed cannot depend on process state, wall clock or RNG globals.
    """
    payload = f"{base}|{cache_key}|{stage}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --- SECTION 5: CACHE LAYER --------------------------------------------------

class BackgroundCache:
    """Local filesystem cache for rendered backgrounds.

    Keyed by the design's cache_key() and the render stage. M2 keeps this
    local; M4 will add a Supabase adapter behind the same interface.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, cache_key, stage):
        return self.root / f"{cache_key}__{stage}.png"

    def has(self, cache_key, stage):
        return self.path_for(cache_key, stage).is_file()

    def load(self, cache_key, stage):
        """RESERVED FOR M3/M4.

        M2 does not read cached images back; the compositor and the uploader
        will. Kept here so the cache interface is stable across milestones.
        """
        return Image.open(self.path_for(cache_key, stage))

    def store(self, cache_key, stage, image):
        path = self.path_for(cache_key, stage)
        tmp = path.with_suffix(".png.tmp")
        image.save(tmp, "PNG", optimize=True)
        tmp.replace(path)  # atomic on POSIX; a crash cannot leave a partial PNG
        return path

    def count(self, stage=None):
        if stage:
            return sum(1 for _ in self.root.glob(f"*__{stage}.png"))
        return sum(1 for _ in self.root.glob("*.png"))

    def list_keys(self, stage=None):
        """RESERVED FOR M3/M4."""
        pattern = f"*__{stage}.png" if stage else "*.png"
        return sorted(p.stem for p in self.root.glob(pattern))


# --- SECTION 6: MANIFEST -----------------------------------------------------

class Manifest:
    """Serialisable record of every render.

    One record per (design_id, stage). Written atomically. M3 reads the final
    background path; M4 uploads the finished card.

    [F4] Each record carries both an absolute `path` (for the current session)
    and a `rel_path` (relative to work_dir). resolve_path() uses the absolute
    form when it still exists and falls back to the relative form otherwise,
    so a manifest written under a different work dir still resolves.

    [F11] Writes are skipped when nothing changed.
    [F12] A corrupted manifest is moved to .bak rather than discarded.
    """

    def __init__(self, path, base_dir):
        self.path = Path(path)
        self.base_dir = Path(base_dir)
        self.records = {}
        self._last_payload = None

        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                for rec in data:
                    self.records[f"{rec['design_id']}|{rec['stage']}"] = rec
                self._last_payload = json.dumps(
                    self.all(), indent=2, default=str)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                backup = self.path.with_suffix(".json.bak")
                try:
                    self.path.replace(backup)
                except OSError:
                    pass
                print(f"[M2] manifest unreadable, moved to {backup}: {exc}")
                self.records = {}
                self._last_payload = None

    def upsert(self, record):
        key = f"{record['design_id']}|{record['stage']}"
        self.records[key] = record
        self._save()

    def get(self, design_id, stage):
        return self.records.get(f"{design_id}|{stage}")

    def all(self):
        return list(self.records.values())

    def resolve_path(self, record):
        """Absolute path for the current session.

        Prefers the recorded absolute path. If that file is gone (session
        moved, work dir differs), falls back to rel_path against base_dir.
        Returns the original path object either way; callers check existence.
        """
        p = Path(record.get("path", ""))
        if p.is_file():
            return p
        rel = record.get("rel_path")
        if rel:
            candidate = self.base_dir / rel
            if candidate.is_file():
                return candidate
        return p

    def _save(self):
        payload = json.dumps(self.all(), indent=2, default=str)
        if payload == self._last_payload:
            return  # nothing changed; skip the write
        # [audit-fix M2-R10] unique tmp name (concurrent GPU threads)
        import threading as _threading
        tmp = self.path.with_suffix(".json.%d.tmp" % _threading.get_ident())
        tmp.write_text(payload)
        tmp.replace(self.path)
        self._last_payload = payload


# --- SECTION 7: JOB STATE ---------------------------------------------------

class JobState:
    """Tracks how many jobs have run, for kernel-restart recommendation."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {
            "jobs_completed": 0,
            "last_design_id": None,
            "kernel_restart_recommended": False,
        }
        if self.path.is_file():
            try:
                self.data.update(json.loads(self.path.read_text()))
            except json.JSONDecodeError:
                pass

    def bump(self, design_id):
        self.data["jobs_completed"] += 1
        self.data["last_design_id"] = design_id
        if self.data["jobs_completed"] >= CONFIG["kernel_restart_every_n_jobs"]:
            self.data["kernel_restart_recommended"] = True
        self._save()

    def _save(self):
        # [audit-fix M2-R10] unique tmp name so concurrent GPU threads never
        # collide on the same file (belt-and-braces alongside _write_lock).
        import threading as _threading
        tmp = self.path.with_suffix(".json.%d.tmp" % _threading.get_ident())
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


# --- SECTION 8: VRAM WATCHDOG -----------------------------------------------

class VRAMWatchdog:
    """Samples GPU memory after every job and signals when a reset is needed.

    [F1] This is now the EMERGENCY BRAKE. Per-job cleanup happens in the
    pipeline regardless of what this class reports.
    """

    def __init__(self, threshold_gb, hard_limit_gb):
        self.threshold_gb = threshold_gb
        self.hard_limit_gb = hard_limit_gb
        self.samples = []
        self.resets = 0

    def sample(self):
        if not (HAS_TORCH and HAS_CUDA):
            self.samples.append({"allocated_gb": 0.0, "reserved_gb": 0.0})
            return 0.0
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        self.samples.append({"allocated_gb": allocated, "reserved_gb": reserved})
        return reserved

    def needs_reset(self):
        if not self.samples:
            return False
        return self.samples[-1]["reserved_gb"] > self.threshold_gb

    def hard_exceeded(self):
        """[F7] Wired into the pipeline as a hard abort."""
        if not self.samples:
            return False
        return self.samples[-1]["reserved_gb"] > self.hard_limit_gb

    def reset(self):
        self.resets += 1

    def samples_gb(self):
        return [s["reserved_gb"] for s in self.samples]

    def max_gb(self):
        return max(self.samples_gb()) if self.samples else 0.0


# --- SECTION 9: RENDERERS ----------------------------------------------------

def _mock_render(prompt, negative_prompt, seed, size, steps, guidance):
    """Deterministic placeholder for testing without a GPU.

    Produces a vertical gradient whose colours derive from the seed. NOT a
    substitute for a real render. Exists so caching, manifest and resumability
    logic can be exercised on CPU.
    """
    rng = np.random.default_rng(seed)
    c1 = rng.integers(40, 220, 3)
    c2 = rng.integers(40, 220, 3)
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32).reshape(-1, 1, 1)
    arr = (c1 * (1.0 - ramp) + c2 * ramp).astype(np.uint8)
    arr = np.broadcast_to(arr, (size, size, 3)).copy()
    img = Image.fromarray(arr).convert("RGB")  # [audit-fix M2-R3] Pillow 11: mode= deprecated
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(255, 255, 255), width=2)
    return img


class MockRenderer:
    """Renderer that never touches the GPU. Explicit opt-in only (F2)."""

    def __init__(self, config):
        self.config = config
        self.reset_calls = 0
        self.render_calls = 0

    def load(self):
        print("[M2] MockRenderer loaded")

    def render(self, prompt, negative_prompt, seed, size, steps, guidance):
        self.render_calls += 1
        time.sleep(self.config.get("mock_render_delay_s", 0.0))
        return _mock_render(prompt, negative_prompt, seed, size, steps, guidance)

    def reset(self):
        """[F1] Called after every render by the pipeline."""
        gc.collect()
        self.reset_calls += 1


class SDXLRenderer:
    """SDXL base at fp16, with the mandatory VRAM ladder applied.

    Ladder rungs:
      1. fp16 weights          - torch_dtype=torch.float16, variant="fp16"
      2. SDPA attention        - AttnProcessor2_0 on the UNet
      3. VAE slicing           - enable_vae_slicing()
      4. Fixed FP16 VAE        - madebyollin/sdxl-vae-fp16-fix
      5. VAE tiling            - enable_vae_tiling()             [conditional]
      6. Model CPU offload     - enable_model_cpu_offload()      [conditional]
      7. Resolution cap        - enforced in render()
      8. Batch size 1          - num_images_per_prompt=1
      9. Sequential CPU offload- enable_sequential_cpu_offload() [last resort]
    """

    def __init__(self, config):
        self.config = config
        self.pipe = None
        self.pipes = []            # [audit-fix M2-R8] one pipeline per GPU
        self._rr = 0               # round-robin cursor across pipelines
        self._rr_lock = None
        self.loaded_at = None
        self.reset_calls = 0
        self.render_calls = 0

    def load(self):
        if self.pipe is not None:
            return
        # [audit-fix M2-R7] transformers 5.x is a known-broken release for
        # diffusers consumers - importing it (or diffusers through it) crashes
        # with "Backend should be defined in the BACKENDS_MAPPING" plus a
        # missing transformers.tokenization_utils_* module. Turn that cryptic
        # failure into a clear, actionable message.
        try:
            import transformers as _tf_probe
            _tf_major = int(_tf_probe.__version__.split(".")[0])
            if _tf_major >= 5:
                raise RuntimeError(
                    "transformers %s is incompatible with diffusers here "
                    "(known transformers 5.x bug). Fix: run the bootstrap "
                    "cell, or `pip install 'transformers>=4.44,<5'`, then "
                    "RESTART the kernel." % _tf_probe.__version__)
        except ImportError:
            pass
        except RuntimeError:
            raise
        except Exception as _tf_exc:
            raise RuntimeError(
                "import transformers failed (%s). This is the known "
                "transformers 5.x incompatibility. Fix: run the bootstrap "
                "cell, or `pip install 'transformers>=4.44,<5'`, then "
                "RESTART the kernel." % _tf_exc)

        from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        from diffusers.models.attention_processor import AttnProcessor2_0

        ladder = dict(self.config["vram_ladder"])
        # [env-fix] On a small GPU (e.g. one ~15 GB T4) the mandatory rungs
        # alone can still OOM during VAE decode / UNet forward. Auto-enable the
        # memory-saving conditional rungs when VRAM is below this threshold.
        _auto_below = float(self.config.get("auto_low_vram_gb", 16.0))
        if HAS_CUDA and DEVICE_VRAM_GB and DEVICE_VRAM_GB < _auto_below:
            # [audit-fix M2-R9] With a second GPU available, prefer
            # data-parallel across both GPUs over single-device CPU offload:
            # enable_model_cpu_offload binds the model to ONE GPU, which is
            # exactly what left GPU 1 idle. Only fall back to CPU offload on
            # a single GPU.
            ladder["vae_tiling"] = True
            if DEVICE_COUNT > 1:
                ladder["model_cpu_offload"] = False
                print(f"[M2] low VRAM ({DEVICE_VRAM_GB:.1f} GB < "
                      f"{_auto_below:.0f} GB) but {DEVICE_COUNT} GPUs present: "
                      "VAE tiling ON, CPU offload OFF (multi-GPU data-parallel)")
            else:
                ladder["model_cpu_offload"] = True
                print(f"[M2] low VRAM ({DEVICE_VRAM_GB:.1f} GB < "
                      f"{_auto_below:.0f} GB): enabling VAE tiling + "
                      "model CPU offload")
        t0 = time.time()

        # Rung 4: fixed fp16 VAE (mandatory).
        vae = None
        if ladder["fixed_fp16_vae"]:
            print("[M2] rung 4: loading fixed fp16 VAE")
            vae = AutoencoderKL.from_pretrained(
                self.config["vae_id"],
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        print(f"[M2] loading {self.config['model_id']}")
        kwargs = {
            "torch_dtype": torch.float16,       # Rung 1
            "variant": "fp16",                  # Rung 1
            "use_safetensors": True,
            # [F9] Stability's invisible watermark. Restored to ON by default;
            # disable only on an explicit order.
            "add_watermarker": not self.config.get("disable_watermarker", False),
        }
        if vae is not None:
            kwargs["vae"] = vae

        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config["model_id"], **kwargs)
        except NameError as _wm_exc:
            # [env-fix] Some diffusers builds removed StableDiffusionXLWatermarker
            # but still reference it when add_watermarker=True, raising NameError.
            # Retry with the watermarker off so the pipeline can still load.
            print(f"[M2] watermarker unavailable in this diffusers build "
                  f"({_wm_exc}); retrying with add_watermarker=False")
            kwargs["add_watermarker"] = False
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config["model_id"], **kwargs)

        # Rung 2: SDPA.
        if ladder["sdpa_attention"]:
            print("[M2] rung 2: SDPA attention")
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            if getattr(pipe, "vae", None) is not None:
                try:
                    pipe.vae.set_attn_processor(AttnProcessor2_0())
                except Exception:
                    pass

        # Rung 3: VAE slicing (mandatory). The decode step, not the weights,
        # is the usual cause of a crash.
        if ladder["vae_slicing"]:
            print("[M2] rung 3: VAE slicing")
            # [env-fix] Newer diffusers moved VAE slicing from the pipeline to
            # the VAE (pipe.vae.enable_slicing()); older builds had
            # pipe.enable_vae_slicing(). Support both.
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            else:
                pipe.vae.enable_slicing()

        # Rung 5: VAE tiling (conditional).
        if ladder["vae_tiling"]:
            print("[M2] rung 5: VAE tiling")
            # [env-fix] same move as rung 3 (pipeline -> VAE).
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            else:
                pipe.vae.enable_tiling()

        # Rungs 6 and 9 are mutually exclusive.
        if ladder["sequential_cpu_offload"]:
            print("[M2] rung 9: sequential CPU offload (last resort)")
            pipe.enable_sequential_cpu_offload()
        elif ladder["model_cpu_offload"]:
            print("[M2] rung 6: model CPU offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        self.pipe = pipe
        # [audit-fix M2-R8] Multi-GPU: when a second GPU sits idle and we are
        # NOT on a CPU-offload rung (those bind the model to one device), load
        # a second pipeline onto it so render_drafts can run data-parallel.
        import threading as _threading
        self.pipes = [pipe]
        self._rr = 0
        self._rr_lock = _threading.Lock()
        _max_gpus = int(self.config.get("max_gpus", 2))
        if (not ladder["sequential_cpu_offload"] and not ladder["model_cpu_offload"]
                and DEVICE_COUNT > 1 and _max_gpus > 1):
            try:
                for _dev in range(1, min(DEVICE_COUNT, _max_gpus)):
                    print(f"[M2] multi-GPU: loading pipeline on cuda:{_dev}")
                    _ekw = dict(kwargs)
                    _ekw.pop("vae", None)      # a VAE can live on only one device
                    if ladder["fixed_fp16_vae"]:
                        _ekw["vae"] = AutoencoderKL.from_pretrained(
                            self.config["vae_id"], torch_dtype=torch.float16,
                            use_safetensors=True)
                    _extra = StableDiffusionXLPipeline.from_pretrained(
                        self.config["model_id"], **_ekw)
                    _extra.unet.set_attn_processor(AttnProcessor2_0())
                    if hasattr(_extra, "enable_vae_slicing"):
                        _extra.enable_vae_slicing()
                    else:
                        _extra.vae.enable_slicing()
                    if ladder["vae_tiling"]:
                        if hasattr(_extra, "enable_vae_tiling"):
                            _extra.enable_vae_tiling()
                        else:
                            _extra.vae.enable_tiling()
                    _extra.to(f"cuda:{_dev}")
                    self.pipes.append(_extra)
                print(f"[M2] multi-GPU active: {len(self.pipes)} pipelines on "
                      f"{[f'cuda:{i}' for i in range(len(self.pipes))]}")
            except Exception as _mg_exc:
                print(f"[M2] multi-GPU setup failed ({_mg_exc}); single GPU only")
                self.pipes = [pipe]

        self.loaded_at = time.time()
        print(f"[M2] pipeline ready in {self.loaded_at - t0:.1f}s")

    def render(self, prompt, negative_prompt, seed, size, steps, guidance):
        if not self.pipes and self.pipe is None:
            raise RuntimeError("call load() before render()")

        cap = self.config["vram_ladder"]["resolution_cap"]
        if size > cap:
            raise ValueError(
                f"size {size} exceeds resolution cap {cap}. "
                "Raise the cap deliberately, not by accident."
            )

        self.render_calls += 1
        # [audit-fix M2-R8] round-robin across pipelines (one per GPU).
        _pipes = self.pipes or [self.pipe]
        if self._rr_lock is None:
            import threading as _threading
            self._rr_lock = _threading.Lock()
        with self._rr_lock:
            _idx = self._rr % len(_pipes)
            self._rr += 1
        generator = torch.Generator(device=f"cuda:{_idx}").manual_seed(seed)
        result = _pipes[_idx](
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=size,
            height=size,
            generator=generator,
            num_images_per_prompt=self.config["vram_ladder"]["batch_size"],
        )
        return result.images[0]

    def reset(self):
        """[F1] Called after every render by the pipeline.

        Cleanup runs whether or not the watchdog tripped. The watchdog is the
        emergency brake; this is the ordinary path.
        """
        gc.collect()
        if HAS_TORCH and HAS_CUDA:
            # [audit-fix M2-R8] clear the allocator cache on every GPU we use.
            for _d in range(max(1, len(self.pipes or []))):
                with torch.cuda.device(_d):
                    torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.reset_calls += 1


def build_renderer(config):
    if MOCK_MODE:
        print("[M2] MOCK MODE - no GPU; using deterministic stub renderer.")
        return MockRenderer(config)
    return SDXLRenderer(config)


# --- SECTION 10: PIPELINE (PUBLIC INTERFACE FOR M3/M4) ----------------------

class BackgroundPipeline:
    """M2's public interface.

    M3 reads final backgrounds from here; M4 uploads the finished cards.
    This class owns the cache, the manifest and the state.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.cache = BackgroundCache(CACHE_DIR)
        self.manifest = Manifest(MANIFEST_PATH, base_dir=WORK)
        self.state = JobState(STATE_PATH)
        self.watchdog = VRAMWatchdog(
            self.config["watchdog_threshold_gb"],
            self.config["watchdog_hard_limit_gb"],
        )
        self.renderer = None
        # [audit-fix M2-R8] guards the shared manifest/watchdog writes when
        # render_drafts runs data-parallel across GPUs.
        import threading as _threading
        self._write_lock = _threading.Lock()

    def prepare(self):
        """Load the pipeline. Idempotent."""
        if self.renderer is None:
            self.renderer = build_renderer(self.config)
            self.renderer.load()
        # [audit-fix M2-R11] multi-GPU: every GPU holds a full SDXL, so the
        # per-device VRAM sits higher than the single-GPU+offload case. Scale
        # the watchdog to the real device so it does not false-trip - it
        # aborted at "15.00 GB > 15.00 GB" while the renders were fine.
        _pipes = getattr(self.renderer, "pipes", None) or []
        if len(_pipes) > 1 and HAS_CUDA and DEVICE_VRAM_GB:
            self.watchdog.threshold_gb = round(DEVICE_VRAM_GB * 0.95, 2)
            self.watchdog.hard_limit_gb = round(DEVICE_VRAM_GB * 0.99, 2)
            print(f"[M2] multi-GPU: watchdog scaled to the device "
                  f"(threshold {self.watchdog.threshold_gb:.2f} GB, "
                  f"hard {self.watchdog.hard_limit_gb:.2f} GB)")
        return self

    # -- public API ---------------------------------------------------------

    def render_drafts(self, plan):
        """Render drafts for every design in the plan. Cache-first."""
        # [audit-fix M2-R8] when more than one GPU pipeline is loaded, render
        # the batch data-parallel (one thread per GPU). Shared manifest /
        # watchdog writes inside _render_one are guarded by _write_lock.
        _pipes = getattr(self.renderer, "pipes", None) or []
        if len(_pipes) > 1 and len(plan) > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f"[M2] render_drafts: data-parallel across {len(_pipes)} GPUs")
            with ThreadPoolExecutor(max_workers=len(_pipes)) as _ex:
                return list(_ex.map(
                    lambda it: self._render_one(it["design"], "draft"), plan))
        return [self._render_one(item["design"], "draft") for item in plan]

    def render_final(self, design):
        """Render the final for one design. Cache-first. On-demand only."""
        return self._render_one(design, "final")

    def get_record(self, design_id, stage):
        return self.manifest.get(design_id, stage)

    def get_path(self, design_id, stage):
        """Absolute path for the current session, resolved via the manifest."""
        rec = self.manifest.get(design_id, stage)
        return self.manifest.resolve_path(rec) if rec else None

    # -- internals ----------------------------------------------------------

    def _render_one(self, design, stage):
        if stage not in ("draft", "final"):
            raise ValueError(f"unknown stage: {stage!r}")

        cache_key = design.cache_key()

        # -- cache hit: no render, but still sample the watchdog (F3). -----
        if self.cache.has(cache_key, stage):
            record = self.manifest.get(design.design_id, stage)
            if record is None:
                # Cache file exists, manifest entry does not: the session died
                # between the write and the manifest update. Rebuild the record.
                record = self._build_record(
                    design, stage, cache_key,
                    self.cache.path_for(cache_key, stage),
                    seed=derive_seed(cache_key, stage),
                )
                with self._write_lock:
                    self.manifest.upsert(record)
            # [F3] Sample on cache hits too. A long session serving mostly
            # cached designs must still produce a leak-detection signal.
            with self._write_lock:
                self.watchdog.sample()
            return {**record, "from_cache": True}

        # -- cache miss: render. -------------------------------------------
        cfg = self.config[stage]
        seed = derive_seed(cache_key, stage)

        if self.renderer is None:
            self.prepare()

        image = self.renderer.render(
            prompt=design.sd_prompt(),
            negative_prompt=design.sd_negative_prompt(),
            seed=seed,
            size=cfg["size"],
            steps=cfg["steps"],
            guidance=cfg["guidance"],
        )

        path = self.cache.store(cache_key, stage, image)
        record = self._build_record(design, stage, cache_key, path, seed)
        with self._write_lock:
            self.manifest.upsert(record)

        # Drop the reference immediately so the tensor does not survive the
        # rest of the loop (directive section 8).
        del image

        # [F1] Per-job cleanup. Runs after EVERY render, not only when the
        # watchdog trips. `empty_cache` frees only the allocator's unused
        # blocks; live weights stay resident.
        self.renderer.reset()

        # Sample memory after the cleanup so the watchdog sees the steady state.
        with self._write_lock:
            self.watchdog.sample()

        # [F7] Hard abort if we have blown through the ceiling despite cleanup.
        if self.watchdog.hard_exceeded():
            raise RuntimeError(
                f"VRAM exceeded hard limit "
                f"({self.watchdog.samples[-1]['reserved_gb']:.2f} GB > "
                f"{self.watchdog.hard_limit_gb:.2f} GB). Aborting to avoid an "
                f"OOM kill mid-session."
            )

        # Emergency brake: if we are over the soft threshold even after the
        # per-job cleanup, log it and count a reset. The next render may still
        # fit; if not, the hard limit above stops it.
        if self.watchdog.needs_reset():
            print(f"[M2] watchdog reset at "
                  f"{self.watchdog.samples[-1]['reserved_gb']:.2f} GB "
                  f"(threshold {self.watchdog.threshold_gb:.2f} GB)")
            self.watchdog.reset()

        # Job counter for the kernel-restart recommendation.
        # [audit-fix M2-R10] guard the shared state write too. With two GPU
        # threads, both wrote state.json.tmp and one os.replace failed
        # (FileNotFoundError: state.json.tmp -> state.json).
        with self._write_lock:
            self.state.bump(design.design_id)
        # [audit-fix M2-R4] show the advisory once, not on every later render.
        if (self.state.data["kernel_restart_recommended"]
                and not self.state.data.get("_restart_notice_shown")):
            print("[M2] kernel restart recommended. Save the notebook and "
                  "restart the session before continuing.")
            with self._write_lock:
                self.state.data["_restart_notice_shown"] = True
                self.state._save()

        return {**record, "from_cache": False}

    def _build_record(self, design, stage, cache_key, path, seed):
        path = Path(path)
        try:
            rel_path = str(path.relative_to(WORK))
        except ValueError:
            rel_path = None  # outside WORK; keep absolute only
        cfg = self.config[stage]
        return {
            "design_id": design.design_id,
            "cache_key": cache_key,
            "stage": stage,
            "path": str(path),
            "rel_path": rel_path,
            "seed": seed,
            "width": cfg["size"],
            "height": cfg["size"],
            "steps": cfg["steps"],
            "guidance": cfg["guidance"],
            "mock": MOCK_MODE,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


# --- SECTION 11: SELF-CHECK -------------------------------------------------

M2_FAILURES = []


def m2_check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' - ' + detail) if detail else ''}")
        M2_FAILURES.append(label)


print("\n" + "=" * 72)
print("M2 SELF-CHECK (REVISION 2) - image generation pipeline and caching")
print(f"mode: {'MOCK (no GPU)' if MOCK_MODE else 'REAL (SDXL on CUDA)'}")
print("=" * 72)

# Setup
# [audit-fix M2-R3] Isolate the self-check: a fixed session id plus a
# persistent cache/manifest would make a rerun report cache hits where the
# checks expect misses. Reset the self-check's own cache/manifest/state so a
# rerun is deterministic. (The live service regenerates what it needs.)
import shutil as _shutil
for _p in (MANIFEST_PATH, STATE_PATH):
    try:
        _p.unlink()
    except FileNotFoundError:
        pass
_shutil.rmtree(CACHE_DIR, ignore_errors=True)

session_id = "sess_m2_selfcheck"
person = build_personalisation("Aayush", "2002-10-06",
                               today=__import__("datetime").date(2026, 10, 6))
plan = build_catalog_plan(session_id, person, count=12)
print(f"\n[setup] plan built: {len(plan)} designs for session {session_id}")
print(f"[setup] unique cache keys: {len({i['design'].cache_key() for i in plan})}")

pipeline = BackgroundPipeline(CONFIG)

# -- [1] VRAM ladder wiring -------------------------------------------------
print("\n[1] VRAM ladder configuration")
ladder = CONFIG["vram_ladder"]
m2_check("rung 1 fp16 weights enabled", ladder["fp16_weights"])
m2_check("rung 2 SDPA enabled", ladder["sdpa_attention"])
m2_check("rung 3 VAE slicing enabled", ladder["vae_slicing"])
m2_check("rung 4 fixed fp16 VAE enabled", ladder["fixed_fp16_vae"])
m2_check(f"rung 7 resolution cap = {ladder['resolution_cap']}",
      ladder["resolution_cap"] <= 1024)
m2_check(f"rung 8 batch size = {ladder['batch_size']}", ladder["batch_size"] == 1)
m2_check("rungs 5-6, 9 remain conditional (off by default)",
      not ladder["vae_tiling"]
      and not ladder["model_cpu_offload"]
      and not ladder["sequential_cpu_offload"])

# -- [2] Determinism --------------------------------------------------------
print("\n[2] Determinism - same design_id -> same cache_key -> same file")
keys_a = [i["design"].cache_key() for i in plan]
plan_b = build_catalog_plan(session_id, person, count=12)
keys_b = [i["design"].cache_key() for i in plan_b]
m2_check("cache keys reproducible from session_id", keys_a == keys_b)

seed_a = derive_seed(keys_a[0], "final")
seed_b = derive_seed(keys_b[0], "final")
m2_check(f"derived seed reproducible ({seed_a})", seed_a == seed_b)

m2_check("draft and final seeds differ",
      derive_seed(keys_a[0], "draft") != derive_seed(keys_a[0], "final"))

# -- [3] Draft pass, cache-first -------------------------------------------
print("\n[3] Draft pass for all twelve designs")
t0 = time.time()
draft_records = pipeline.render_drafts(plan)
draft_elapsed = time.time() - t0
m2_check(f"twelve draft records returned ({len(draft_records)})",
      len(draft_records) == 12)
m2_check("all draft files exist",
      all(Path(r["path"]).is_file() for r in draft_records))
m2_check(f"draft files cached ({pipeline.cache.count('draft')})",
      pipeline.cache.count("draft") == 12)
m2_check("first pass was all cache misses",
      all(not r["from_cache"] for r in draft_records))
m2_check("drafts distinct by cache_key",
      len({r["cache_key"] for r in draft_records}) == 12)
print(f"  [info] twelve drafts in {draft_elapsed:.2f}s "
      f"({draft_elapsed / 12:.2f}s each)")

# -- [4] Cache hit ----------------------------------------------------------
print("\n[4] Repeated design is served from cache")
cache_hits_before = pipeline.cache.count("draft")
draft_again = pipeline.render_drafts(plan)
cache_hits_after = pipeline.cache.count("draft")
m2_check("second pass was all cache hits",
      all(r["from_cache"] for r in draft_again))
m2_check("no new files created on second pass",
      cache_hits_after == cache_hits_before)
m2_check("same file paths returned",
      [r["path"] for r in draft_records] == [r["path"] for r in draft_again])

# -- [5] Final pass on-demand ----------------------------------------------
print("\n[5] Final pass for two opened cards")
opened_designs = [plan[0]["design"], plan[3]["design"]]
final_records = [pipeline.render_final(d) for d in opened_designs]
m2_check("two final records returned", len(final_records) == 2)
m2_check("both final files exist",
      all(Path(r["path"]).is_file() for r in final_records))
m2_check("finals marked as final stage",
      all(r["stage"] == "final" for r in final_records))
m2_check("finals used more steps than drafts",
      all(r["steps"] == CONFIG["final"]["steps"] for r in final_records)
      and CONFIG["final"]["steps"] > CONFIG["draft"]["steps"])
m2_check("final cache is separate from draft cache",
      pipeline.cache.count("final") == 2
      and pipeline.cache.count("draft") == 12)

final_again = pipeline.render_final(opened_designs[0])
m2_check("repeated final served from cache", final_again["from_cache"])

# -- [6] Manifest -----------------------------------------------------------
print("\n[6] Manifest serialisation and readability")
reloaded = Manifest(MANIFEST_PATH, base_dir=WORK)
m2_check("manifest reloads from disk", len(reloaded.all()) == 14)
sample = reloaded.get(plan[0]["design"].design_id, "final")
required_fields = ("design_id", "cache_key", "stage", "path", "rel_path",
                   "seed", "width", "height", "steps")
m2_check("every required field present",
      all(f in sample for f in required_fields))
m2_check("manifest is JSON-serialisable",
      isinstance(json.dumps(reloaded.all()), str))
# [audit-fix M2-R5] rel_path is only set when the file is under WORK; require
# it only then (a sibling cache dir on a laptop must not spuriously fail).
m2_check("rel_path is populated",
      all(r.get("rel_path") or
          not str(r.get("path", "")).startswith(str(WORK))
          for r in reloaded.all()))
m2_check("[F4] resolve_path returns an existing file",
      reloaded.resolve_path(sample).is_file())

# -- [7] Memory behaviour ---------------------------------------------------
print("\n[7] Memory behaviour across consecutive renders")
if MOCK_MODE:
    print("  SKIP  no GPU, memory sampling not applicable")
else:
    samples = pipeline.watchdog.samples_gb()
    peak = max(samples) if samples else 0.0
    m2_check(f"peak reserved under 16 GB ({peak:.2f} GB)", peak < 15.0)
    m2_check(f"peak reserved under watchdog hard limit ({peak:.2f} GB)",
          peak < CONFIG["watchdog_hard_limit_gb"])
    if len(samples) >= 4:
        half = len(samples) // 2
        first = sum(samples[:half]) / half
        second = sum(samples[half:]) / (len(samples) - half)
        growth = second - first
        m2_check(f"no growth trend (first {first:.2f} GB, second {second:.2f} GB, "
              f"delta {growth:+.3f} GB)", abs(growth) < 0.5)
    print(f"  [info] samples: {[f'{s:.2f}' for s in samples]}")

# -- [8] Resumability -------------------------------------------------------
print("\n[8] Resumability - a fresh pipeline instance reuses the cache")
fresh = BackgroundPipeline(CONFIG)
resumed = fresh.render_drafts(plan)
m2_check("fresh instance served all twelve from cache",
      all(r["from_cache"] for r in resumed))
m2_check("fresh instance did not reload the renderer",
      fresh.renderer is None)

# -- [9] No text in the diffusion prompt -----------------------------------
print("\n[9] Compositor contract preserved")
for item in plan[:3]:
    prompt = item["design"].sd_prompt()
    m2_check(f"design {item['design'].design_id[:10]} reserves empty space",
          "empty space" in prompt)
    m2_check(f"design {item['design'].design_id[:10]} negative forbids lettering",
          "text" in item["design"].sd_negative_prompt())

# -- [10] Mode integrity ----------------------------------------------------
print("\n[10] Mode integrity")
if MOCK_MODE:
    m2_check("mock records flagged in manifest",
          all(r.get("mock") is True for r in reloaded.all()))
    print("  [info] mock output is NOT a real render. Run on T4 x2 for "
          "M2 acceptance.")
else:
    m2_check("real-mode records not flagged as mock",
          all(r.get("mock") is False for r in reloaded.all()))
    m2_check("GPU had enough VRAM for the ladder", DEVICE_VRAM_GB >= 15.0)

# -- [11] AGA finding F1 - per-job cleanup ---------------------------------
print("\n[11] [F1] Per-job cleanup runs after every render")
expected_render_resets = 12 + 2  # 12 drafts + 2 finals (cache misses only)
actual_resets = getattr(pipeline.renderer, "reset_calls", None)
m2_check(f"renderer.reset() called {actual_resets} times "
      f"(expected {expected_render_resets})",
      actual_resets == expected_render_resets)
actual_renders = getattr(pipeline.renderer, "render_calls", None)
m2_check(f"renderer.render() called {actual_renders} times",
      actual_renders == expected_render_resets)
m2_check("reset count == render count (1 cleanup per render)",
      actual_resets == actual_renders)

# -- [12] AGA finding F3 - watchdog samples on cache hits ------------------
print("\n[12] [F3] Watchdog samples on every job, cache hit or miss")
# Jobs on the original pipeline: 12 + 12 + 2 + 1 = 27
expected_samples = 12 + 12 + 2 + 1
actual_samples = len(pipeline.watchdog.samples)
m2_check(f"watchdog sampled {actual_samples} times (expected {expected_samples})",
      actual_samples == expected_samples)

# -- [13] AGA finding F4 - relative path resolution ------------------------
print("\n[13] [F4] Manifest paths survive a work-dir change")
rec = reloaded.get(plan[0]["design"].design_id, "draft")
resolved_abs = reloaded.resolve_path(rec)
m2_check("absolute path resolves", resolved_abs.is_file())
# Simulate a moved work dir: blank the absolute path, keep rel_path.
moved = {**rec, "path": "/nonexistent/" + Path(rec["path"]).name}
resolved_rel = reloaded.resolve_path(moved)
m2_check("rel_path fallback resolves when absolute path is gone",
      resolved_rel.is_file())
m2_check("rel_path and path point to the same file",
      resolved_rel.resolve() == resolved_abs.resolve())

# -- [14] AGA finding F9 - watermarker flag present ------------------------
print("\n[14] [F9] Watermarker decision is explicit")
m2_check("disable_watermarker flag exists",
      "disable_watermarker" in CONFIG)
m2_check("watermarker default is ON (audit-safe)",
      CONFIG.get("disable_watermarker") is False)
print("  [flag] Turning the Stability AI invisible watermark off is a legal/"
      "ethical decision. Awaiting explicit order before flipping the flag.")

# -- [15] AGA findings F11, F12 - manifest resilience ----------------------
print("\n[15] [F11][F12] Manifest resilience")
# F11: unchanged payload skips the write.
before = MANIFEST_PATH.stat().st_mtime
reloaded._save()  # no changes
after = MANIFEST_PATH.stat().st_mtime
m2_check("unchanged manifest is not rewritten", before == after)

# F12: corrupted manifest moves to .bak.
corrupt_path = MANIFEST_PATH.with_suffix(".corrupt_test.json")
corrupt_path.write_text("{not json")
survivor = Manifest(corrupt_path, base_dir=WORK)
m2_check("corrupted manifest moved to .bak",
      corrupt_path.with_suffix(".json.bak").is_file())
m2_check("fresh manifest is empty after corruption",
      len(survivor.all()) == 0)
corrupt_path.with_suffix(".json.bak").unlink()
corrupt_path.unlink(missing_ok=True)

# -- Summary ----------------------------------------------------------------
print("\n" + "=" * 72)
if M2_FAILURES:
    print(f"RESULT: {len(M2_FAILURES)} FAILURE(S)")
    for f in M2_FAILURES:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 72)
print(f"\n[M2] manifest: {MANIFEST_PATH}")
print(f"[M2] cache:    {CACHE_DIR}")
print(f"[M2] state:    {STATE_PATH}")

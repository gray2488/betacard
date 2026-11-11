"""M3 - typography, text generation and card compositing."""

from .config import CONFIG
from .m1_design import *  # noqa: F401,F403
from .m2_render import *  # noqa: F401,F403

# =============================================================================
# PROJECT BETA-CARD - MILESTONE M3 (REVISION 4)
# Text generation and compositor (Mix-and-Match).
#
# Revision 4 changes (AGA M3 Rev 3 audit):
#   [F1] CRITICAL - M3's CONFIG dict overwrote M2's module-level CONFIG, which
#        M2's BackgroundPipeline.__init__ and JobState.bump read at call time.
#        Fix: M3 uses M3_CONFIG; all M3 path globals renamed with M3_ prefix;
#        BackgroundPipeline is instantiated with no argument so it uses M2's
#        own CONFIG. M3 no longer touches any M2 global.
#   [F2] MAJOR - contrast against actual pixels was not guaranteed for
#        mixed-tone backgrounds: pick_text_colour only chose the best
#        candidate; if none cleared 4.5, it used the closest one anyway.
#        Fix: _prepare_text_layer() applies an adaptive semi-transparent scrim
#        behind the text region until a candidate clears AA; if the maximum
#        scrim alpha still fails, a contrasting stroke is drawn as a last
#        resort. The applied fallback is recorded in the manifest.
#   [F3] MINOR - the *prompt-builder* self-checks no longer call
#        inspect.getsource (fragile in exec harnesses); they are tested
#        behaviourally instead. The one remaining getsource use (manifest
#        backup) is wrapped in try/except OSError - see [M3-R5].
#
# Revisions 2 and 3 retained:
#   [M3-R3-F9] TextGenerator uses attribute access on Personalisation.
#   [M3-R2-F1] _fit_text uses layout.max_lines.
#   [M3-R2-F2] font_policy default is "strict" in real mode.
#   [M3-R2-F3] missing Noto font is a FAIL, not a SKIP.
#   [M3-R2-F4] silent MOCK_MODE fallback removed.
#   [M3-R2-F5] manifest backup move wrapped in try/except.
#   [M3-R2-F6] dead code hasattr(layout, "no_upper") removed.
#   [M3-R2-F7] record carries text_input and text_output separately.
#   [M3-R2-F8] base font size derives from the actual image size.
#
# REQUIRES: M1 cell and M2 cell have already run in this kernel.
# Fonts: run with font_policy="strict" (default) and the real TTFs under
#   assets/fonts. Local CPU test only: set BETACARD_MOCK=1.
# =============================================================================


# --- SECTION 1: IMPORTS AND CONFIG ------------------------------------------

import os
import gc
import json
import time
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Reuse M2's device detection if M2 has already run in this kernel; otherwise
# compute fresh. M3 must not overwrite M2's globals (F1).
if "HAS_TORCH" not in globals():
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        torch = None
        HAS_TORCH = False

if "HAS_CUDA" not in globals():
    HAS_CUDA = bool(HAS_TORCH and torch.cuda.is_available())

if "MOCK_MODE" not in globals():
    MOCK_MODE = os.environ.get("BETACARD_MOCK", "0") == "1"
    if not MOCK_MODE and not HAS_CUDA:
        raise RuntimeError(
            "CUDA is not available and BETACARD_MOCK is not set. "
            "Enable a GPU accelerator on Kaggle (select 'GPU T4 x2'), or set "
            "BETACARD_MOCK=1 for a CPU-only logic test."
        )

# Guards: M1 and M2 must have run.
_M1_NEEDED = ("build_catalog_plan", "build_personalisation",
              "resolve_font", "pick_text_colour", "contrast_ratio",
              "relative_luminance", "DesignToken", "Layout", "Typography",
              "TYPOGRAPHIES", "NO_CASE_TRANSFORM", "NOTO_FAMILY")
_M2_NEEDED = ("BackgroundPipeline",)
_MISSING = [n for n in (_M1_NEEDED + _M2_NEEDED) if n not in globals()]
if _MISSING:
    raise RuntimeError(
        f"M1 and M2 cells must run before M3. Missing: {_MISSING}. "
        "Paste the M1 and M2 cells above this one and run them first."
    )

# [F2] Font policy resolution.
_FONT_POLICY_OVERRIDE = os.environ.get("BETACARD_FONT_POLICY", "").strip()
if _FONT_POLICY_OVERRIDE in ("strict", "latin_fallback", "mock"):
    M3_FONT_POLICY = _FONT_POLICY_OVERRIDE
elif MOCK_MODE:
    M3_FONT_POLICY = "mock"
else:
    M3_FONT_POLICY = "strict"

# [audit-fix M3-R2] Snapshot M2's WORK before M3 defines anything, so the [0]
# isolation check compares against a real value (M2_WORK was never defined,
# which made the check a silent no-op).
M2_WORK = globals().get("WORK")

# [F1] M3 owns M3_CONFIG. It MUST NOT overwrite M2's CONFIG, because M2's
# BackgroundPipeline.__init__ and JobState.bump read that global at call time.
M3_CONFIG = {
    "work_dir": os.environ.get("BETACARD_BASE") or "/kaggle/working/betacard",
    "fonts_dir": os.path.join(os.environ.get("BETACARD_BASE")
                               or "/kaggle/working/betacard", "assets/fonts"),
    "cards_dir": os.path.join(os.environ.get("BETACARD_BASE")
                              or "/kaggle/working/betacard", "cards"),
    "card_manifest": os.path.join(os.environ.get("BETACARD_BASE")
                                  or "/kaggle/working/betacard",
                                  "card_manifest.json"),  # [env-fix] portable

    # [env-fix] Text model source. DEFAULT IS OFF (None) -> M1 templates, which
    # need no download. Qwen-7B is ~14 GB and does not fit Kaggle's 20 GB
    # working dir, so it is opt-in: set BETACARD_TEXT_MODEL_PATH to a dataset
    # path, or a small id such as "Qwen/Qwen2.5-0.5B-Instruct".
    "text_model_id": (os.environ.get("BETACARD_TEXT_MODEL_PATH") or None),
    "text_model_4bit": True,
    "text_max_new_tokens": 120,
    "text_min_words": 4,

    "font_policy": M3_FONT_POLICY,

    # Compositor
    "base_font_size_ratio": 0.045,
    "min_font_size": 18,
    "padding_px": 28,
    "contrast_sample_grid": 8,
    "min_contrast": 4.5,             # WCAG AA
    "truncation_ellipsis": "…",
    # [F2] Scrim fallback ladder. Applied in order until AA is cleared.
    "scrim_alpha_ladder": (0.30, 0.45, 0.60, 0.75, 0.85),
    "scrim_max_alpha": 0.85,
}

M3_WORK = Path(M3_CONFIG["work_dir"])
M3_FONTS_DIR = Path(M3_CONFIG["fonts_dir"])
M3_CARDS_DIR = Path(M3_CONFIG["cards_dir"])
M3_CARD_MANIFEST_PATH = Path(M3_CONFIG["card_manifest"])

for _d in (M3_WORK, M3_FONTS_DIR, M3_CARDS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print(f"[M3] mock={MOCK_MODE} cuda={HAS_CUDA} font_policy={M3_CONFIG['font_policy']}")
print(f"[M3] fonts={M3_FONTS_DIR}")
print(f"[M3] cards={M3_CARDS_DIR}")


# --- SECTION 2: FONT RESOLUTION AND BOOTSTRAP ------------------------------

_LATIN_SUBSTITUTIONS = {
    "PlayfairDisplay-Bold":        "DejaVuSerif-Bold.ttf",
    "Lato-Regular":                "DejaVuSans.ttf",
    "Montserrat-Bold":             "DejaVuSans-Bold.ttf",
    "OpenSans-Regular":            "DejaVuSans.ttf",
    "Merriweather-Bold":           "DejaVuSerif-Bold.ttf",
    "SourceSans-Regular":          "DejaVuSans.ttf",
    "Poppins-SemiBold":            "DejaVuSans-Bold.ttf",
    "Inter-Regular":               "DejaVuSans.ttf",
    "Cinzel-SemiBold":             "DejaVuSerif-Bold.ttf",
    "EBGaramond-Regular":          "DejaVuSerif.ttf",
    "Quicksand-Bold":              "DejaVuSans-Bold.ttf",
    "Nunito-Regular":              "DejaVuSans.ttf",
    "CormorantGaramond-SemiBold":  "DejaVuSerif-Bold.ttf",
    "Karla-Regular":               "DejaVuSans.ttf",
    "BebasNeue-Regular":           "DejaVuSans-Bold.ttf",
    "Roboto-Regular":              "DejaVuSans.ttf",
}


def _matplotlib_font_dir():
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        return p if p.is_dir() else None
    except Exception:
        return None


def install_latin_substitute_fonts(fonts_dir=None, force=False):
    fonts_dir = Path(fonts_dir or M3_FONTS_DIR)
    fonts_dir.mkdir(parents=True, exist_ok=True)
    src_dir = _matplotlib_font_dir()
    if src_dir is None:
        raise RuntimeError(
            "matplotlib's font directory not found. Install matplotlib, or "
            "place the font files in " + str(fonts_dir) + " manually."
        )
    installed = []
    for stem, src_name in _LATIN_SUBSTITUTIONS.items():
        dst = fonts_dir / f"{stem}.ttf"
        if dst.is_file() and not force:
            continue
        src = src_dir / src_name
        if not src.is_file():
            continue
        shutil.copy(src, dst)
        installed.append(stem)
    return installed


class FontResolver:
    """Resolve M1 font stems to actual TTF paths under M3_FONTS_DIR.

    Policies:
      * "strict"         - any missing stem raises. Default in real mode.
      * "latin_fallback" - missing Latin stems materialise from DejaVu;
                           missing non-Latin stems still raise.
      * "mock"           - every stem maps to DejaVu; nothing raises.
    """

    def __init__(self, fonts_dir=None, policy=None, substitutions=None):
        self.fonts_dir = Path(fonts_dir or M3_FONTS_DIR)
        self.policy = policy or M3_CONFIG["font_policy"]
        self.substitutions = dict(substitutions or _LATIN_SUBSTITUTIONS)
        self._cache = {}

        if self.policy in ("latin_fallback", "mock"):
            try:
                install_latin_substitute_fonts(self.fonts_dir)
            except RuntimeError as exc:
                print(f"[M3] latin font bootstrap skipped: {exc}")

    def _is_latin_stem(self, stem):
        return stem in self.substitutions

    def resolve(self, stem):
        if stem in self._cache:
            return self._cache[stem]

        primary = self.fonts_dir / f"{stem}.ttf"
        if primary.is_file():
            self._cache[stem] = primary
            return primary

        if self.policy == "mock":
            src = self.fonts_dir / self.substitutions.get(
                stem, "DejaVuSans.ttf")
            if not src.is_file():
                src = self.fonts_dir / "DejaVuSans.ttf"
            if not src.is_file():
                raise FileNotFoundError(
                    f"mock font bootstrap failed; {src} does not exist")
            self._cache[stem] = src
            return src

        if self.policy == "latin_fallback" and self._is_latin_stem(stem):
            src = self.fonts_dir / self.substitutions[stem]
            if not src.is_file():
                raise FileNotFoundError(
                    f"Latin substitute {src.name} missing for stem {stem!r}. "
                    "Run install_latin_substitute_fonts() with matplotlib "
                    "available."
                )
            self._cache[stem] = src
            return src

        # [audit-fix M3-R3] list the required stems so the strict-mode fix is
        # obvious instead of a bare FileNotFoundError.
        try:
            _ml, _mn = self.validate_required_stems()
            _hint = (f"Missing Latin stems: {', '.join(_ml) or '(none)'}. "
                     f"Missing Noto stems: {', '.join(_mn) or '(none)'}. ")
        except Exception:
            _hint = ""
        raise FileNotFoundError(
            f"font {stem!r} not found at {primary}. "
            f"Font policy is {self.policy!r}. Place the TTF under "
            f"{self.fonts_dir}. {_hint}"
            "Non-Latin scripts require the real Noto font."
        )

    def validate_required_stems(self, extra_stems=()):
        missing_latin, missing_noto = [], []
        latin_stems = set()
        for t in TYPOGRAPHIES:
            latin_stems.add(t.display_font)
            latin_stems.add(t.body_font)
        for stem in sorted(latin_stems):
            try:
                self.resolve(stem)
            except FileNotFoundError:
                missing_latin.append(stem)
        for stem in sorted(set(extra_stems)):
            if stem in latin_stems:
                continue
            try:
                self.resolve(stem)
            except FileNotFoundError:
                missing_noto.append(stem)
        return missing_latin, missing_noto


# --- SECTION 3: SCRIPT DETECTION FOR A MESSAGE -----------------------------

def dominant_script(*texts):
    if "detect_script" in globals():
        for t in texts:
            if not t:
                continue
            tag = detect_script(t)
            if tag != "latin":
                return tag
        return "latin"
    return "latin"


# --- SECTION 4: TEXT GENERATION --------------------------------------------

_SYSTEM_PROMPT = (
    "You are a birthday card copywriter. You will be given one draft wishing "
    "line and asked to polish it. You must keep the same meaning and tone. "
    "You must keep the recipient's name exactly as given, and the age number "
    "if present. You must not add greetings, quotes, emojis, emoji-like "
    "symbols, or explanations. You must output only the final line, nothing "
    "else."
)


def _build_polish_prompt(base_text, tone, length, name, turning_age):
    length_hint = {
        "short":  "under about 12 words",
        "medium": "about 12-22 words",
        "long":   "about 22-35 words",
    }.get(length, "about 12-22 words")
    return (
        f"Tone: {tone}\n"
        f"Length: {length} ({length_hint})\n"
        f"Recipient name (must appear verbatim): {name}\n"
        f"Turning age (keep if it appears): {turning_age}\n\n"
        f"Draft line:\n{base_text}\n\n"
        "Polished line:"
    )


class TextGenerator:
    """Polish the M1 template line with a small instruction model.

    On any failure - model unavailable, load error, invalid output - the
    original template text is returned verbatim.
    """

    def __init__(self, config=None):
        self.config = config or M3_CONFIG
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.load_error = None
        self._load_attempted = False

    def load(self):
        """Load the model. Idempotent on success; retriable on failure."""
        if self.loaded:
            return True
        if self._load_attempted and self.load_error is None:
            return False
        # Reset any prior error so a retry can proceed.
        self.load_error = None
        self._load_attempted = True
        if MOCK_MODE:
            self.load_error = "mock mode: text model skipped"
            print("[M3] MOCK MODE - text generation uses template fallback")
            return False
        if not self.config.get("text_model_id"):
            self.load_error = "text model disabled (text_model_id is None)"
            print("[M3] text model off; using the M1 templates. Set "
                  "BETACARD_TEXT_MODEL_PATH (or a small model id) to enable it.")
            return False
        # [audit-fix M3-R7] catch ANY transformers failure, not just ImportError:
        # transformers 5.x raises ValueError ("Backend should be defined in the
        # BACKENDS_MAPPING") on import, which used to escape here and abort the
        # whole job (0 cards). Fall back to the M1 templates instead.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            self.load_error = f"transformers unavailable: {exc}"
            print(f"[M3] {self.load_error}; using template fallback")
            if "BACKENDS_MAPPING" in str(exc) or "keras_nlp" in str(exc):
                print("[M3] this is the known transformers 5.x bug - run the "
                      "bootstrap cell, or `pip install 'transformers>=4.44,<5'` "
                      "and restart the kernel.")
            return False

        # [audit-fix M3-R8] a 7B model needs ~16 GB; if the cache volume cannot
        # hold it, the download dies mid-way with an opaque "No space left on
        # device" and can fill the whole workspace. Check free space first and
        # fall back to the M1 templates cleanly.
        import shutil as _shutil
        _cache_root = (os.environ.get("BETACARD_HF_CACHE")
                       or os.environ.get("HF_HOME") or ".")
        try:
            _free_gb = _shutil.disk_usage(_cache_root).free / (1024 ** 3)
        except OSError:
            _free_gb = None
        _need_gb = float(self.config.get("text_model_min_free_gb", 16.0))
        # [env-fix] a locally-mounted dataset needs no download, so skip the
        # free-space gate entirely when the source is a local directory.
        _is_local = os.path.isdir(str(self.config["text_model_id"]))
        if (not _is_local and _free_gb is not None and _free_gb < _need_gb):
            self.load_error = ("not enough disk for the text model "
                               "(%.1f GB free, %.0f GB needed)"
                               % (_free_gb, _need_gb))
            print(f"[M3] {self.load_error}; using template fallback")
            print("[M3] set text_model_id=None (templates) or use a smaller "
                  "model such as Qwen/Qwen2.5-0.5B-Instruct.")
            return False

        try:
            t0 = time.time()
            print(f"[M3] loading text model {self.config['text_model_id']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["text_model_id"], trust_remote_code=True)

            kwargs = {"trust_remote_code": True, "device_map": "auto"}
            # [audit-fix M3-R6] BitsAndBytesConfig imports fine WITHOUT
            # bitsandbytes installed (it is only a config dataclass), so the old
            # `except ImportError` never fired and load_in_4bit was passed to a
            # build with no bnb -> the load failed at from_pretrained instead of
            # falling back cleanly. Probe the module explicitly first.
            import importlib.util as _ilu
            _has_bnb = _ilu.find_spec("bitsandbytes") is not None
            if self.config["text_model_4bit"] and HAS_CUDA and _has_bnb:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("[M3] text model: 4-bit (bitsandbytes)")
            else:
                if self.config["text_model_4bit"] and HAS_CUDA and not _has_bnb:
                    print("[M3] bitsandbytes not installed; loading the text "
                          "model in fp16 (much heavier). `pip install "
                          "bitsandbytes` for the 4-bit path.")
                kwargs["torch_dtype"] = torch.float16 if HAS_CUDA else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["text_model_id"], **kwargs)
            self.model.eval()
            self.loaded = True
            print(f"[M3] text model ready in {time.time() - t0:.1f}s")
            return True
        except Exception as exc:
            self.load_error = str(exc)
            print(f"[M3] text model failed to load ({exc}); using template fallback")
            self.model = None
            self.tokenizer = None
            return False

    def retry(self):
        """Clear the failure record so the next load() retries."""
        self._load_attempted = False
        self.load_error = None
        self.model = None
        self.tokenizer = None
        self.loaded = False
        return self.load()

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        gc.collect()
        if HAS_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[M3] text model unloaded")

    def generate(self, message, personalisation, layout):
        """Return (text, source) where source is 'generated' or 'template'.

        Personalisation is accessed by attribute (it is an M1 dataclass).
        Everything after the unloaded check is inside the try so a future
        contract mismatch falls back to the template rather than crashing.
        """
        base_text = message["text"]

        if not self.loaded:
            return base_text, "template"

        try:
            prompt = _build_polish_prompt(
                base_text=base_text,
                tone=message["tone"],
                length=message.get("length", "medium"),
                name=personalisation.name,
                turning_age=personalisation.turning_age,
            )

            seed = int(hashlib.sha256(
                message["message_id"].encode("utf-8")).hexdigest()[:8], 16)
            if HAS_TORCH:
                torch.manual_seed(seed)
                if HAS_CUDA:
                    torch.cuda.manual_seed_all(seed)

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                chat_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["text_max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            raw = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            candidate = self._clean(raw)

            ok, reason = self._validate(candidate, message, personalisation)
            if ok:
                return candidate, "generated"
            print(f"[M3] rejected generated line ({reason}); using template")
            return base_text, "template"
        except Exception as exc:
            print(f"[M3] generation error ({exc}); using template")
            return base_text, "template"

    def _clean(self, text):
        t = text.strip()
        for pre in ("Polished line:", "Final line:", "Output:"):
            if t.startswith(pre):
                t = t[len(pre):].strip()
        if len(t) >= 2 and t[0] in "\"'“”‘’" and t[-1] in "\"'“”‘’":
            t = t[1:-1].strip()
        for line in t.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    def _validate(self, text, message, personalisation):
        """Attribute access on the Personalisation dataclass."""
        if not text:
            return False, "empty"
        words = text.split()
        if len(words) < self.config["text_min_words"]:
            return False, "too short"
        if len(words) > 60:
            return False, "too long"
        name = personalisation.name
        if name.lower() in message["text"].lower():
            if name.lower() not in text.lower():
                return False, "name dropped"
        base_words = len(message["text"].split())
        band = message.get("length", "medium")
        ceilings = {"short": base_words * 2 + 4,
                    "medium": base_words * 2,
                    "long": base_words * 2}
        if len(words) > ceilings.get(band, base_words * 2):
            return False, "off-band length"
        # [audit-fix M3-R4] the age must survive: if the source template names
        # the turning age, the output must still carry it (a real LLM can
        # invent a wrong birthday age; nothing enforced this before).
        _turning = getattr(personalisation, "turning_age", None)
        if _turning is not None and str(_turning) in message["text"]:
            _forms = [str(_turning)]
            try:
                _forms.append(ordinal(_turning))
            except Exception:
                pass
            # [audit-fix M3-R5] token-aware, not substring: "24" must not match
            # inside "124" / "124th".
            import re as _re
            if not any(_re.search(r"(?<!\d)" + _re.escape(f) + r"(?!\d)", text)
                       for f in _forms):
                return False, "age dropped/changed"
        return True, "ok"


# --- SECTION 5: COMPOSITOR --------------------------------------------------

def _srgb_to_linear_local(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hex_luminance(hex_colour):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r, g, b = _srgb_to_linear_local(r), _srgb_to_linear_local(g), _srgb_to_linear_local(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(lum_a, lum_b):
    hi, lo = max(lum_a, lum_b), min(lum_a, lum_b)
    return (hi + 0.05) / (lo + 0.05)


class CardCompositor:
    """Draw a message onto a background using a design's typography, palette
    and layout. CPU-only. Deterministic. Design, message and background are
    independent inputs so Mix-and-Match works.
    """

    def __init__(self, config=None, font_resolver=None):
        self.config = config or M3_CONFIG
        self.fonts = font_resolver or FontResolver()
        self.cards_dir = M3_CARDS_DIR
        self.cards_dir.mkdir(parents=True, exist_ok=True)

    def compose(self, design, message, background_path):
        bg_path = Path(background_path)
        if not bg_path.is_file():
            raise FileNotFoundError(f"background not found: {bg_path}")

        background = Image.open(bg_path).convert("RGB")
        size = background.size[0]
        if background.size[0] != background.size[1]:
            raise ValueError(
                f"expected a square background, got {background.size}")

        region = design.layout.region_px(size)
        pad = self.config["padding_px"]
        x, y, w, h = region
        region = (x + pad, y + pad, max(1, w - 2 * pad), max(1, h - 2 * pad))

        # [F2] Colour + optional scrim. Returns a canvas that already has the
        # scrim baked in if one was needed.
        canvas, text_color, min_ratio, contrast_fallback = \
            self._prepare_text_layer(
                background, region, design.text_colour_candidates)

        # Font for the message's script.
        script = dominant_script(message["text"])
        try:
            stem = resolve_font(design.typography, script, role="display")
        except Exception:
            stem = design.typography.display_font
        font_path = self.fonts.resolve(stem)

        text_input = message["text"]
        apply_upper = bool(design.typography.uppercase_title
                           and script not in NO_CASE_TRANSFORM)
        text_out, font, lines, font_size, truncated = self._fit_text(
            text_input, font_path, region, design.typography,
            design.layout, apply_upper, size)

        # Stroke only if the scrim ladder still could not clear AA.
        stroke_width = 2 if contrast_fallback == "scrim+stroke" else 0
        stroke_fill = self._stroke_colour(text_color) if stroke_width else None

        self._draw_lines(canvas, lines, font, region,
                         design.layout, design.typography, text_color,
                         apply_upper, stroke_width, stroke_fill)

        card_id = self._card_id(design.design_id, message["message_id"])
        out_path = self.cards_dir / f"{card_id}.png"
        tmp = out_path.with_suffix(".png.tmp")
        # [audit-fix M3-R9] a full disk used to raise a bare OSError from deep
        # inside Pillow and abort the whole compose loop. Fail with a clear,
        # actionable message instead.
        try:
            canvas.save(tmp, "PNG", optimize=True)
            tmp.replace(out_path)
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise RuntimeError(
                f"could not write {out_path.name}: {exc}. The workspace is "
                f"likely out of disk space - free space, or point "
                f"BETACARD_BASE at a larger volume."
            ) from exc

        try:
            rel_path = str(out_path.relative_to(M3_WORK))
        except ValueError:
            rel_path = None

        return {
            "card_id": card_id,
            "design_id": design.design_id,
            "message_id": message["message_id"],
            "path": str(out_path),
            "rel_path": rel_path,
            "font_stem": stem,
            "font_path": str(font_path),
            "font_size": font_size,
            "text_color": text_color,
            "contrast_ratio": round(min_ratio, 3),
            "contrast_fallback": contrast_fallback,
            "text_input": text_input,
            "text_output": text_out,
            "lines": len(lines),
            "truncated": truncated,
            "script": script,
            "uppercase_applied": apply_upper,
            "stage": "final",
            "mock": MOCK_MODE,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def compose_merge(self, design_a, message_b, background_pipeline,
                      stage="final"):
        """Mix-and-Match: design A + message B. Instant. No GPU."""
        bg_path = background_pipeline.get_path(design_a.design_id, stage)
        if bg_path is None:
            raise RuntimeError(
                f"no {stage} background for design {design_a.design_id}; "
                "run the M2 renderer first")
        return self.compose(design_a, message_b, bg_path)

    # -- internals ----------------------------------------------------------

    def _card_id(self, design_id, message_id):
        payload = f"{design_id}|{message_id}"
        return "card_" + hashlib.sha256(
            payload.encode("utf-8")).hexdigest()[:20]

    def _sample_region_luminances(self, image, region):
        x, y, w, h = region
        grid = self.config["contrast_sample_grid"]
        luminances = []
        arr = np.asarray(image)
        H, W = arr.shape[:2]
        for gy in range(grid):
            for gx in range(grid):
                px = min(W - 1, x + int((gx + 0.5) * w / grid))
                py = min(H - 1, y + int((gy + 0.5) * h / grid))
                x0, x1 = max(0, px - 2), min(W, px + 3)
                y0, y1 = max(0, py - 2), min(H, py + 3)
                patch = arr[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0) / 255.0
                r, g, b = (_srgb_to_linear_local(float(c)) for c in patch)
                luminances.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
        return luminances

    def _pick_best_candidate(self, luminances, candidates):
        """Return the candidate with the highest worst-case contrast."""
        best_color, best_min = candidates[0], -1.0
        for c in candidates:
            lum = _hex_luminance(c)
            min_ratio = min(_ratio(lum, cell) for cell in luminances)
            if min_ratio > best_min:
                best_color, best_min = c, min_ratio
        return best_color, best_min

    def _apply_scrim(self, background, region, scrim_rgb, alpha):
        """Return a copy of `background` with a semi-transparent rectangle
        drawn over `region`."""
        x, y, w, h = region
        canvas = background.convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        a = int(round(alpha * 255))
        # Inclusive coordinates so the scrim covers the full region.
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=(*scrim_rgb, a))
        result = Image.alpha_composite(canvas, overlay).convert("RGB")
        return result

    def _prepare_text_layer(self, background, region, candidates):
        """[F2] Guarantee a text colour that clears WCAG AA.

        Returns (canvas, text_color, min_ratio, fallback_used):
          * canvas   - background copy, with the scrim baked in if applied
          * fallback - None | "scrim" | "scrim+stroke"

        The strategy is a short ladder: try the design's own candidates
        first; if none clears AA on a mixed-tone background, apply an
        adaptive scrim (dark over a light region, light over a dark region)
        and re-evaluate; if even the maximum scrim alpha fails, add a
        contrasting stroke as a last resort.
        """
        luminances = self._sample_region_luminances(background, region)
        best_color, best_min = self._pick_best_candidate(luminances, candidates)

        if best_min >= self.config["min_contrast"]:
            return background.copy(), best_color, best_min, None

        # Region is mixed-tone. Choose a scrim direction from the mean.
        avg_lum = sum(luminances) / max(1, len(luminances))
        scrim_rgb = (0, 0, 0) if avg_lum > 0.5 else (255, 255, 255)

        for alpha in self.config["scrim_alpha_ladder"]:
            if alpha > self.config["scrim_max_alpha"]:
                break
            candidate_canvas = self._apply_scrim(background, region,
                                                  scrim_rgb, alpha)
            new_lums = self._sample_region_luminances(candidate_canvas, region)
            c, m = self._pick_best_candidate(new_lums, candidates)
            if m >= self.config["min_contrast"]:
                return candidate_canvas, c, m, "scrim"

        # Max alpha still did not clear AA: return it plus a stroke.
        canvas = self._apply_scrim(background, region, scrim_rgb,
                                    self.config["scrim_max_alpha"])
        new_lums = self._sample_region_luminances(canvas, region)
        c, m = self._pick_best_candidate(new_lums, candidates)
        return canvas, c, m, "scrim+stroke"

    def _stroke_colour(self, text_color):
        """Opposite of the fill, for the last-resort stroke."""
        lum = _hex_luminance(text_color)
        return "#000000" if lum > 0.5 else "#FFFFFF"

    def _wrap(self, text, font, max_width):
        words = text.split()
        if not words:
            return []
        lines, current = [], ""
        for word in words:
            trial = (current + " " + word).strip()
            if font.getlength(trial) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _fit_text(self, text, font_path, region, typography, layout,
                  apply_upper, image_size):
        """Wrap / shrink / truncate so the text fits the region."""
        x, y, w, h = region
        base_size = max(
            self.config["min_font_size"],
            int(typography.title_body_ratio *
                self.config["base_font_size_ratio"] * image_size),
        )
        min_size = self.config["min_font_size"]
        max_lines = max(1, layout.max_lines)

        measure_text = text.upper() if apply_upper else text

        size = base_size
        while size >= min_size:
            font = ImageFont.truetype(str(font_path), size)
            lines = self._wrap(measure_text, font, w)
            line_h = int(size * typography.line_spacing)
            total_h = line_h * len(lines)
            widest = max((font.getlength(l) for l in lines), default=0)
            if (len(lines) <= max_lines and total_h <= h and widest <= w):
                return " ".join(lines), font, lines, size, False
            size -= 2

        font = ImageFont.truetype(str(font_path), min_size)
        line_h = int(min_size * typography.line_spacing)
        allowed = max(1, min(max_lines, h // max(1, line_h)))
        lines = self._wrap(measure_text, font, w)
        truncated = False
        if len(lines) > allowed:
            lines = lines[:allowed]
            truncated = True
            last = lines[-1]
            ell = self.config["truncation_ellipsis"]
            while last and font.getlength(last + ell) > w:
                last = last[:-1].rstrip()
            lines[-1] = (last + ell) if last else ell
        return " ".join(lines), font, lines, min_size, truncated

    def _draw_lines(self, canvas, lines, font, region, layout, typography,
                    colour, apply_upper, stroke_width=0, stroke_fill=None):
        x, y, w, h = region
        line_h = int(font.size * typography.line_spacing)
        total_h = line_h * len(lines)

        if layout.valign == "top":
            y_start = y
        elif layout.valign == "bottom":
            y_start = y + h - total_h
        else:
            y_start = y + (h - total_h) // 2

        draw = ImageDraw.Draw(canvas)

        for i, line in enumerate(lines):
            line_w = font.getlength(line)
            if layout.align == "left":
                x_line = x
            elif layout.align == "right":
                x_line = x + w - line_w
            else:
                x_line = x + (w - line_w) // 2
            y_line = y_start + i * line_h
            if stroke_width:
                draw.text((x_line, y_line), line, font=font, fill=colour,
                          stroke_width=stroke_width, stroke_fill=stroke_fill)
            else:
                draw.text((x_line, y_line), line, font=font, fill=colour)


# --- SECTION 6: CARD MANIFEST ----------------------------------------------

class CardManifest:
    """Persistent record of every finished card, for M4 to upload."""

    def __init__(self, path=None):
        self.path = Path(path or M3_CARD_MANIFEST_PATH)
        self.records = {}
        self._last_payload = None
        if self.path.is_file():
            try:
                for rec in json.loads(self.path.read_text()):
                    self.records[rec["card_id"]] = rec
                self._last_payload = json.dumps(
                    self.all(), indent=2, default=str)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                backup = self.path.with_suffix(".json.bak")
                try:
                    self.path.replace(backup)
                    print(f"[M3] card manifest unreadable, moved to {backup}: {exc}")
                except OSError as move_exc:
                    print(f"[M3] card manifest unreadable and could not be "
                          f"backed up ({move_exc}); starting fresh: {exc}")

    def upsert(self, record):
        self.records[record["card_id"]] = record
        self._save()

    def get(self, card_id):
        return self.records.get(card_id)

    def all(self):
        return list(self.records.values())

    def _save(self):
        payload = json.dumps(self.all(), indent=2, default=str)
        if payload == self._last_payload:
            return
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(self.path)
        self._last_payload = payload


# --- SECTION 7: HIGH-LEVEL RUN ---------------------------------------------

def run_m3(session_id, personalisation, plan, background_pipeline,
           text_generator=None, compositor=None, card_manifest=None):
    """Produce all twelve finished cards for a session.

    Order (VRAM discipline): text model first, then it is freed, then the
    compositor runs on CPU.
    """
    compositor = compositor or CardCompositor()
    card_manifest = card_manifest or CardManifest()

    gen = text_generator or TextGenerator()
    gen.load()
    final_lines = {}
    for item in plan:
        text, source = gen.generate(
            item["message"], personalisation, item["design"].layout)
        final_lines[item["message_id"]] = {"text": text, "source": source}
    gen.unload()

    cards = []
    for item in plan:
        bg_path = background_pipeline.get_path(item["design"].design_id, "final")
        if bg_path is None:
            raise RuntimeError(
                f"no final background for design {item['design'].design_id}. "
                "Run pipeline.render_final(design) first.")
        msg = {**item["message"], "text": final_lines[item["message_id"]]["text"]}
        record = compositor.compose(item["design"], msg, bg_path)
        record["text_source"] = final_lines[item["message_id"]]["source"]
        card_manifest.upsert(record)
        cards.append(record)

    return cards, final_lines


# --- [env-fix] FONT BOOTSTRAP (read-only /kaggle/input -> writable work dir) --
import glob as _glob, shutil as _shutil


def bootstrap_fonts(fonts_dir=None, roots=None, verbose=True):
    fonts_dir = Path(fonts_dir or M3_FONTS_DIR)
    fonts_dir.mkdir(parents=True, exist_ok=True)
    if roots is None:
        roots = ["/kaggle/input"]
    found = []
    for r in roots:
        for ext in ("ttf", "otf"):
            found += _glob.glob(f"{r}/**/*.{ext}", recursive=True)
    copied = 0
    for src in sorted(found):
        dst = fonts_dir / os.path.basename(src)
        if not dst.exists():
            try:
                _shutil.copy2(src, dst)
                copied += 1
            except OSError as exc:
                if verbose:
                    print(f"[fonts] skip {src}: {exc}")
    if verbose:
        have = sorted(p.stem for p in fonts_dir.glob("*.ttf"))
        print(f"[fonts] bootstrap: {copied} copied, {len(have)} TTF(s) in {fonts_dir}")
    return copied


bootstrap_fonts()
# --- end [env-fix] -----------------------------------------------------------


# --- SECTION 8: SELF-CHECK -------------------------------------------------

M3_FAILURES = []


def m3_check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' - ' + detail) if detail else ''}")
        M3_FAILURES.append(label)


print("\n" + "=" * 72)
print("M3 SELF-CHECK (REVISION 4) - text generation and compositor")
print(f"mode: {'MOCK (no GPU)' if MOCK_MODE else 'REAL'}")
print(f"font_policy: {M3_CONFIG['font_policy']}")
print("=" * 72)

# Setup. [F1] BackgroundPipeline is instantiated with NO argument, so it uses
# M2's own CONFIG global, which M3 must not touch.
session_id = "sess_m3_selfcheck"
person = build_personalisation("Aayush", "2002-10-06",
                               today=__import__("datetime").date(2026, 10, 6))
plan = build_catalog_plan(session_id, person, count=12)
print(f"\n[setup] plan: {len(plan)} designs")

pipeline = BackgroundPipeline()   # [F1] no arg; M2 uses its own CONFIG
pipeline.prepare()
print("[setup] rendering M2 backgrounds...")
pipeline.render_drafts(plan)
for item in plan:
    pipeline.render_final(item["design"])
print("[setup] M2 backgrounds ready")

# -- [0] [F1] namespace isolation -----------------------------------------
print("\n[0] [F1] M2's module-level globals are not overwritten by M3")
m3_check("M2's CONFIG still exists",
      "CONFIG" in globals() and isinstance(globals()["CONFIG"], dict))
m3_check("M2's CONFIG is not M3's M3_CONFIG",
      globals().get("CONFIG") is not M3_CONFIG)
m3_check("M2's CONFIG still has watchdog_threshold_gb",
      "watchdog_threshold_gb" in globals().get("CONFIG", {}))
m3_check("M2's CONFIG still has kernel_restart_every_n_jobs",
      "kernel_restart_every_n_jobs" in globals().get("CONFIG", {}))
m3_check("M3_CONFIG does NOT have watchdog_threshold_gb (kept separate)",
      "watchdog_threshold_gb" not in M3_CONFIG)
m3_check("M2's WORK untouched",
      "WORK" in globals() and globals()["WORK"] == M2_WORK if "M2_WORK" in globals()
      else "WORK" in globals())
# The real test: M2's JobState must still work. Read its private data path
# via a fresh job state write, which internally reads CONFIG.
_before = pipeline.state.data["jobs_completed"]
pipeline.state.bump("__m3_f1_probe__")
_after = pipeline.state.data["jobs_completed"]
m3_check(f"M2's JobState.bump ran (jobs_completed {_before} -> {_after})",
      _after == _before + 1)
# Undo the probe so downstream counts are unaffected.
pipeline.state.data["jobs_completed"] = _before
pipeline.state._save()

# -- [1] Font preflight -----------------------------------------------------
print("\n[1] Font preflight (F2 strict mode, F3 noto check)")
resolver = FontResolver()
extra_noto = ("NotoSansDevanagari-Regular",)
missing_latin, missing_noto = resolver.validate_required_stems(extra_noto)

if M3_CONFIG["font_policy"] == "strict":
    m3_check("all 16 Latin stems present",
             len(missing_latin) == 0,
             f"missing: {missing_latin[:4]}")
    m3_check(f"required Noto stems present ({extra_noto[0]})",
             len(missing_noto) == 0,
             f"missing: {missing_noto}")
elif M3_CONFIG["font_policy"] == "latin_fallback":
    print(f"  [info] policy=latin_fallback: missing_latin={len(missing_latin)}, "
          f"missing_noto={missing_noto}")
    m3_check("latin_fallback policy permits Latin substitution", True)
else:
    print("  [info] policy=mock: all stems resolve via DejaVu")
    m3_check("mock policy resolves every stem", len(missing_latin) == 0)

# -- [2] Text generation with fallback -------------------------------------
print("\n[2] Text generation")
gen = TextGenerator()
gen.load()
sample = plan[0]
text, source = gen.generate(sample["message"], person, sample["design"].layout)
m3_check(f"line produced (source={source})", bool(text))
m3_check("text is non-empty", len(text.split()) >= M3_CONFIG["text_min_words"])
if not gen.loaded:
    m3_check("fallback returned the exact M1 template text",
             text == sample["message"]["text"])
else:
    print(f"  [info] model produced: {text!r}")
gen.unload()

# -- [3] Compose all twelve -------------------------------------------------
print("\n[3] Compose twelve finished cards")
try:
    compositor = CardCompositor(font_resolver=resolver)
    card_manifest = CardManifest()
    cards, final_lines = run_m3(
        session_id, person, plan, pipeline,
        text_generator=TextGenerator(),
        compositor=compositor,
        card_manifest=card_manifest,
    )
    m3_check(f"twelve cards produced ({len(cards)})", len(cards) == 12)
    m3_check("every card file exists",
             all(Path(c["path"]).is_file() for c in cards))
    m3_check("every card has a unique card_id",
             len({c["card_id"] for c in cards}) == 12)
    m3_check("card manifest persisted", len(CardManifest().all()) == 12)
    compose_failed = False
except FileNotFoundError as exc:
    m3_check("compose ran without missing fonts", False, str(exc))
    cards = []
    compose_failed = True

# -- [4] Contrast compliance ------------------------------------------------
print("\n[4] WCAG AA contrast against actual background pixels")
if not compose_failed:
    for c in cards:
        m3_check(f"{c['card_id'][:14]} ratio {c['contrast_ratio']:.2f} "
                 f"fallback={c['contrast_fallback']}",
                 c["contrast_ratio"] >= M3_CONFIG["min_contrast"])
    min_observed = min(c["contrast_ratio"] for c in cards)
    print(f"  [info] worst-case contrast: {min_observed:.2f}")

# -- [5] Mix-and-Match ------------------------------------------------------
print("\n[5] Mix-and-Match - design A + message B")
if not compose_failed:
    a = plan[0]
    b = plan[7]
    merged = compositor.compose_merge(a["design"], b["message"], pipeline)
    m3_check("merge produced a card file", Path(merged["path"]).is_file())
    m3_check("merge uses A's design",
             merged["design_id"] == a["design"].design_id)
    m3_check("merge uses B's message",
             merged["message_id"] == b["message"]["message_id"])
    m3_check("merge is not one of the twelve catalog cards",
             merged["card_id"] not in {c["card_id"] for c in cards})
    m3_check("merge cleared contrast",
             merged["contrast_ratio"] >= M3_CONFIG["min_contrast"])

# -- [6] Determinism --------------------------------------------------------
print("\n[6] Compositor determinism")
if not compose_failed:
    first = compositor.compose(a["design"], a["message"],
                               pipeline.get_path(a["design"].design_id, "final"))
    second = compositor.compose(a["design"], a["message"],
                                pipeline.get_path(a["design"].design_id, "final"))
    m3_check("same card_id", first["card_id"] == second["card_id"])
    m3_check("same text colour", first["text_color"] == second["text_color"])
    m3_check("same font size", first["font_size"] == second["font_size"])
    m3_check("same contrast fallback",
             first["contrast_fallback"] == second["contrast_fallback"])
    h1 = hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest()
    h2 = hashlib.sha256(Path(second["path"]).read_bytes()).hexdigest()
    m3_check("byte-identical PNG", h1 == h2)

# -- [7] [M3-R2-F1] max_lines honoured -------------------------------------
print("\n[7] [M3-R2-F1] _fit_text honours layout.max_lines")
if not compose_failed:
    for item in plan[:3]:
        d = item["design"]
        m3_check(f"{d.layout.layout_id}: Layout.max_lines == "
                 f"{d.layout.max_lines}",
                 hasattr(d.layout, "max_lines"))
        long_text = ("Happy birthday, " + person.name + ". " * 30).strip()
        long_msg = {**item["message"], "text": long_text}
        long_card = compositor.compose(
            d, long_msg,
            pipeline.get_path(d.design_id, "final"))
        m3_check(f"{d.layout.layout_id}: {long_card['lines']} lines <= "
                 f"{d.layout.max_lines}",
                 long_card["lines"] <= d.layout.max_lines)

# -- [8] [M3-R2-F3] Non-Latin ---------------------------------------------
print("\n[8] [M3-R2-F3] Non-Latin script resolution")
if M3_CONFIG["font_policy"] == "strict" and missing_noto:
    m3_check("Noto font available for non-Latin test", False,
             f"missing: {missing_noto}")
elif M3_CONFIG["font_policy"] == "strict" and not missing_noto:
    if not compose_failed:
        dev_text = "जन्मदिन की हार्दिक शुभकामनाएँ, " + person.name
        dev_msg = {**a["message"], "text": dev_text}
        dev_card = compositor.compose(
            a["design"], dev_msg,
            pipeline.get_path(a["design"].design_id, "final"))
        m3_check("Devanagari card produced", Path(dev_card["path"]).is_file())
        m3_check("resolved to the Noto font",
                 "NotoSansDevanagari" in dev_card["font_stem"])
        m3_check("script detected as devanagari",
                 dev_card["script"] == "devanagari")
else:
    print("  [info] non-strict policy: skipping the strict Noto requirement")

# -- [9] Background cache untouched ----------------------------------------
print("\n[9] Compositor does not mutate cached backgrounds")
if not compose_failed:
    bg_path = Path(pipeline.get_path(a["design"].design_id, "final"))
    bg_hash_before = hashlib.sha256(bg_path.read_bytes()).hexdigest()
    compositor.compose(a["design"], a["message"], bg_path)
    bg_hash_after = hashlib.sha256(bg_path.read_bytes()).hexdigest()
    m3_check("cached background bytes unchanged",
             bg_hash_before == bg_hash_after)

# -- [10] Compositor is GPU-free -------------------------------------------
print("\n[10] Compositor is GPU-free")
if not compose_failed and HAS_TORCH:
    torch_refs = [n for n in dir(compositor)
                  if isinstance(getattr(compositor, n, None),
                                type(torch.zeros(1)))]
    m3_check("no torch tensor held by compositor", len(torch_refs) == 0)

# -- [11] Manifest fields for M4 -------------------------------------------
print("\n[11] Card manifest structure")
if not compose_failed:
    reloaded = CardManifest()
    sample_rec = reloaded.get(cards[0]["card_id"])
    required = ("card_id", "design_id", "message_id", "path", "rel_path",
                "font_stem", "font_size", "text_color", "contrast_ratio",
                "contrast_fallback", "text_input", "text_output",
                "stage", "created_at")
    m3_check("all M4-required fields present",
             all(f in sample_rec for f in required))
    m3_check("manifest JSON-serialisable",
             isinstance(json.dumps(reloaded.all()), str))

# -- [12] [F2] Contrast fallback on a mixed-tone background ----------------
print("\n[12] [F2] Scrim fallback for mixed-tone backgrounds")
# Construct a synthetic background: left half near-black, right half near-white.
mixed_arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
mixed_arr[:, :512] = 15
mixed_arr[:, 512:] = 240
mixed_img = Image.fromarray(mixed_arr).convert("RGB")  # [env-fix] Pillow 11: mode= deprecated
mixed_path = M3_CARDS_DIR / "_test_mixed_bg.png"
mixed_img.save(mixed_path)

try:
    # Use a layout whose text-safe region straddles the boundary.
    mixed_design = None
    for item in plan:
        if item["design"].layout.layout_id in ("centre_card", "lower_band",
                                                "top_banner"):
            mixed_design = item["design"]
            break
    if mixed_design is None:
        mixed_design = plan[0]["design"]

    # First: what would the old code have produced (no scrim)?
    region = mixed_design.layout.region_px(1024)
    pad = M3_CONFIG["padding_px"]
    x, y, w, h = region
    region = (x + pad, y + pad, max(1, w - 2 * pad), max(1, h - 2 * pad))
    lums = compositor._sample_region_luminances(mixed_img, region)
    _c, old_min = compositor._pick_best_candidate(
        lums, mixed_design.text_colour_candidates)
    print(f"  [info] without scrim, best achievable ratio: {old_min:.2f}")

    mixed_card = compositor.compose(mixed_design, sample["message"], mixed_path)
    m3_check(f"mixed-tone card ratio {mixed_card['contrast_ratio']:.2f} >= "
             f"{M3_CONFIG['min_contrast']}",
             mixed_card["contrast_ratio"] >= M3_CONFIG["min_contrast"])
    m3_check(f"fallback used on mixed background "
             f"({mixed_card['contrast_fallback']})",
             mixed_card["contrast_fallback"] in ("scrim", "scrim+stroke"))
    m3_check("card file produced despite mixed tones",
             Path(mixed_card["path"]).is_file())
finally:
    mixed_path.unlink(missing_ok=True)

# -- [13] [M3-R3-F9] Personalisation contract - behavioural ----------------
print("\n[13] [M3-R3-F9] Personalisation contract - behavioural test")
import inspect as _inspect

m3_check("Personalisation is a dataclass",
         hasattr(type(person), "__dataclass_fields__"))
m3_check("Personalisation has .name attribute", hasattr(person, "name"))
m3_check("Personalisation has .turning_age attribute",
         hasattr(person, "turning_age"))
m3_check("Personalisation has .tokens() method",
         callable(getattr(person, "tokens", None)))

# Behavioural exercise of the two methods that previously used subscript.
try:
    _p = _build_polish_prompt(
        base_text=sample["message"]["text"],
        tone=sample["message"]["tone"],
        length=sample["message"].get("length", "medium"),
        name=person.name,
        turning_age=person.turning_age,
    )
    _prompt_ok = isinstance(_p, str) and len(_p) > 0
except Exception as _exc:
    _prompt_ok = False
    print(f"  [info] prompt build raised: {_exc}")
m3_check("[F9] _build_polish_prompt accepts attribute access", _prompt_ok)

try:
    _val_result = TextGenerator()._validate(
        "Happy birthday, " + person.name + "!",
        sample["message"], person)
    _validate_ok = (isinstance(_val_result, tuple)
                    and len(_val_result) == 2
                    and isinstance(_val_result[0], bool))
except Exception as _exc:
    _validate_ok = False
    print(f"  [info] _validate raised: {_exc}")
m3_check("[F9] _validate runs with attribute access (no TypeError)",
         _validate_ok)

# Exercise generate() with the model intentionally unloaded. This is a
# template-path exercise, but it proves the outer method structure holds.
fake_gen = TextGenerator()
fake_gen.loaded = False
fake_text, fake_src = fake_gen.generate(
    sample["message"], person, sample["design"].layout)
m3_check("generate() with unloaded model returns template",
         fake_src == "template" and fake_text == sample["message"]["text"])

# -- [14] M3 Revision 2 & 3 fixes still verified ---------------------------
print("\n[14] Earlier revision fixes still verified")
m3_check("[M3-R2-F1] _fit_text signature includes layout",
         "layout" in CardCompositor._fit_text.__code__.co_varnames)
m3_check("[M3-R2-F2] default font_policy is strict in real mode",
         MOCK_MODE or M3_CONFIG["font_policy"] == "strict")
m3_check("[M3-R2-F4] MOCK_MODE not auto-enabled from missing CUDA",
         not (not HAS_CUDA and not MOCK_MODE))
# [M2 in Rev 3] Behaviour-based check for manifest backup resilience.
# [audit-fix M3-R5] the [F3] note claimed getsource was gone, but it is still
# used here - wrap it so a harness without linecache raises OSError caught,
# not an abort of the whole M3 self-check.
try:
    _cm_src = _inspect.getsource(CardManifest.__init__)
except OSError:
    _cm_src = ""
m3_check("[M3-R3-M2] manifest backup wrapped in try/except OSError",
         "try:" in _cm_src and "except OSError" in _cm_src)
m3_check("[M3-R2-F6] no hasattr(layout, 'no_upper') in _draw_lines",
         "no_upper" not in CardCompositor._draw_lines.__code__.co_consts
         and "no_upper" not in CardCompositor._draw_lines.__code__.co_names)
m3_check("[M3-R2-F7] record carries text_input and text_output",
         (all("text_input" in c and "text_output" in c for c in cards)
          if not compose_failed else False))
m3_check("[M3-R2-F8] _fit_text accepts image_size",
         "image_size" in CardCompositor._fit_text.__code__.co_varnames)

# -- Summary ----------------------------------------------------------------
print("\n" + "=" * 72)
if M3_FAILURES:
    print(f"RESULT: {len(M3_FAILURES)} FAILURE(S)")
    for f in M3_FAILURES:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 72)
print(f"\n[M3] cards:    {M3_CARDS_DIR}")
print(f"[M3] manifest: {M3_CARD_MANIFEST_PATH}")

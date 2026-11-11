"""M1 - design token library and template bank. Pure data, pure functions."""

# =============================================================================
# PROJECT BETA-CARD - MILESTONE M1 (REVISION 2)
# Design token library and template bank.
# No GPU. No network. No file I/O. Pure data and pure functions.
#
# Revision 2 changes (AGA audit M1, 4 minor findings):
#   [1] Collision resolver now mutates layout+motif (components of
#       combination_key). Previous loop mutated texture, which is not in the
#       key, so it could never resolve a collision. Latent bug fixed.
#   [2] pick_text_colour marked RESERVED FOR M3.
#   [3] DesignToken.text_colour_candidates marked RESERVED FOR M3.
#   [4] _break_family_runs rng parameter removed (unused).
# =============================================================================

import hashlib
import json
import random
import re
import unicodedata

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime


# =============================================================================
# SECTION 1 - PALETTES AND WCAG CONTRAST
# =============================================================================

MIN_CONTRAST = 4.5  # WCAG AA, normal text


@dataclass(frozen=True, slots=True)
class Palette:
    palette_id: str
    name: str
    background: str   # hex; prompt anchor + compositor fallback fill
    text: str
    accent: str
    secondary: str
    mood: str         # prompt fragment

    def as_dict(self):
        return {
            "palette_id": self.palette_id, "name": self.name,
            "background": self.background, "text": self.text,
            "accent": self.accent, "secondary": self.secondary,
            "mood": self.mood,
        }


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_colour):
    h = hex_colour.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected 6-digit hex, got {hex_colour!r}")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b):
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def is_contrast_safe(background, text, min_ratio=MIN_CONTRAST):
    return contrast_ratio(background, text) >= min_ratio


def pick_text_colour(background, candidates, min_ratio=MIN_CONTRAST):
    """First candidate clearing min_ratio; else the highest-contrast one.

    RESERVED FOR M3 (compositor). Not called in M1.

    Used when a generated background drifts away from the palette we asked
    for, and the palette's own text colour no longer clears WCAG AA against
    the actual pixels. This is the 'contrast failures -> high-contrast token
    swap' edge case required by the directive, Section 6.
    """
    best, best_ratio = candidates[0], -1.0
    for c in candidates:
        r = contrast_ratio(background, c)
        if r >= min_ratio:
            return c
        if r > best_ratio:
            best, best_ratio = c, r
    return best


PALETTES = (
    Palette("midnight_gold", "Midnight Gold", "#0E1A2B", "#F2D479", "#C9A227", "#3A4A63",
            "deep midnight blue with warm antique gold"),
    Palette("blush_cream", "Blush Cream", "#FDF2F0", "#6B2D3C", "#E8A0A8", "#F5D5C8",
            "soft blush pink and warm cream, gentle and airy"),
    Palette("sage_terracotta", "Sage Terracotta", "#EDE7DC", "#4A3B32", "#B5651D", "#8A9A7B",
            "muted sage green with terracotta, earthy and calm"),
    Palette("royal_plum", "Royal Plum", "#2E1A47", "#F5E6C8", "#9B6BC4", "#6B4A8C",
            "deep royal plum with pale cream and violet"),
    Palette("ocean_teal", "Ocean Teal", "#0B3C49", "#E8F6F5", "#46B3A9", "#1E6B72",
            "deep ocean teal with bright aquamarine"),
    Palette("sunset_coral", "Sunset Coral", "#FFF1E6", "#7A2E1E", "#FF7A59", "#FFC4A3",
            "warm sunset coral and peach on soft ivory"),
    Palette("forest_moss", "Forest Moss", "#1E2D1F", "#DCE8C8", "#7FA650", "#3F5A38",
            "dark forest green with moss and pale lichen"),
    Palette("cotton_candy", "Cotton Candy", "#FDF0F7", "#5C2A54", "#F49AC2", "#C6A8E0",
            "cotton candy pink and lilac, bright and sweet"),
    Palette("charcoal_neon", "Charcoal Neon", "#141414", "#E8FF6B", "#00E5C0", "#2B2B2B",
            "near-black charcoal with electric lime and neon aqua"),
    Palette("dusty_rose", "Dusty Rose", "#F2E3E5", "#5A2A35", "#C98B9B", "#E0C4C8",
            "dusty rose and antique mauve, soft and vintage"),
    Palette("sapphire_silver", "Sapphire Silver", "#10233F", "#E6ECF5", "#7FA8D9", "#4A6FA5",
            "deep sapphire blue with cool silver"),
    Palette("warm_amber", "Warm Amber", "#2B1B0E", "#FFD79A", "#E08A2E", "#7A4A1E",
            "warm amber and burnt honey on dark umber"),
    Palette("ivory_ink", "Ivory Ink", "#FAF7F0", "#1C1C1C", "#B03A2E", "#6E6A5E",
            "clean ivory paper with black ink and a single red accent"),
    Palette("lavender_mist", "Lavender Mist", "#EFEAF7", "#382A5C", "#9B8FD4", "#C7BEE8",
            "pale lavender mist with soft violet"),
)

PALETTE_BY_ID = {p.palette_id: p for p in PALETTES}


# =============================================================================
# SECTION 2 - TYPOGRAPHY AND SCRIPT-AWARE FONT RESOLUTION
# =============================================================================

FONT_DIR = "assets/fonts"


@dataclass(frozen=True, slots=True)
class Typography:
    typography_id: str
    name: str
    display_font: str
    body_font: str
    title_body_ratio: float
    line_spacing: float
    tracking: float
    uppercase_title: bool
    mood: str

    def as_dict(self):
        return {
            "typography_id": self.typography_id, "name": self.name,
            "display_font": self.display_font, "body_font": self.body_font,
            "title_body_ratio": self.title_body_ratio,
            "line_spacing": self.line_spacing, "tracking": self.tracking,
            "uppercase_title": self.uppercase_title, "mood": self.mood,
        }


TYPOGRAPHIES = (
    Typography("elegant_serif", "Elegant Serif", "PlayfairDisplay-Bold", "Lato-Regular",
               1.9, 1.35, 0.02, False, "elegant high-contrast serif typography"),
    Typography("geometric_sans", "Geometric Sans", "Montserrat-Bold", "OpenSans-Regular",
               1.7, 1.40, 0.00, False, "clean geometric sans-serif typography"),
    Typography("literary", "Literary", "Merriweather-Bold", "SourceSans-Regular",
               1.8, 1.45, 0.00, False, "warm literary serif typography"),
    Typography("modern_round", "Modern Round", "Poppins-SemiBold", "Inter-Regular",
               1.65, 1.42, 0.01, False, "friendly rounded modern typography"),
    Typography("classical_caps", "Classical Caps", "Cinzel-SemiBold", "EBGaramond-Regular",
               1.6, 1.38, 0.10, True, "classical roman capitals with wide letter spacing"),
    Typography("soft_rounded", "Soft Rounded", "Quicksand-Bold", "Nunito-Regular",
               1.7, 1.50, 0.00, False, "soft rounded playful typography"),
    Typography("editorial", "Editorial", "CormorantGaramond-SemiBold", "Karla-Regular",
               2.1, 1.32, 0.03, False, "refined editorial serif typography"),
    Typography("bold_poster", "Bold Poster", "BebasNeue-Regular", "Roboto-Regular",
               2.3, 1.30, 0.06, True, "bold condensed poster lettering"),
)

TYPOGRAPHY_BY_ID = {t.typography_id: t for t in TYPOGRAPHIES}

NOTO_FAMILY = {
    "latin": "NotoSans-Regular", "cyrillic": "NotoSans-Regular",
    "greek": "NotoSans-Regular", "devanagari": "NotoSansDevanagari-Regular",
    "bengali": "NotoSansBengali-Regular", "gurmukhi": "NotoSansGurmukhi-Regular",
    "gujarati": "NotoSansGujarati-Regular", "tamil": "NotoSansTamil-Regular",
    "telugu": "NotoSansTelugu-Regular", "kannada": "NotoSansKannada-Regular",
    "malayalam": "NotoSansMalayalam-Regular", "arabic": "NotoSansArabic-Regular",
    "hebrew": "NotoSansHebrew-Regular", "thai": "NotoSansThai-Regular",
    "han": "NotoSansSC-Regular", "japanese": "NotoSansJP-Regular",
    "korean": "NotoSansKR-Regular",
}

NO_CASE_TRANSFORM = {
    "han", "japanese", "korean", "arabic", "hebrew", "thai", "devanagari",
    "bengali", "gurmukhi", "gujarati", "tamil", "telugu", "kannada", "malayalam",
}


def resolve_font(typography, script, role="body"):
    """Return a font file stem that can actually render `script`."""
    if script in ("latin", "cyrillic", "greek"):
        return typography.display_font if role == "display" else typography.body_font
    return NOTO_FAMILY.get(script, typography.body_font)


# =============================================================================
# SECTION 3 - LAYOUTS
# =============================================================================


@dataclass(frozen=True, slots=True)
class Layout:
    layout_id: str
    name: str
    text_safe: tuple          # (x, y, w, h) normalised, origin top-left
    align: str
    valign: str
    max_lines: int
    composition: str

    def as_dict(self):
        return {
            "layout_id": self.layout_id, "name": self.name,
            "text_safe": list(self.text_safe), "align": self.align,
            "valign": self.valign, "max_lines": self.max_lines,
            "composition": self.composition,
        }

    def region_px(self, size):
        x, y, w, h = self.text_safe
        return (round(x * size), round(y * size), round(w * size), round(h * size))


LAYOUTS = (
    Layout("centre_card", "Centre Card", (0.12, 0.32, 0.76, 0.36), "center", "middle", 4,
           "generous empty space in the centre of the frame, decorative elements "
           "pushed to the corners and edges"),
    Layout("lower_band", "Lower Band", (0.09, 0.62, 0.82, 0.28), "center", "top", 3,
           "empty space across the lower third, visual interest concentrated in the "
           "upper two thirds"),
    Layout("top_banner", "Top Banner", (0.09, 0.08, 0.82, 0.26), "center", "middle", 3,
           "empty space across the top quarter, decorative detail filling the lower "
           "two thirds"),
    Layout("left_panel", "Left Panel", (0.06, 0.26, 0.40, 0.48), "left", "middle", 5,
           "composition weighted to the right side, clear empty space on the left "
           "half of the frame"),
    Layout("right_panel", "Right Panel", (0.54, 0.26, 0.40, 0.48), "left", "middle", 5,
           "composition weighted to the left side, clear empty space on the right "
           "half of the frame"),
    Layout("bottom_caption", "Bottom Caption", (0.08, 0.74, 0.84, 0.18), "center", "top", 2,
           "large open decorative field above, clean empty strip along the very "
           "bottom edge"),
)

LAYOUT_BY_ID = {l.layout_id: l for l in LAYOUTS}


# =============================================================================
# SECTION 4 - MOTIFS AND TEXTURES (feed the diffusion prompt only)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Motif:
    motif_id: str
    name: str
    family: str
    prompt: str

    def as_dict(self):
        return {"motif_id": self.motif_id, "name": self.name,
                "family": self.family, "prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class Texture:
    texture_id: str
    name: str
    prompt: str

    def as_dict(self):
        return {"texture_id": self.texture_id, "name": self.name, "prompt": self.prompt}


MOTIFS = (
    Motif("balloons", "Balloons", "objects",
          "a scatter of birthday balloons, some rising out of frame"),
    Motif("confetti", "Confetti", "objects",
          "a burst of paper confetti and streamers caught mid-fall"),
    Motif("cake", "Cake", "objects",
          "a single stylised celebration cake with lit candles"),
    Motif("gift_ribbon", "Gift and Ribbon", "objects",
          "wrapped gift boxes with curling satin ribbon"),
    Motif("floral_wreath", "Floral Wreath", "floral",
          "an open circular wreath of seasonal flowers and leaves"),
    Motif("botanical", "Botanical", "floral",
          "trailing botanical stems, leaves and seed heads"),
    Motif("blossom_branch", "Blossom Branch", "floral",
          "a flowering branch sweeping across one corner"),
    Motif("geometric", "Geometric", "geometric",
          "overlapping geometric shapes, circles, arcs and triangles"),
    Motif("art_deco", "Art Deco", "geometric",
          "art deco fan rays and stepped symmetry"),
    Motif("stars_moon", "Stars and Moon", "celestial",
          "a crescent moon with a field of small stars"),
    Motif("bokeh_lights", "Bokeh Lights", "celestial",
          "soft out-of-focus bokeh light circles"),
    Motif("paper_cut", "Paper Cut", "craft",
          "layered paper-cut craft shapes with soft drop shadows"),
)

TEXTURES = (
    Texture("paper_grain", "Paper Grain", "fine paper grain texture"),
    Texture("soft_gradient", "Soft Gradient", "smooth soft colour gradient"),
    Texture("watercolour", "Watercolour", "watercolour paper bleed and wash"),
    Texture("linen_weave", "Linen Weave", "subtle linen fabric weave"),
    Texture("matte_film", "Matte Film", "matte film finish with gentle grain"),
    Texture("risograph", "Risograph", "risograph print texture with slight offset"),
)

MOTIF_BY_ID = {m.motif_id: m for m in MOTIFS}
TEXTURE_BY_ID = {t.texture_id: t for t in TEXTURES}


# =============================================================================
# SECTION 5 - DESIGN TOKEN
# =============================================================================

NEGATIVE_PROMPT = (
    "text, letters, words, typography, font, watermark, signature, caption, "
    "label, logo, numbers, writing, calligraphy, handwriting, "
    "lowres, blurry, jpeg artifacts, deformed, extra limbs, frame, border"
)


@dataclass(frozen=True, slots=True)
class DesignToken:
    design_id: str
    palette: Palette
    typography: Typography
    layout: Layout
    motif: Motif
    texture: Texture

    @property
    def combination_key(self):
        """The triple the diversity rule forbids repeating."""
        return (self.palette.palette_id, self.layout.layout_id, self.motif.motif_id)

    @property
    def text_colour_candidates(self):
        """RESERVED FOR M3 (compositor). Colours the compositor may try, best first.

        The compositor samples the generated background and passes this list
        to pick_text_colour() to resolve the final text colour.
        """
        return [self.palette.text, self.palette.accent, "#FFFFFF", "#111111"]

    def region_px(self, size=1024):
        return self.layout.region_px(size)

    def sd_prompt(self):
        return (
            f"birthday card background, {self.motif.prompt}, "
            f"{self.texture.prompt}, {self.palette.mood}, "
            f"{self.layout.composition}, {self.typography.mood}, "
            "flat illustration, centred composition, high detail, "
            "clean empty space reserved for text"
        )

    @staticmethod
    def sd_negative_prompt():
        return NEGATIVE_PROMPT

    def as_dict(self):
        return {
            "design_id": self.design_id,
            "palette": self.palette.as_dict(),
            "typography": self.typography.as_dict(),
            "layout": self.layout.as_dict(),
            "motif": self.motif.as_dict(),
            "texture": self.texture.as_dict(),
            "sd_prompt": self.sd_prompt(),
            "sd_negative_prompt": self.sd_negative_prompt(),
        }

    def cache_key(self):
        """Stable key for the background cache.

        Only fields that change the pixels are included, so a typography tweak
        does not invalidate an expensive render.
        """
        payload = json.dumps({
            "motif": self.motif.motif_id,
            "texture": self.texture.texture_id,
            "palette": self.palette.palette_id,
            "layout": self.layout.layout_id,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def make_design_id(palette, layout, motif, texture):
    payload = f"{palette.palette_id}|{layout.layout_id}|{motif.motif_id}|{texture.texture_id}"
    return "dsg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_design(palette, typography, layout, motif, texture):
    return DesignToken(
        design_id=make_design_id(palette, layout, motif, texture),
        palette=palette, typography=typography, layout=layout,
        motif=motif, texture=texture,
    )


# =============================================================================
# SECTION 6 - PERSONALISATION AND INPUT VALIDATION
# =============================================================================

MAX_NAME_LENGTH = 32
MIN_YEAR = 1900

_NAME_ALLOWED = re.compile(r"[^\w\s'\-.·]", re.UNICODE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_INJECTION = re.compile(r"[{}<>\\/\"`$]")

_PROFANITY = {
    "fuck", "fucking", "shit", "bitch", "bastard", "cunt", "dick",
    "asshole", "slut", "whore", "nigger", "faggot", "retard",
}


class InputError(ValueError):
    """Raised for input the frontend must reject before submitting a job."""


@dataclass(frozen=True, slots=True)
class Personalisation:
    name: str
    dob: date
    age: int
    turning_age: int
    is_birthday_today: bool
    script: str

    def tokens(self):
        return {
            "name": self.name,
            "age": str(self.age),
            "turning_age": str(self.turning_age),
            "turning_ordinal": ordinal(self.turning_age),
            "age_ordinal": ordinal(self.age),
        }

    def as_dict(self):
        return {
            "name": self.name, "dob": self.dob.isoformat(),
            "age": self.age, "turning_age": self.turning_age,
            "is_birthday_today": self.is_birthday_today, "script": self.script,
        }


def ordinal(n):
    """1 -> 1st, 2 -> 2nd, 11 -> 11th, 24 -> 24th."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def detect_script(text):
    """Best-effort script tag for font selection.

    Returns the script of the first non-Latin letter found, so a name like
    'Priya' stays Latin while 'प्रिया' resolves to Devanagari.
    """
    ranges = (
        ("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF),
        ("gurmukhi", 0x0A00, 0x0A7F), ("gujarati", 0x0A80, 0x0AFF),
        ("tamil", 0x0B80, 0x0BFF), ("telugu", 0x0C00, 0x0C7F),
        ("kannada", 0x0C80, 0x0CFF), ("malayalam", 0x0D00, 0x0D7F),
        ("thai", 0x0E00, 0x0E7F), ("hebrew", 0x0590, 0x05FF),
        ("arabic", 0x0600, 0x06FF), ("cyrillic", 0x0400, 0x04FF),
        ("greek", 0x0370, 0x03FF), ("korean", 0xAC00, 0xD7AF),
        ("japanese", 0x3040, 0x30FF), ("han", 0x4E00, 0x9FFF),
    )
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for tag, lo, hi in ranges:
            if lo <= cp <= hi:
                return tag
    return "latin"


def normalise_name(raw):
    if raw is None:
        raise InputError("name is required")

    name = unicodedata.normalize("NFC", str(raw))
    name = _CONTROL.sub("", name)
    name = " ".join(name.split()).strip()

    if not name:
        raise InputError("name is required")
    if _INJECTION.search(name):
        raise InputError("name contains characters that are not allowed")
    if len(name) > MAX_NAME_LENGTH:
        raise InputError(f"name must be {MAX_NAME_LENGTH} characters or fewer")
    if not any(ch.isalpha() for ch in name):
        raise InputError("name must contain at least one letter")

    # [audit-fix M1-R3] doc only: this deliberately *strips* punctuation that
    # is not name-safe (e.g. "Alice@Home" -> "AliceHome"). Dangerous input was
    # already rejected by the _INJECTION check above; everything else is
    # silently removed rather than rejected. Behaviour kept as-is (M1 freeze).
    cleaned = _NAME_ALLOWED.sub("", name).strip()
    if not cleaned:
        raise InputError("name must contain at least one letter")

    lowered = cleaned.casefold()
    words = re.split(r"[\s'\-·]+", lowered)
    if any(w in _PROFANITY for w in words if w):
        raise InputError("name failed the content filter")

    return cleaned


def parse_dob(raw, today=None):
    today = today or date.today()

    if isinstance(raw, date):
        dob = raw
    else:
        text = str(raw).strip()
        dob = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y"):
            try:
                dob = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
        if dob is None:
            raise InputError("date of birth must be a valid date (YYYY-MM-DD)")

    if dob > today:
        raise InputError("date of birth cannot be in the future")
    if dob.year < MIN_YEAR:
        raise InputError(f"date of birth must be after {MIN_YEAR}")
    if dob.year > today.year:
        raise InputError("date of birth cannot be in the future")

    return dob


def _completed_years(dob, today):
    years = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return years


def build_personalisation(raw_name, raw_dob, today=None):
    today = today or date.today()
    name = normalise_name(raw_name)
    dob = parse_dob(raw_dob, today)

    age = _completed_years(dob, today)
    is_birthday_today = (today.month, today.day) == (dob.month, dob.day)
    turning_age = age if is_birthday_today else age + 1

    if age < 0 or age > 130:
        raise InputError("date of birth is out of range")

    return Personalisation(
        name=name, dob=dob, age=age, turning_age=turning_age,
        is_birthday_today=is_birthday_today, script=detect_script(name),
    )


# =============================================================================
# SECTION 7 - TEMPLATE BANK AND MESSAGE BUILDING
# =============================================================================

TONE_ORDER = ("warm", "funny", "formal", "poetic",
              "playful", "sentimental", "celebratory", "reflective")

LENGTH_BANDS = ("short", "medium", "long")


@dataclass(frozen=True, slots=True)
class Template:
    template_id: str
    tone: str
    length: str
    text: str

    def as_dict(self):
        return {"template_id": self.template_id, "tone": self.tone,
                "length": self.length, "text": self.text}


TEMPLATES = (
    # -- warm ---------------------------------------------------------------
    Template("warm_01", "warm", "short",
             "Happy birthday, {name}. Here's to a year of quiet, steady good things."),
    Template("warm_02", "warm", "medium",
             "{turning_ordinal} looks good on you, {name}. May this year be as kind "
             "to you as you are to everyone else."),
    Template("warm_03", "warm", "medium",
             "Another year, {name}, and still the warmest person in any room. "
             "Happy birthday."),
    Template("warm_04", "warm", "long",
             "Happy {turning_ordinal} birthday, {name}. Whatever this year brings, "
             "I hope it brings it gently, and I hope you know how glad we are "
             "that you're here for it."),
    Template("warm_05", "warm", "short",
             "{name}, happy {turning_ordinal}. You make it look easy."),
    Template("warm_06", "warm", "medium",
             "To {name}, on turning {turning_age} - thank you for being the person "
             "people come back to. Happy birthday."),
    # -- funny --------------------------------------------------------------
    Template("funny_01", "funny", "short",
             "{name}, you're not {turning_age}. You're {turning_age} plus shipping "
             "and handling."),
    Template("funny_02", "funny", "medium",
             "Happy {turning_ordinal}, {name}. Your back will remind you about it "
             "tomorrow."),
    Template("funny_03", "funny", "medium",
             "{name}, at {turning_age} you are officially vintage. Apparently that "
             "means you're worth more now."),
    Template("funny_04", "funny", "long",
             "Congratulations {name}, you have survived {turning_age} years of "
             "being yourself, which by any reasonable measure is an endurance "
             "event. Cake is the medal."),
    Template("funny_05", "funny", "short",
             "{turning_age}, {name}. Still no adult supervision. Excellent."),
    Template("funny_06", "funny", "medium",
             "{name}, the candles are a fire hazard and the cake is a lie. Happy "
             "{turning_ordinal} anyway."),
    # -- formal -------------------------------------------------------------
    Template("formal_01", "formal", "short",
             "Warmest congratulations to {name} on a {turning_ordinal} birthday."),
    Template("formal_02", "formal", "medium",
             "On behalf of everyone here, our sincere good wishes to {name} on the "
             "occasion of a {turning_ordinal} birthday."),
    Template("formal_03", "formal", "medium",
             "It is our privilege to wish {name} a very happy {turning_ordinal} "
             "birthday, and many more to come."),
    Template("formal_04", "formal", "long",
             "With great pleasure we extend to {name} our warmest wishes on the "
             "occasion of a {turning_ordinal} birthday, together with every good "
             "hope for the year ahead."),
    Template("formal_05", "formal", "short",
             "To {name}, on reaching {turning_age}: our very best wishes."),
    Template("formal_06", "formal", "medium",
             "We mark with pleasure the {turning_ordinal} birthday of {name}, and "
             "offer our congratulations."),
    # -- poetic -------------------------------------------------------------
    Template("poetic_01", "poetic", "short",
             "The years arrive like tide, {name}. You remain unbothered."),
    Template("poetic_02", "poetic", "medium",
             "{turning_age} winters in, {name}, and you are all spring. Happy "
             "birthday."),
    Template("poetic_03", "poetic", "medium",
             "Somewhere a candle is learning your name, {name}. Happy "
             "{turning_ordinal}."),
    Template("poetic_04", "poetic", "long",
             "One more year folded into the map of you, {name}. The roads behind "
             "are lit, the roads ahead are open, and the light is entirely your "
             "own. Happy birthday."),
    Template("poetic_05", "poetic", "short",
             "{name}: {turning_age} years of weather, all of it survived."),
    Template("poetic_06", "poetic", "medium",
             "Let this be a small bright hour in a long bright life, {name}. Happy "
             "{turning_ordinal}."),
    # -- playful ------------------------------------------------------------
    Template("playful_01", "playful", "short",
             "Cake first, questions later. Happy {turning_ordinal}, {name}!"),
    Template("playful_02", "playful", "medium",
             "Officially {turning_age}, officially unstoppable. Go on then, {name}."),
    Template("playful_03", "playful", "medium",
             "Confetti has been deployed, {name}. There is no going back. Happy "
             "{turning_ordinal}!"),
    Template("playful_04", "playful", "long",
             "Attention {name}: you are hereby {turning_age} years old and legally "
             "obligated to have an unreasonable amount of fun today. This notice "
             "cannot be appealed."),
    Template("playful_05", "playful", "short",
             "Sprinkles! {name}! {turning_ordinal}! In that order."),
    Template("playful_06", "playful", "medium",
             "Balloons have been dispatched to your location, {name}. Happy "
             "{turning_ordinal}!"),
    # -- sentimental --------------------------------------------------------
    Template("sentimental_01", "sentimental", "short",
             "{name}, {turning_age} years of you is the best thing any of us got "
             "to witness."),
    Template("sentimental_02", "sentimental", "medium",
             "Twelve months ago we didn't know what this year would ask of you, "
             "{name}. You answered well. Happy birthday."),
    Template("sentimental_03", "sentimental", "medium",
             "Every year I am gladder that you were born, {name}. Happy "
             "{turning_ordinal}."),
    Template("sentimental_04", "sentimental", "long",
             "There is a version of every year where you are not in it, {name}, and "
             "I am grateful every single day that we do not live in that one. Happy "
             "{turning_ordinal} birthday."),
    Template("sentimental_05", "sentimental", "short",
             "{name}. Still here. Still ours. Happy {turning_ordinal}."),
    Template("sentimental_06", "sentimental", "medium",
             "You have been quietly holding a lot of people together for "
             "{turning_age} years, {name}. Today we hold you."),
    # -- celebratory --------------------------------------------------------
    Template("celebratory_01", "celebratory", "short",
             "Happy {turning_ordinal} birthday, {name}! The whole room is on its "
             "feet."),
    Template("celebratory_02", "celebratory", "medium",
             "Raise everything - glasses, voices, roof. {name} is {turning_age} "
             "today."),
    Template("celebratory_03", "celebratory", "medium",
             "{turning_age} and thriving. Happy birthday, {name}!"),
    Template("celebratory_04", "celebratory", "long",
             "Today the calendar belongs to {name}. {turning_age} years in, and "
             "this is still the best party anyone has ever thrown. Let's make it "
             "count."),
    Template("celebratory_05", "celebratory", "short",
             "{name} is {turning_age}! Sound the horns!"),
    Template("celebratory_06", "celebratory", "medium",
             "A whole year of {name} ahead of us. Happy {turning_ordinal}! Start "
             "the music."),
    # -- reflective ---------------------------------------------------------
    Template("reflective_01", "reflective", "short",
             "{turning_age} is a good place to stand, {name}."),
    Template("reflective_02", "reflective", "medium",
             "Not every year has to be a big one, {name}. This one can simply be "
             "yours. Happy birthday."),
    Template("reflective_03", "reflective", "medium",
             "Look back kindly, look forward slowly, {name}. Happy {turning_ordinal}."),
    Template("reflective_04", "reflective", "long",
             "You have been {turning_age} for exactly one day, {name}. There is no "
             "rush to decide what it means. Sit with it a while, then go and have "
             "some cake."),
    Template("reflective_05", "reflective", "short",
             "{name}, {turning_age} years. Enough to know better, wise enough not to."),
    Template("reflective_06", "reflective", "medium",
             "Some birthdays are for celebrating and some are for noticing, {name}. "
             "This one can be both."),
)

TEMPLATE_BY_ID = {t.template_id: t for t in TEMPLATES}


def templates_for_tone(tone):
    matches = tuple(t for t in TEMPLATES if t.tone == tone)
    if not matches:
        raise KeyError(f"no templates for tone {tone!r}")
    return matches


def fill_template(template, personalisation):
    """Substitute personalisation slots.

    Deliberately not str.format: a name that slipped past validation containing
    a brace would otherwise raise or inject.
    """
    text = template.text
    for key, value in personalisation.tokens().items():
        text = text.replace("{" + key + "}", value)
    return " ".join(text.split()).strip()


def make_message_id(text, tone, language="en"):
    payload = f"{tone}|{language}|{text}"
    return "msg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_message(template, personalisation, language="en"):
    text = fill_template(template, personalisation)
    return {
        "message_id": make_message_id(text, template.tone, language),
        "text": text, "tone": template.tone, "language": language,
        "length": template.length, "source": "template",
        "template_id": template.template_id,
    }


# =============================================================================
# SECTION 8 - SELECTOR (deterministic, seeded from session_id)
# =============================================================================

DEFAULT_CARD_COUNT = 12


def _stable_seed(*parts):
    """SHA-256 based seed.

    Python's builtin hash() is salted per process, so a job resuming in a fresh
    kernel would otherwise produce a different twelve cards than the one that
    died. That would quietly break the resumability requirement.
    """
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _break_family_runs(motifs):
    """Greedy reorder so no two adjacent motifs share a family, where possible.

    Deterministic: always takes the first eligible motif. The previous
    signature carried an unused rng parameter (AGA finding 4); removed.
    """
    remaining = list(motifs)
    ordered = []
    while remaining:
        last_family = ordered[-1].family if ordered else None
        pick = next((i for i, m in enumerate(remaining) if m.family != last_family), 0)
        ordered.append(remaining.pop(pick))
    return ordered


def _pick_typography(palette, layout, rng):
    """Pair typography to layout so tall layouts get tighter setting."""
    if layout.max_lines >= 5:
        pool = [t for t in TYPOGRAPHIES if t.title_body_ratio <= 1.8]
    elif layout.max_lines <= 2:
        pool = [t for t in TYPOGRAPHIES if t.title_body_ratio >= 1.8]
    else:
        pool = list(TYPOGRAPHIES)
    return rng.choice(pool or list(TYPOGRAPHIES))


def _resolve_design_collision(palette_id, layouts, motifs, layout_start, motif_start, seen):
    """Walk layout and motif forward to find a fresh (palette, layout, motif) triple.

    AGA finding 1: only components of combination_key are mutated here.
    Advancing texture would not help, because texture is not part of the key,
    so the collision would survive every step of the loop.

    Returns (layout_index, motif_index), or None if no fresh triple exists
    within this palette.
    """
    n_l, n_m = len(layouts), len(motifs)
    for dl in range(n_l):
        li = (layout_start + dl) % n_l
        for dm in range(n_m):
            mi = (motif_start + dm) % n_m
            if (palette_id, layouts[li].layout_id, motifs[mi].motif_id) not in seen:
                return li, mi
    return None


def select_designs(count=DEFAULT_CARD_COUNT, *, seed):
    """Pick `count` visually distinct designs.

    Diversity rules enforced:
      * no repeated (palette, layout, motif) combination
      * no repeated palette while the palette library has room
      * motif families spread rather than clustered

    Palette is deliberately never advanced as a collision-avoidance axis: the
    'no repeated palette' rule is the strongest constraint and must hold.
    Layout and motif are the fallback axes.
    """
    if count > len(PALETTES):
        raise ValueError(
            f"requested {count} designs but only {len(PALETTES)} palettes exist; "
            "add palettes rather than repeating one"
        )

    rng = random.Random(_stable_seed("designs", seed))

    palettes = list(PALETTES)
    layouts = list(LAYOUTS)
    motifs = list(MOTIFS)
    textures = list(TEXTURES)

    rng.shuffle(palettes)
    rng.shuffle(layouts)
    rng.shuffle(motifs)
    rng.shuffle(textures)

    motifs = _break_family_runs(motifs)

    designs = []
    seen = set()

    for i in range(count):
        palette = palettes[i % len(palettes)]
        layout_idx = i % len(layouts)
        motif_idx = i % len(motifs)
        texture = textures[i % len(textures)]

        if (palette.palette_id,
                layouts[layout_idx].layout_id,
                motifs[motif_idx].motif_id) in seen:
            resolved = _resolve_design_collision(
                palette.palette_id, layouts, motifs, layout_idx, motif_idx, seen)
            if resolved is None:
                raise RuntimeError(
                    f"could not satisfy design diversity rules for palette "
                    f"{palette.palette_id!r}")
            layout_idx, motif_idx = resolved

        layout = layouts[layout_idx]
        motif = motifs[motif_idx]
        seen.add((palette.palette_id, layout.layout_id, motif.motif_id))

        typography = _pick_typography(palette, layout, rng)
        designs.append(build_design(palette, typography, layout, motif, texture))

    return designs


def select_tones(count=DEFAULT_CARD_COUNT, *, seed):
    """Spread tones across the set so the merge has genuinely different lines."""
    rng = random.Random(_stable_seed("tones", seed))
    tones = list(TONE_ORDER)
    rng.shuffle(tones)
    return [tones[i % len(tones)] for i in range(count)]


def build_catalog_plan(session_id, personalisation, count=DEFAULT_CARD_COUNT,
                       tone=None):
    """Produce the twelve (design, message) pairs for a session.

    Pure function. No GPU, no network, no side effects. This is the contract
    the brain executes against in M2 and M3.

    [audit-fix M1-R1] When `tone` is set every card uses it (mirrors the
    standalone backend's R-C1); count must not exceed the tone's template pool.
    """
    if tone is not None and tone not in TONE_ORDER:
        raise ValueError(f"tone must be one of {TONE_ORDER}, got {tone!r}")
    designs = select_designs(count, seed=session_id)
    if tone is None:
        tones = select_tones(count, seed=session_id)
    else:
        # [audit-fix M1-R2] the bank holds only 6 templates per tone, so a
        # 12-card request cannot be all-one-tone. Honour the requested tone for
        # as many cards as it has templates, then fill the remainder from the
        # other tones (every message stays distinct).
        _primary = len(templates_for_tone(tone))
        _fill_tones = [t for t in TONE_ORDER if t != tone]
        tones = [tone] * min(count, _primary)
        _i = 0
        while len(tones) < count:
            tones.append(_fill_tones[_i % len(_fill_tones)])
            _i += 1

    rng = random.Random(_stable_seed("messages", session_id))
    used_templates = set()
    plan = []

    for index, (design, t) in enumerate(zip(designs, tones)):
        candidates = templates_for_tone(t)

        budget = design.layout.max_lines
        preferred = {2: ("short",), 3: ("short", "medium"),
                     4: ("medium", "long"), 5: ("long", "medium")}.get(budget, ("medium",))

        pool = [t for t in candidates
                if t.length in preferred and t.template_id not in used_templates]
        if not pool:
            pool = [t for t in candidates if t.template_id not in used_templates]
        if not pool:
            pool = list(candidates)

        template = rng.choice(pool)
        used_templates.add(template.template_id)

        message = build_message(template, personalisation)
        plan.append({
            "index": index, "session_id": session_id,
            "design_id": design.design_id, "message_id": message["message_id"],
            "design": design, "message": message,
        })

    return plan


def plan_as_records(plan):
    """Serialisable view of the plan, for the catalog and for the audit log."""  # [audit-fix M4-R6] doc only
    return [
        {
            "index": item["index"], "session_id": item["session_id"],
            "design_id": item["design_id"], "message_id": item["message_id"],
            "design": item["design"].as_dict(), "message": item["message"],
            "background_cache_key": item["design"].cache_key(),
            "sd_prompt": item["design"].sd_prompt(),
        }
        for item in plan
    ]


# =============================================================================
# SECTION 9 - M1 SELF-CHECK
# =============================================================================

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' - ' + detail) if detail else ''}")
        FAILURES.append(label)


print("=" * 72)
print("M1 SELF-CHECK (REVISION 2) - design token library and template bank")
print("=" * 72)

print("\n[1] Palette contrast (text on background, WCAG AA >= 4.5)")
for p in PALETTES:
    ratio = contrast_ratio(p.background, p.text)
    check(f"{p.palette_id:<20} ratio {ratio:5.2f}", ratio >= MIN_CONTRAST)

print("\n[2] Library sizes")
check(f"palettes >= 12 ({len(PALETTES)})", len(PALETTES) >= 12)
check(f"typographies >= 6 ({len(TYPOGRAPHIES)})", len(TYPOGRAPHIES) >= 6)
check(f"layouts >= 6 ({len(LAYOUTS)})", len(LAYOUTS) >= 6)
check(f"motifs >= 12 ({len(MOTIFS)})", len(MOTIFS) >= 12)

print("\n[3] Layout text-safe regions inside the canvas")
for l in LAYOUTS:
    x, y, w, h = l.text_safe
    check(f"{l.layout_id:<16} x+w={x + w:.2f} y+h={y + h:.2f}",
          0.0 <= x and 0.0 <= y and x + w <= 1.0 and y + h <= 1.0)

print("\n[4] Determinism - same session_id, same designs")
a = select_designs(12, seed="sess_abc123")
b = select_designs(12, seed="sess_abc123")
c = select_designs(12, seed="sess_xyz789")
check("repeat call is identical", [d.design_id for d in a] == [d.design_id for d in b])
check("different seed differs", [d.design_id for d in a] != [d.design_id for d in c])

print("\n[5] Diversity rules on the twelve-card set")
designs = select_designs(12, seed="sess_abc123")
combos = [d.combination_key for d in designs]
check(f"12 unique (palette, layout, motif) triples ({len(set(combos))})",
      len(set(combos)) == 12)
pal_counts = Counter(d.palette.palette_id for d in designs)
check(f"no repeated palette ({len(pal_counts)} distinct)", max(pal_counts.values()) == 1)
check("every design has a distinct id", len({d.design_id for d in designs}) == 12)

print("\n[6] Input validation")
p = build_personalisation("Aayush", "2002-10-06", today=date(2026, 10, 6))
check(f"turning age on the day = {p.turning_age}",
      p.turning_age == 24 and p.is_birthday_today)
p2 = build_personalisation("Aayush", "2002-10-07", today=date(2026, 10, 6))
check(f"turning age one day before = {p2.turning_age}",
      p2.turning_age == 24 and not p2.is_birthday_today)
p3 = build_personalisation("Aayush", "2002-10-05", today=date(2026, 10, 6))
check(f"completed age one day after = {p3.age}",
      p3.age == 24 and p3.turning_age == 25)

for bad, why in [("", "empty"), ("   ", "whitespace"), ("<script>", "injection chars"),
                 ("A" * 40, "too long"), ("12345", "no letters")]:
    try:
        build_personalisation(bad, "2002-10-06", today=date(2026, 10, 6))
        check(f"rejects {why}", False)
    except InputError:
        check(f"rejects {why}", True)

try:
    build_personalisation("Test", "2099-01-01", today=date(2026, 10, 6))
    check("rejects future dob", False)
except InputError:
    check("rejects future dob", True)

print("\n[7] Script detection and font resolution")
for sample, expected in [("Aayush", "latin"), ("प्रिया", "devanagari"), ("علي", "arabic"),
                         ("田中", "han"), ("김민", "korean"), ("Мария", "cyrillic")]:
    got = detect_script(sample)
    check(f"{sample!r} -> {got}", got == expected)

typ = TYPOGRAPHIES[0]
check("latin uses the paired display font",
      resolve_font(typ, "latin", "display") == typ.display_font)
check("devanagari falls back to Noto",
      resolve_font(typ, "devanagari", "display").startswith("NotoSansDevanagari"))

print("\n[8] Ordinals")
for n, want in [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                (12, "12th"), (13, "13th"), (21, "21st"), (24, "24th"), (101, "101st")]:
    check(f"{n} -> {ordinal(n)}", ordinal(n) == want)

print("\n[9] Tone spread across the twelve cards")
plan = build_catalog_plan("sess_abc123", p)
tones = Counter(item["message"]["tone"] for item in plan)
check(f"all {len(TONE_ORDER)} tones present ({len(tones)})", len(tones) == len(TONE_ORDER))
check(f"no tone used more than twice (max {max(tones.values())})", max(tones.values()) <= 2)
check("twelve distinct messages", len({i["message_id"] for i in plan}) == 12)

print("\n[10] Compositor contract - text never baked into diffusion")
check("negative prompt forbids lettering",
      "text" in plan[0]["design"].sd_negative_prompt())
check("every design reserves empty space",
      all("empty space" in i["design"].sd_prompt() for i in plan))

print("\n[11] Resumability - plan is reproducible from session_id alone")
plan_again = build_catalog_plan("sess_abc123", p)
check("rebuilt plan matches",
      [i["design_id"] for i in plan] == [i["design_id"] for i in plan_again]
      and [i["message_id"] for i in plan] == [i["message_id"] for i in plan_again])

print("\n[12] Collision resolver - AGA finding 1 fix exercised directly")
# Force a guaranteed collision and confirm the resolver escapes it.
forced_seen = {("midnight_gold", "centre_card", "balloons")}
resolved = _resolve_design_collision(
    "midnight_gold", LAYOUTS, MOTIFS, 0, 0, forced_seen)
check("resolver returns a fresh triple",
      resolved is not None and resolved != (0, 0))
if resolved is not None:
    li, mi = resolved
    fresh = (("midnight_gold", LAYOUTS[li].layout_id, MOTIFS[mi].motif_id)
             not in forced_seen)
    check(f"returned triple is genuinely fresh ({LAYOUTS[li].layout_id}, "
          f"{MOTIFS[mi].motif_id})", fresh)

# Exhaustive check: block every layout/motif for one palette and confirm
# the resolver correctly reports exhaustion rather than looping forever.
exhausted = {("midnight_gold", l.layout_id, m.motif_id)
             for l in LAYOUTS for m in MOTIFS}
check("resolver reports None when no fresh triple exists",
      _resolve_design_collision(
          "midnight_gold", LAYOUTS, MOTIFS, 0, 0, exhausted) is None)

# Confirm the old bug would have been caught: mutating texture cannot resolve
# a collision, because texture is not part of combination_key.
sample = designs[0]
check("texture is not part of combination_key",
      sample.texture.texture_id not in sample.combination_key)

print("\n[13] AGA findings 2-4 verified as resolved")
check("finding 2: pick_text_colour exists (reserved for M3)",
      callable(pick_text_colour))
check("finding 3: text_colour_candidates exists (reserved for M3)",
      isinstance(sample.text_colour_candidates, list))
import inspect
sig = inspect.signature(_break_family_runs)
check(f"finding 4: _break_family_runs signature is {sig}",
      "rng" not in sig.parameters)

print("\n" + "=" * 72)
print("SAMPLE CATALOG PLAN - session sess_abc123, 'Aayush', turning 24")
print("=" * 72)

for r in plan_as_records(plan):
    d, m = r["design"], r["message"]
    print(f"\n  [{r['index']:>2}] {d['palette']['name']} / {d['layout']['name']} / "
          f"{d['motif']['name']} / {d['texture']['name']}")
    print(f"       tone={m['tone']:<12} length={m['length']:<6} "
          f"cache={r['background_cache_key'][:12]}")
    print(f"       \"{m['text']}\"")

print("\n" + "=" * 72)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 72)

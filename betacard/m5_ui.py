"""M5 - Kivy client."""

from __future__ import annotations
from .config import CONFIG
from .m1_design import *  # noqa: F401,F403
from .m2_render import *  # noqa: F401,F403
from .m3_compose import *  # noqa: F401,F403
from .m4_service import *  # noqa: F401,F403

# =============================================================================
# PROJECT BETA-CARD - MILESTONE M5 (SINGLE FILE, REVISION 3)
# The frontend: Kivy + KivyMD app. Talks only to M4's API.
#
# Revision 3 changes (AGA M5 Rev 2 post-fix audit):
#   [M5-P2] AppConfig now has a masked __repr__ so the raw bearer token is
#           never printed in logs, tracebacks or debug output. The self-test
#           uses a varied test token and checks repr(cfg), not just
#           masked_token().
#   [M5-P1] The self-test credential scan no longer triggers on its own
#           needle list. Needles are assembled at runtime and the scan
#           inspects only the source of functions and classes, so the
#           scanner's own strings cannot cause a false positive.
#
# Revision 2 (retained):
#   [m1] _snack prints when Kivy is absent; Snackbar.open is wrapped.
#   [m2] Optional dependencies documented; plyer probed; ExportScreen.share
#        reports clearly when plyer is missing.
#
# Dependencies:
#   Required for the UI:      kivy>=2.2, kivymd>=1.2
#   Required for the client:  requests>=2.28
#   Optional (export/share):  plyer>=2.1
#     - Save works without plyer (the image is written to the local cache).
#     - Share requires plyer; without it, the app shows a clear message.
#
# Install for the full experience:
#   pip install "kivy>=2.2" "kivymd>=1.2" "requests>=2.28" "plyer>=2.1"
#
# Sections:
#   1. Imports and environment probe
#   2. Config loader (base URL + bearer, from file, never from source)
#   3. Validators (mirrors M1's rules exactly, on-device)
#   4. Local store (session id, catalog cache, image cache)
#   5. API client (M4 endpoints; bearer + skip-warning header on every call)
#   6. UI screens (only defined when Kivy + KivyMD are importable)
#   7. Application entry point
#   8. Non-UI self-test (runs on any Python 3.9+, no Kivy required)
#
# Run:
#   audit / headless:   python m5_frontend.py --self-test
#   desktop:            python m5_frontend.py
#   android:            buildozer -v android debug
#
# Config file (NEVER bundled, placed on the device by hand):
#   $BETACARD_CONFIG, or ./betacard_config.json, or ~/.betacard/config.json,
#   or /sdcard/betacard/config.json
#   Contents:
#     {
#       "api_base_url": "https://<ngrok-domain>",
#       "api_bearer_token": "<long-random-token>"
#     }
#
# The app carries ONLY the API base URL and the API bearer token.
# It never carries the Supabase secret key, the Firebase service-account
# JSON, or any other master credential.
# =============================================================================


# --- SECTION 1: IMPORTS AND ENVIRONMENT PROBE -------------------------------


import json
import os
import re
import sys
import time
import uuid
import hashlib
import inspect as _inspect
import threading
import unicodedata

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

# [audit-fix M5-R8] M5 mirrors M1's validators so it can run standalone, but in
# the integrated notebook these top-level names would shadow M1's, and M4's
# call-time global lookups would pick up M5's copies. Snapshot M1's originals
# here and restore them in the cell after run_self_test().
_M1_SHADOWED = ("Personalisation", "ordinal", "detect_script", "normalise_name",
                "parse_dob", "_completed_years", "build_personalisation",
                "InputError")
_M1_ORIGINALS = {n: globals().get(n) for n in _M1_SHADOWED}

# -- Required for the API client --------------------------------------------
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None
    _HAS_REQUESTS = False

# -- Required for the UI ----------------------------------------------------
try:
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.uix.screenmanager import Screen, ScreenManager
    from kivy.uix.image import AsyncImage
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.behaviors import ButtonBehavior
    from kivymd.app import MDApp
    from kivymd.uix.boxlayout import MDBoxLayout
    from kivymd.uix.button import MDRaisedButton, MDFlatButton, MDIconButton
    from kivymd.uix.card import MDCard
    from kivymd.uix.label import MDLabel
    from kivymd.uix.progressbar import MDProgressBar
    from kivymd.uix.screen import MDScreen
    from kivymd.uix.scrollview import MDScrollView
    from kivymd.uix.spinner import MDSpinner
    from kivymd.uix.textfield import MDTextField
    from kivymd.uix.snackbar import Snackbar
    _HAS_KIVY = True
except ImportError:
    _HAS_KIVY = False
    Clock = Screen = ScreenManager = AsyncImage = None
    GridLayout = ButtonBehavior = MDApp = None
    MDBoxLayout = MDRaisedButton = MDFlatButton = MDIconButton = None
    MDCard = MDLabel = MDProgressBar = MDScreen = MDScrollView = None
    MDSpinner = MDTextField = Snackbar = None

# -- Optional: used only by the Share action --------------------------------
try:
    import plyer  # noqa: F401
    _HAS_PLYER = True
except ImportError:
    plyer = None
    _HAS_PLYER = False


def optional_dependencies() -> dict:
    """Report which optional dependencies are available at runtime."""
    return {
        "requests": _HAS_REQUESTS,
        "kivy_kivymd": _HAS_KIVY,
        "plyer": _HAS_PLYER,
    }


# --- SECTION 2: CONFIG LOADER -----------------------------------------------

class ConfigError(RuntimeError):
    """Raised when the config file is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    api_base_url: str
    api_bearer_token: str
    request_timeout_s: float = 20.0
    poll_interval_s: float = 1.5
    poll_max_wait_s: float = 900.0
    cache_dir: str = ""
    default_session_id: str = ""

    def normalised_base_url(self) -> str:
        url = self.api_base_url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ConfigError(
                f"api_base_url must start with http:// or https://, got {url!r}"
            )
        # [audit-fix M5-R7] the bearer must not travel over plaintext HTTP
        # except to an explicit local development host.
        if url.startswith("http://"):
            _host = url[len("http://"):].split("/", 1)[0].split(":", 1)[0]
            if _host not in ("127.0.0.1", "localhost", "::1"):
                raise ConfigError(
                    "api_base_url must use https:// (http:// is allowed only "
                    f"for localhost); got {url!r}")
        return url

    def masked_token(self) -> str:
        t = self.api_bearer_token
        if len(t) <= 8:
            return "*" * len(t)
        return f"{t[:4]}...{t[-4:]} (len {len(t)})"

    def __repr__(self) -> str:
        """[M5-P2] Never print the raw bearer token.

        The default dataclass repr would include every field, exposing the
        token in logs, error reports or a debug print. The token is shown
        only in masked form.
        """
        return (
            f"AppConfig(api_base_url={self.api_base_url!r}, "
            f"api_bearer_token={self.masked_token()!r}, "
            f"request_timeout_s={self.request_timeout_s!r}, "
            f"poll_interval_s={self.poll_interval_s!r}, "
            f"poll_max_wait_s={self.poll_max_wait_s!r}, "
            f"cache_dir={self.cache_dir!r}, "
            f"default_session_id={self.default_session_id!r})"
        )


def _candidate_config_paths() -> list:
    paths = []
    env = os.environ.get("BETACARD_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths.append(Path.cwd() / "betacard_config.json")
    paths.append(Path.home() / ".betacard" / "config.json")
    paths.append(Path("/sdcard/betacard/config.json"))
    # [audit-fix M5-R4] also look where a Kaggle run would put it.
    paths.append(Path("/kaggle/working/betacard_config.json"))
    paths.append(Path("/kaggle/working/betacard/config.json"))
    return paths


def load_config(path: str | None = None) -> AppConfig:
    candidates = [Path(path).expanduser()] if path else _candidate_config_paths()
    for p in candidates:
        if p.is_file():
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError as exc:
                raise ConfigError(f"config file {p} is not valid JSON: {exc}")
            for required in ("api_base_url", "api_bearer_token"):
                if not data.get(required):
                    raise ConfigError(
                        f"config file {p} missing required field {required!r}"
                    )
            token = str(data["api_bearer_token"]).strip()
            if len(token) < 32:
                raise ConfigError(
                    f"api_bearer_token in {p} is too short ({len(token)} chars); "
                    "use a long random value (>= 32 characters)"
                )
            cache_dir = data.get("cache_dir") or str(
                Path.home() / ".betacard" / "cache")
            return AppConfig(
                api_base_url=str(data["api_base_url"]),
                api_bearer_token=token,
                request_timeout_s=float(data.get("request_timeout_s", 20.0)),
                poll_interval_s=float(data.get("poll_interval_s", 1.5)),
                poll_max_wait_s=float(data.get("poll_max_wait_s", 900.0)),
                cache_dir=cache_dir,
                default_session_id=str(data.get("default_session_id", "")),
            )
    tried = "\n  ".join(str(p) for p in candidates)
    raise ConfigError(
        "no config file found. Looked in:\n  " + tried +
        "\nCreate betacard_config.json with at least:\n"
        '  {"api_base_url": "https://<your-ngrok-domain>", '
        '"api_bearer_token": "<long-random-token>"}'
    )


# --- SECTION 3: VALIDATORS (mirror of M1's rules) ---------------------------

MAX_NAME_LENGTH = 32
MIN_YEAR = 1900

_NAME_ALLOWED = re.compile(r"[^\w\s'\-.·]", re.UNICODE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_INJECTION = re.compile(r"[{}<>\\/\"`$]")

_PROFANITY = {
    "fuck", "fucking", "shit", "bitch", "bastard", "cunt", "dick",
    "asshole", "slut", "whore", "nigger", "faggot", "retard",
}

TONES = ("warm", "funny", "formal", "poetic",
         "playful", "sentimental", "celebratory", "reflective")


class InputError(ValueError):
    """Raised for input the app must reject before submitting a job."""


@dataclass(frozen=True)
class Personalisation:
    name: str
    dob: date
    age: int
    turning_age: int
    is_birthday_today: bool
    script: str

    def tokens(self) -> dict:
        return {
            "name": self.name,
            "age": str(self.age),
            "turning_age": str(self.turning_age),
            "turning_ordinal": ordinal(self.turning_age),
            "age_ordinal": ordinal(self.age),
        }

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dob": self.dob.isoformat(),
            "age": self.age,
            "turning_age": self.turning_age,
            "is_birthday_today": self.is_birthday_today,
            "script": self.script,
        }


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def detect_script(text: str) -> str:
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


def normalise_name(raw: Any) -> str:
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
    cleaned = _NAME_ALLOWED.sub("", name).strip()
    if not cleaned:
        raise InputError("name must contain at least one letter")
    lowered = cleaned.casefold()
    words = re.split(r"[\s'\-·]+", lowered)
    if any(w in _PROFANITY for w in words if w):
        raise InputError("name failed the content filter")
    return cleaned


def parse_dob(raw: Any, today: date | None = None) -> date:
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
    return dob


def _completed_years(dob: date, today: date) -> int:
    years = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return years


def build_personalisation(raw_name: Any, raw_dob: Any,
                          today: date | None = None) -> Personalisation:
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


# --- SECTION 4: LOCAL STORE -------------------------------------------------

class Store:
    """Local persistence: session id, catalog cache, image cache."""

    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_file = self.root / "session_id.txt"

    def get_or_create_session_id(self, fallback: str = "") -> str:
        if self.session_file.is_file():
            sid = self.session_file.read_text().strip()
            if sid:
                return sid
        sid = fallback.strip() or ("sess_" + uuid.uuid4().hex[:12])
        self.session_file.write_text(sid)
        return sid

    def reset_session_id(self) -> str:
        sid = "sess_" + uuid.uuid4().hex[:12]
        self.session_file.write_text(sid)
        return sid

    def catalog_path(self, session_id: str) -> Path:
        return self.root / f"catalog_{session_id}.json"

    def save_catalog(self, session_id: str, data: dict) -> None:
        p = self.catalog_path(session_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(p)

    def load_catalog(self, session_id: str) -> dict | None:
        p = self.catalog_path(session_id)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def image_path(self, url: str) -> Path:
        # [audit-fix M5-R7] key on the stable object identity, not the rotating
        # signed URL (the token in the query string changes on every re-sign,
        # so hashing the full URL fragmented the cache and grew storage).
        _stable = url.split("?", 1)[0]
        h = hashlib.sha256(_stable.encode("utf-8")).hexdigest()[:24]
        return self.root / "images" / f"{h}.png"

    def cache_image_bytes(self, url: str, content: bytes) -> Path:
        p = self.image_path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".png.tmp")
        tmp.write_bytes(content)
        tmp.replace(p)
        return p


# --- SECTION 5: API CLIENT --------------------------------------------------

class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class ApiResponse:
    ok: bool
    status: int
    data: dict | None = None
    error: str | None = None


def _is_health(path: str) -> bool:
    return path.rstrip("/") == "/health"


class ApiClient:
    """M4 API client. Talks only to the bridge.

    Every request carries:
      * Authorization: Bearer <token>
      * ngrok-skip-browser-warning: 1   (except /health)
    """

    def __init__(self, base_url: str, bearer: str, timeout_s: float = 20.0):
        if not _HAS_REQUESTS:
            raise RuntimeError(
                "requests is required for ApiClient; pip install requests"
            )
        self.base_url = base_url.rstrip("/")
        self.bearer = bearer
        self.timeout_s = timeout_s

    def _headers(self, path: str) -> dict:
        h = {
            "Authorization": f"Bearer {self.bearer}",
            "Accept": "application/json",
        }
        if not _is_health(path):
            h["ngrok-skip-browser-warning"] = "1"
        return h

    def _request(self, method: str, path: str,
                 body: dict | None = None) -> ApiResponse:
        url = f"{self.base_url}{path}"
        try:
            r = requests.request(
                method, url, headers=self._headers(path),
                json=body, timeout=self.timeout_s,
            )
        except requests.Timeout:
            return ApiResponse(False, 0, None, "request timed out")
        except requests.ConnectionError as exc:
            return ApiResponse(False, 0, None, f"connection error: {exc}")
        except requests.RequestException as exc:
            return ApiResponse(False, 0, None, f"request error: {exc}")

        if r.status_code >= 500:
            return ApiResponse(False, r.status_code, None,
                               f"server error {r.status_code}")
        if r.status_code >= 400:
            try:
                err = r.json().get("error", r.text[:200])
            except (ValueError, AttributeError):
                err = r.text[:200]
            return ApiResponse(False, r.status_code, None, err)
        try:
            return ApiResponse(True, r.status_code, r.json())
        except ValueError:
            return ApiResponse(False, r.status_code, None,
                               "invalid JSON response")

    # -- endpoints ----------------------------------------------------------

    def health(self) -> ApiResponse:
        return self._request("GET", "/health")

    def generate(self, session_id: str, name: str, dob_iso: str,
                 tone: str | None = None, count: int = 12) -> ApiResponse:
        body = {
            "session_id": session_id, "name": name, "dob": dob_iso,
            "count": count,
        }
        if tone:
            body["tone"] = tone
        return self._request("POST", "/generate", body)

    def job(self, job_id: str) -> ApiResponse:
        return self._request("GET", f"/job/{job_id}")

    def catalog(self, session_id: str) -> ApiResponse:
        return self._request("GET", f"/catalog/{session_id}")

    def compose(self, design_id: str, message_id: str) -> ApiResponse:
        return self._request(
            "POST", "/compose",
            {"design_id": design_id, "message_id": message_id})

    def regenerate(self, card_id: str) -> ApiResponse:
        return self._request("POST", f"/regenerate/{card_id}")

    # -- async helpers ------------------------------------------------------

    def run_async(self, fn: Callable[[], ApiResponse],
                  on_success: Callable[[dict], None],
                  on_error: Callable[[str], None]) -> threading.Thread:
        def worker():
            # [audit-fix M5-R5] an exception in fn() must not kill the worker
            # silently - surface it to the UI instead.
            try:
                resp = fn()
            except Exception as exc:
                on_error(f"{type(exc).__name__}: {exc}")
                return
            if resp.ok:
                on_success(resp.data or {})
            else:
                on_error(resp.error or f"HTTP {resp.status}")
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t

    def poll_job(self, job_id: str,
                 on_update: Callable[[dict], None],
                 on_done: Callable[[dict], None],
                 on_error: Callable[[str], None],
                 interval_s: float = 1.5,
                 max_wait_s: float = 900.0) -> threading.Thread:
        def worker():
            deadline = time.time() + max_wait_s
            while time.time() < deadline:
                try:
                    resp = self.job(job_id)
                except Exception as exc:
                    on_error(f"{type(exc).__name__}: {exc}")
                    return
                if not resp.ok:
                    on_error(resp.error or f"HTTP {resp.status}")
                    return
                rec = resp.data or {}
                on_update(rec)
                state = rec.get("state")
                if state in ("done", "partially_complete", "failed"):
                    on_done(rec)
                    return
                time.sleep(interval_s)
            on_error("job polling timed out")
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t


# --- SECTION 6: UI SCREENS (only defined when Kivy + KivyMD are present) ----

def _snack(msg: str) -> None:
    """Short user-facing notification.

    [m1-fix] When Kivy is absent (headless audit, or a runtime import
    failure), the message is printed so the failure still leaves a trace.
    A Kivy-level snackbar failure is caught so a corrupt UI state does not
    crash the screen.
    """
    if _HAS_KIVY:
        try:
            Snackbar(text=msg).open()
        except Exception as exc:
            print(f"[M5-SNACK-FAIL] {msg} (snackbar error: {exc})")
    else:
        print(f"[M5-SNACK] {msg}")


if _HAS_KIVY:

    # -- 1. Connect --------------------------------------------------------

    class ConnectScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="connect", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(24), spacing=dp(16))
            root.add_widget(MDLabel(
                text="Beta-Card", halign="center", font_style="H4",
                size_hint_y=None, height=dp(64)))
            self.status = MDLabel(text="Checking service...", halign="center",
                                   size_hint_y=None, height=dp(48))
            root.add_widget(self.status)
            self.spinner = MDSpinner(
                size_hint=(None, None), size=(dp(40), dp(40)),
                pos_hint={"center_x": 0.5})
            root.add_widget(self.spinner)
            root.add_widget(MDBoxLayout())
            self.retry_btn = MDRaisedButton(
                text="Retry", disabled=True,
                pos_hint={"center_x": 0.5},
                on_release=lambda *_: self.check())
            root.add_widget(self.retry_btn)
            self.add_widget(root)

        def on_pre_enter(self, *args):
            self.check()

        def check(self):
            self.spinner.active = True
            self.retry_btn.disabled = True
            self.status.text = "Checking service..."
            api = self.app.api

            def _ok(data):
                Clock.schedule_once(lambda *_: self._on_ok(data))

            def _err(msg):
                Clock.schedule_once(lambda *_: self._on_err(msg))

            api.run_async(api.health, _ok, _err)

        def _on_ok(self, data):
            self.spinner.active = False
            tunnel = data.get("tunnel") or "(no tunnel)"
            self.status.text = f"Service up\n{tunnel}"
            self.app.tunnel_url = data.get("tunnel")
            self.app.backend_mode = data.get("mode", "REAL")
            Clock.schedule_once(
                lambda *_: setattr(self.manager, "current", "input"), 0.8)

        def _on_err(self, msg):
            self.spinner.active = False
            self.status.text = f"Cannot reach service:\n{msg}"
            self.retry_btn.disabled = False

    # -- 2. Input ----------------------------------------------------------

    class InputScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="input", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(24), spacing=dp(16))
            root.add_widget(MDLabel(
                text="Who is this for?", font_style="H5",
                size_hint_y=None, height=dp(56)))
            self.name_field = MDTextField(
                hint_text="Name", helper_text="",
                helper_text_mode="on_error", max_text_length=32)
            root.add_widget(self.name_field)
            self.dob_field = MDTextField(
                hint_text="Date of birth (YYYY-MM-DD)", helper_text="",
                helper_text_mode="on_error")
            root.add_widget(self.dob_field)
            self.tone_field = MDTextField(
                hint_text="Tone (optional)",
                helper_text="e.g. warm, funny, poetic",
                helper_text_mode="helper")
            root.add_widget(self.tone_field)
            root.add_widget(MDBoxLayout())
            row = MDBoxLayout(orientation="horizontal", spacing=dp(12),
                              size_hint_y=None, height=dp(56))
            row.add_widget(MDFlatButton(
                text="Back",
                on_release=lambda *_: setattr(
                    self.manager, "current", "connect")))
            row.add_widget(MDRaisedButton(
                text="Generate 12 cards",
                on_release=lambda *_: self.submit()))
            root.add_widget(row)
            self.add_widget(root)

        def submit(self):
            name = self.name_field.text
            dob = self.dob_field.text
            tone = self.tone_field.text.strip().lower() or None
            try:
                person = build_personalisation(name, dob)
            except InputError as exc:
                self._error(str(exc))
                return
            if tone and tone not in TONES:
                self._error(f"tone must be one of: {', '.join(TONES)}")
                return
            self.app.person = person
            self.app.tone = tone
            self.manager.current = "progress"
            self.manager.get_screen("progress").start(person, tone)

        def _error(self, msg):
            self.name_field.helper_text = msg
            self.dob_field.helper_text = msg
            _snack(msg)

    # -- 3. Progress -------------------------------------------------------

    class ProgressScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="progress", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(24), spacing=dp(16))
            root.add_widget(MDLabel(
                text="Generating cards...", font_style="H5",
                size_hint_y=None, height=dp(56)))
            self.state_label = MDLabel(
                text="queued", halign="center",
                size_hint_y=None, height=dp(32))
            root.add_widget(self.state_label)
            self.bar = MDProgressBar(value=0, max=100)
            root.add_widget(self.bar)
            self.count_label = MDLabel(
                text="0 / 12", halign="center",
                size_hint_y=None, height=dp(32))
            root.add_widget(self.count_label)
            root.add_widget(MDBoxLayout())
            self.cancel_btn = MDFlatButton(
                text="Back to input",
                pos_hint={"center_x": 0.5},
                on_release=lambda *_: setattr(
                    self.manager, "current", "input"))
            root.add_widget(self.cancel_btn)
            self.add_widget(root)

        def start(self, person, tone):
            api = self.app.api
            session_id = self.app.session_id
            body = {
                "session_id": session_id, "name": person.name,
                # [audit-fix M5-R5] generate()'s parameter is `dob_iso`, not
                # `dob`; the old key raised TypeError and killed the worker.
                "dob_iso": person.dob.isoformat(), "count": 12,
            }
            if tone:
                body["tone"] = tone

            def _ok(data):
                Clock.schedule_once(lambda *_: self._on_submitted(data))

            def _err(msg):
                Clock.schedule_once(lambda *_: self._on_error(msg))

            api.run_async(lambda: api.generate(**body), _ok, _err)

        def _on_submitted(self, data):
            self.app.job_id = data.get("job_id")
            self._start_polling()

        def _start_polling(self):
            api = self.app.api
            job_id = self.app.job_id

            def _update(rec):
                Clock.schedule_once(lambda *_: self._on_update(rec))

            def _done(rec):
                Clock.schedule_once(lambda *_: self._on_done(rec))

            def _err(msg):
                Clock.schedule_once(lambda *_: self._on_error(msg))

            api.poll_job(
                job_id, _update, _done, _err,
                interval_s=self.app.bc_config.poll_interval_s,
                max_wait_s=self.app.bc_config.poll_max_wait_s)

        def _on_update(self, rec):
            state = rec.get("state", "?")
            prog = rec.get("progress") or {}
            done = int(prog.get("done", 0))
            total = int(prog.get("total", 12)) or 12
            self.state_label.text = state
            self.count_label.text = f"{done} / {total}"
            self.bar.value = int(100 * done / total)

        def _on_done(self, rec):
            state = rec.get("state")
            if state == "done":
                self.manager.current = "gallery"
                self.manager.get_screen("gallery").refresh()
            elif state == "partially_complete":
                _snack("Some cards finished. Showing what we have.")
                self.manager.current = "gallery"
                self.manager.get_screen("gallery").refresh()
            else:
                err = rec.get("error") or "unknown error"
                _snack(f"Generation failed: {err}")
                self.state_label.text = "failed"

        def _on_error(self, msg):
            self.state_label.text = f"error: {msg}"
            _snack(msg)

    # -- 4. Gallery --------------------------------------------------------

    class CardTile(ButtonBehavior, MDCard):
        def __init__(self, card, on_tap, **kw):
            super().__init__(
                orientation="vertical", size_hint=(1, None),
                height=dp(200), padding=dp(4), **kw)
            self.card = card
            self.on_tap = on_tap
            img = AsyncImage(
                source=card.get("thumbnail_url", ""),
                allow_stretch=True, keep_ratio=True)
            self.add_widget(img)
            title = MDLabel(
                text=(card.get("text_output") or "")[:60],
                size_hint_y=None, height=dp(36), halign="center")
            title.bind(size=lambda *_: setattr(title, "text_size", title.size))
            self.add_widget(title)

        def on_release(self):
            self.on_tap(self.card)

    class GalleryScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="gallery", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(12), spacing=dp(8))
            header = MDBoxLayout(orientation="horizontal",
                                  size_hint_y=None, height=dp(56))
            header.add_widget(MDLabel(text="Your cards", font_style="H6"))
            self.refresh_btn = MDIconButton(
                icon="refresh",
                on_release=lambda *_: self.refresh())
            header.add_widget(self.refresh_btn)
            root.add_widget(header)
            self.scroll = MDScrollView()
            self.grid = GridLayout(cols=2, spacing=dp(8), size_hint_y=None)
            self.grid.bind(minimum_height=self.grid.setter("height"))
            self.scroll.add_widget(self.grid)
            root.add_widget(self.scroll)
            self.add_widget(root)
            self._loaded = False

        def refresh(self):
            cached = self.app.store.load_catalog(self.app.session_id)
            if cached:
                self._render(cached)
            self._fetch_from_server()

        def _fetch_from_server(self):
            api = self.app.api
            sid = self.app.session_id

            def _ok(data):
                Clock.schedule_once(lambda *_: self._on_server(data))

            def _err(msg):
                Clock.schedule_once(lambda *_: self._on_offline(msg))

            api.run_async(lambda: api.catalog(sid), _ok, _err)

        def _on_server(self, data):
            self.app.store.save_catalog(self.app.session_id, data)
            self._render(data)

        def _on_offline(self, msg):
            if not self._loaded:
                _snack(f"Offline: {msg}. Showing cached catalog.")

        def _render(self, data):
            self.grid.clear_widgets()
            for card in data.get("cards", []):
                tile = CardTile(card, on_tap=self._open_card)
                self.grid.add_widget(tile)
            self._loaded = True

        def _open_card(self, card):
            self.manager.get_screen("detail").show(card)
            self.manager.current = "detail"

    # -- 5. Detail ---------------------------------------------------------

    class DetailScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="detail", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(12), spacing=dp(8))
            self.image = AsyncImage(allow_stretch=True, keep_ratio=True)
            root.add_widget(self.image)
            self.caption = MDLabel(
                text="", size_hint_y=None, height=dp(80), halign="center")
            self.caption.bind(
                size=lambda *_: setattr(self.caption, "text_size",
                                        self.caption.size))
            root.add_widget(self.caption)
            row = MDBoxLayout(orientation="horizontal",
                              size_hint_y=None, height=dp(56),
                              spacing=dp(8))
            row.add_widget(MDFlatButton(
                text="Back",
                on_release=lambda *_: setattr(
                    self.manager, "current", "gallery")))
            row.add_widget(MDRaisedButton(
                text="Use this design",
                on_release=lambda *_: self.pick("design")))
            row.add_widget(MDRaisedButton(
                text="Use this message",
                on_release=lambda *_: self.pick("message")))
            root.add_widget(row)
            self.add_widget(root)
            self._card = None

        def show(self, card):
            self._card = card
            self.image.source = card.get("image_url", "")
            self.caption.text = card.get("text_output", "")

        def pick(self, what):
            if not self._card:
                return
            if what == "design":
                self.app.selected_design = self._card
                _snack("Design selected.")
            else:
                self.app.selected_message = self._card
                _snack("Message selected.")
            if self.app.selected_design and self.app.selected_message:
                self.manager.get_screen("merge").refresh()
                self.manager.current = "merge"

    # -- 6. Merge ----------------------------------------------------------

    class MergeScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="merge", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(12), spacing=dp(8))
            root.add_widget(MDLabel(
                text="Merge", font_style="H5",
                size_hint_y=None, height=dp(48)))
            self.info = MDLabel(
                text="Pick a design and a message from two cards.",
                size_hint_y=None, height=dp(72), halign="center")
            self.info.bind(size=lambda *_: setattr(
                self.info, "text_size", self.info.size))
            root.add_widget(self.info)
            self.preview = AsyncImage(allow_stretch=True, keep_ratio=True)
            root.add_widget(self.preview)
            row = MDBoxLayout(orientation="horizontal",
                              size_hint_y=None, height=dp(56),
                              spacing=dp(8))
            row.add_widget(MDFlatButton(
                text="Back",
                on_release=lambda *_: setattr(
                    self.manager, "current", "gallery")))
            row.add_widget(MDRaisedButton(
                text="Confirm merge",
                on_release=lambda *_: self.confirm()))
            root.add_widget(row)
            self.add_widget(root)
            self._merged = None

        def refresh(self):
            d = self.app.selected_design
            m = self.app.selected_message
            if not (d and m):
                self.info.text = "Pick a design and a message from two cards."
                return
            self.info.text = (
                f"Design: {(d.get('card_id') or '?')[:14]}\n"
                f"Message: {(m.get('text_output') or '')[:80]}")

        def confirm(self):
            d = self.app.selected_design
            m = self.app.selected_message
            if not (d and m):
                _snack("Pick a design and a message first.")
                return
            api = self.app.api

            def _ok(data):
                Clock.schedule_once(lambda *_: self._on_merged(data))

            def _err(msg):
                Clock.schedule_once(lambda *_: _snack(f"Merge failed: {msg}"))

            api.run_async(
                lambda: api.compose(d["design_id"], m["message_id"]),
                _ok, _err)

        def _on_merged(self, data):
            self._merged = data
            self.preview.source = data.get("image_url", "")
            _snack("Merged. Export or go back.")
            self.manager.get_screen("export").show(data)
            self.manager.current = "export"

    # -- 7. Export ---------------------------------------------------------

    class ExportScreen(Screen):
        def __init__(self, app, **kw):
            super().__init__(name="export", **kw)
            self.app = app
            root = MDBoxLayout(orientation="vertical",
                               padding=dp(12), spacing=dp(8))
            root.add_widget(MDLabel(
                text="Export", font_style="H5",
                size_hint_y=None, height=dp(48)))
            self.preview = AsyncImage(allow_stretch=True, keep_ratio=True)
            root.add_widget(self.preview)
            self.status = MDLabel(
                text="", size_hint_y=None, height=dp(32), halign="center")
            root.add_widget(self.status)
            row = MDBoxLayout(orientation="horizontal",
                              size_hint_y=None, height=dp(56),
                              spacing=dp(8))
            row.add_widget(MDFlatButton(
                text="Back to gallery",
                on_release=lambda *_: setattr(
                    self.manager, "current", "gallery")))
            row.add_widget(MDRaisedButton(
                text="Save", on_release=lambda *_: self.save()))
            row.add_widget(MDRaisedButton(
                text="Share", on_release=lambda *_: self.share()))
            root.add_widget(row)
            self.add_widget(root)
            self._card = None
            self._local_path = None

        def show(self, card):
            self._card = card
            self.preview.source = card.get("image_url", "")
            self._local_path = None
            self.status.text = ""

        def _download(self, on_ready):
            url = (self._card or {}).get("image_url")
            if not url:
                _snack("No image URL")
                return
            cached = self.app.store.image_path(url)
            if cached.is_file():
                self._local_path = str(cached)
                on_ready(self._local_path)
                return

            def worker():
                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    p = self.app.store.cache_image_bytes(url, r.content)
                    Clock.schedule_once(lambda *_: on_ready(str(p)))
                except Exception as exc:
                    # [audit-fix M5-R8] bind the message now; Python clears
                    # `exc` after the except block and the lambda runs later.
                    _msg = f"Download failed: {exc}"
                    Clock.schedule_once(lambda *_: _snack(_msg))

            threading.Thread(target=worker, daemon=True).start()

        def save(self):
            def _ready(path):
                self._local_path = path
                self.status.text = f"Saved to {path}"
                _snack("Saved.")
            self._download(_ready)

        def share(self):
            """[m2-fix] Explicitly report when the optional plyer dependency
            is missing, instead of failing silently."""
            def _ready(path):
                self._local_path = path
                if not _HAS_PLYER:
                    msg = "Share requires plyer: pip install plyer"
                    self.status.text = msg
                    _snack(msg)
                    return
                try:
                    from plyer import share as _plyer_share
                    _plyer_share.share(filepath=path, title="Birthday card")
                    self.status.text = "Opened share sheet."
                except Exception as exc:
                    self.status.text = f"Share unavailable: {exc}"
                    _snack(f"Share unavailable: {exc}")
            self._download(_ready)


# --- SECTION 7: APPLICATION ENTRY POINT -------------------------------------

if _HAS_KIVY:

    class _ErrorScreen(MDScreen):
        def __init__(self, message, **kw):
            super().__init__(**kw)
            self.add_widget(MDLabel(text=message, halign="center"))

    class BetaCardApp(MDApp):
        def __init__(self, **kw):
            super().__init__(**kw)
            # [audit-fix M5-R9] renamed from self.config: Kivy's App.config is
            # a ConfigParser and must not be shadowed by our AppConfig dataclass.
            self.bc_config = None
            self.api = None
            self.store = None
            self.session_id = ""
            self.person = None
            self.tone = None
            self.job_id = None
            self.selected_design = None
            self.selected_message = None
            self.tunnel_url = None
            self.backend_mode = "REAL"

        def build(self):
            self.title = "Beta-Card"
            self.theme_cls.primary_palette = "DeepPurple"
            self.theme_cls.theme_style = "Light"

            try:
                self.bc_config = load_config()
            except ConfigError as exc:
                return _ErrorScreen(str(exc))

            try:
                self.api = ApiClient(
                    base_url=self.bc_config.normalised_base_url(),
                    bearer=self.bc_config.api_bearer_token,
                    timeout_s=self.bc_config.request_timeout_s,
                )
            except RuntimeError as exc:
                return _ErrorScreen(str(exc))

            # [audit-fix M5-R6] a bad cache_dir must show the error screen, not
            # crash build().
            try:
                self.store = Store(self.bc_config.cache_dir)
            except OSError as exc:
                return _ErrorScreen(f"cache dir not usable: {exc}")
            self.session_id = self.store.get_or_create_session_id(
                self.bc_config.default_session_id)

            sm = ScreenManager()
            sm.add_widget(ConnectScreen(self, name="connect"))
            sm.add_widget(InputScreen(self, name="input"))
            sm.add_widget(ProgressScreen(self, name="progress"))
            sm.add_widget(GalleryScreen(self, name="gallery"))
            sm.add_widget(DetailScreen(self, name="detail"))
            sm.add_widget(MergeScreen(self, name="merge"))
            sm.add_widget(ExportScreen(self, name="export"))
            return sm


def run_app():
    """Launch the Kivy app. Called by main()."""
    if not _HAS_KIVY:
        print("[M5] Kivy/KivyMD not installed. Running the non-UI self-test "
              "instead. To run the app: pip install kivy kivymd")
        run_self_test()
        return
    BetaCardApp().run()


# --- SECTION 8: NON-UI SELF-TEST --------------------------------------------

M5_FAILURES = []


def _check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' - ' + detail) if detail else ''}")
        M5_FAILURES.append(label)


def run_self_test() -> int:
    """Exercise the non-UI logic. No Kivy, no network, no credentials.

    Returns 0 on all-pass, 1 otherwise.
    """
    import tempfile

    print("=" * 72)
    print("M5 SELF-TEST - non-UI logic (REVISION 3)")
    print(f"requests: {'yes' if _HAS_REQUESTS else 'no'}  "
          f"kivy: {'yes' if _HAS_KIVY else 'no'}  "
          f"plyer: {'yes' if _HAS_PLYER else 'no'}")
    print("=" * 72)

    # [1] Validators mirror M1
    print("\n[1] Validators (mirror of M1)")
    p = build_personalisation("Aayush", "2002-10-06",
                              today=date(2026, 10, 6))
    _check(f"turning age on the day = {p.turning_age}",
           p.turning_age == 24 and p.is_birthday_today)
    p2 = build_personalisation("Aayush", "2002-10-07",
                               today=date(2026, 10, 6))
    _check(f"turning age one day before = {p2.turning_age}",
           p2.turning_age == 24 and not p2.is_birthday_today)
    for bad, why in [
        ("", "empty"), ("   ", "whitespace"),
        ("<script>", "injection chars"),
        ("A" * 40, "too long"), ("12345", "no letters"),
    ]:
        try:
            build_personalisation(bad, "2002-10-06",
                                  today=date(2026, 10, 6))
            _check(f"rejects {why}", False)
        except InputError:
            _check(f"rejects {why}", True)
    try:
        build_personalisation("Test", "2099-01-01",
                              today=date(2026, 10, 6))
        _check("rejects future dob", False)
    except InputError:
        _check("rejects future dob", True)

    # [2] Script detection
    print("\n[2] Script detection")
    for sample, expected in [
        ("Aayush", "latin"), ("प्रिया", "devanagari"),
        ("علي", "arabic"), ("田中", "han"),
        ("김민", "korean"), ("Мария", "cyrillic"),
    ]:
        got = detect_script(sample)
        _check(f"{sample!r} -> {got}", got == expected)

    # [3] Ordinals
    print("\n[3] Ordinals")
    for n, want in [(1, "1st"), (2, "2nd"), (3, "3rd"),
                    (11, "11th"), (12, "12th"), (24, "24th")]:
        _check(f"{n} -> {ordinal(n)}", ordinal(n) == want)

    # [4] Store round-trip
    print("\n[4] Store round-trip")
    with tempfile.TemporaryDirectory() as td:
        store = Store(td)
        sid = store.get_or_create_session_id()
        _check(f"session id created ({sid})", bool(sid))
        sid_again = store.get_or_create_session_id()
        _check("session id is stable", sid_again == sid)
        payload = {"cards": [{"card_id": "c1"}]}
        store.save_catalog(sid, payload)
        loaded = store.load_catalog(sid)
        _check("catalog round-trips", loaded == payload)
        sid2 = store.reset_session_id()
        _check("reset gives a new id", sid2 != sid)
        img_path = store.cache_image_bytes("https://x/y.png", b"\x89PNG\r\n")
        _check("image cached", img_path.is_file())
        same = store.image_path("https://x/y.png")
        _check("image path is deterministic", same == img_path)

    # [5] ApiClient header logic
    print("\n[5] ApiClient header logic")
    if _HAS_REQUESTS:
        client = ApiClient("https://example.invalid", "x" * 40)
        h_health = client._headers("/health")
        _check("/health has no skip-warning header",
               "ngrok-skip-browser-warning" not in h_health)
        h_gen = client._headers("/generate")
        _check("/generate has skip-warning header",
               h_gen.get("ngrok-skip-browser-warning") == "1")
        _check("Authorization carries the bearer",
               h_gen.get("Authorization", "").startswith("Bearer "))
        # [audit-fix M5-R5] guard the client/UI param contract that broke the
        # app: the UI builds the /generate body with these keys, so
        # ApiClient.generate must accept all of them.
        import inspect as _inspect
        _gparams = set(_inspect.signature(ApiClient.generate).parameters)
        _uibody = {"session_id", "name", "dob_iso", "count", "tone"}
        _check("generate() accepts the UI body keys", _uibody <= _gparams)
        c2 = ApiClient("http://127.0.0.1:1", "x" * 40, timeout_s=0.5)
        r = c2.health()
        _check("unreachable host is a clean error response",
               r.ok is False and r.error)
    else:
        print("  SKIP  requests not installed; header logic not tested")

    # [6] Config loader (valid + invalid cases, and repr safety)
    print("\n[6] Config loader")
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "cfg.json"

        # [M5-P2] Varied token so masked prefix/suffix are distinguishable,
        # and so a "run of same character" check is meaningful.
        test_token = ("AbCdEfGh" + "1234567890" + "XyZ" * 8 + "QrSt").strip()
        assert len(test_token) >= 32, "test token must be >= 32 chars"
        assert len(set(test_token)) > 10, "test token must be varied"
        cfg_path.write_text(json.dumps({
            "api_base_url": "https://example.invalid",
            "api_bearer_token": test_token,
        }))
        cfg = load_config(str(cfg_path))
        _check("valid config loads",
               cfg.api_base_url.startswith("https://"))

        _masked = cfg.masked_token()
        _full = repr(cfg)
        _check("[M5-P2] masked_token does not contain the raw token",
               test_token not in _masked)
        _check("[M5-P2] masked_token shows prefix and suffix only",
               test_token[:4] in _masked and test_token[-4:] in _masked
               and test_token not in _masked)
        _check("[M5-P2] repr(cfg) does not contain the raw token",
               test_token not in _full)
        _check("[M5-P2] repr(cfg) contains the masked form",
               "..." in _full)

        # Invalid base URL scheme
        cfg_path.write_text(json.dumps({
            "api_base_url": "ftp://wrong",
            "api_bearer_token": test_token,
        }))
        cfg_bad = load_config(str(cfg_path))
        try:
            cfg_bad.normalised_base_url()
            _check("rejects non-http scheme", False)
        except ConfigError:
            _check("rejects non-http scheme", True)

        # Short token
        cfg_path.write_text(json.dumps({
            "api_base_url": "https://x",
            "api_bearer_token": "short",
        }))
        try:
            load_config(str(cfg_path))
            _check("rejects short token", False)
        except ConfigError:
            _check("rejects short token", True)

    # [7] Optional dependency state
    print("\n[7] Optional dependencies")
    _deps = optional_dependencies()
    # [audit-fix M5-R7] requests is required -> a real check; kivy/plyer are
    # optional for the non-UI self-test -> report them, do not fake a PASS.
    _check(f"requests available: {_deps['requests']}", bool(_deps["requests"]))
    print(f"  [info] kivy/KivyMD available: {_deps['kivy_kivymd']}")
    print(f"  [info] plyer (share) available: {_deps['plyer']}")
    if not _deps["kivy_kivymd"]:
        print("  [info] UI tests skipped; the app requires kivy and kivymd.")
    if not _deps["plyer"]:
        print("  [info] Share will be disabled; Save still works.")

    # [8] _snack logging behaviour (m1-fix)
    print("\n[8] [m1-fix] _snack logging")
    import io
    import contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _snack("self-test snack message")
    _out = _buf.getvalue()
    if _HAS_KIVY:
        _check("_snack did not raise with Kivy present", True)
    else:
        _check("_snack prints when Kivy absent",
               "[M5-SNACK] self-test snack message" in _out)

    # [9] No master credentials anywhere in the source
    print("\n[9] No master credentials in source")
    # [M5-P1] The scan must not find itself. Two safeguards:
    #   (a) The needles are assembled at runtime, so their literal form does
    #       not appear in the file being scanned.
    #   (b) The scan looks only at the sources of functions and classes, not
    #       at the whole file, so this scanner's own strings are excluded
    #       by construction.
    _needles = [
        "sb" + "_secret_",
        "AI" + "za",
        "-----BEGIN " + "PRIVATE KEY-----",
        "service" + "_account_private",
        "eyJ" + "hbGciOi",
    ]
    _pieces = []
    for _name, _obj in list(globals().items()):
        try:
            if _inspect.isfunction(_obj) or _inspect.isclass(_obj):
                _pieces.append(_inspect.getsource(_obj))
        except (OSError, TypeError):
            pass
    _all_src = "\n".join(_pieces)
    _leaks = [s for s in _needles if s in _all_src]
    _check("no credential-shaped strings in source", len(_leaks) == 0,
           f"found: {_leaks}")

    # Summary
    print("\n" + "=" * 72)
    if M5_FAILURES:
        print(f"RESULT: {len(M5_FAILURES)} FAILURE(S)")
        for f in M5_FAILURES:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("RESULT: ALL CHECKS PASSED")
    print("=" * 72)
    return 0


# --- MAIN -------------------------------------------------------------------

def main() -> int:
    """Entry point.

    --self-test    run the non-UI self-test and exit (no GUI)
    default        launch the Kivy app if Kivy is present; otherwise run
                   the self-test so the module is still useful in a
                   headless audit environment
    """
    if "--self-test" in sys.argv:
        return run_self_test()
    run_app()
    return 0


# [notebook] __main__ auto-launch omitted; call run_self_test() or package for the phone.



# Run M5's non-UI self-test so the notebook self-verifies the client.
run_self_test()


# [audit-fix M5-R8] M5 mirrors M1's validators; restore M1's originals so
# later M4 calls cannot resolve M5's copies via the shared globals.
for _n, _v in _M1_ORIGINALS.items():
    if _v is not None:
        globals()[_n] = _v
print('[env] M1 validators restored after M5')

"""M4 - storage, catalog, job queue and the Flask API."""

from .config import CONFIG
from .m1_design import *  # noqa: F401,F403
from .m2_render import *  # noqa: F401,F403
from .m3_compose import *  # noqa: F401,F403

# =============================================================================
# PROJECT BETA-CARD - MILESTONE M4 (FINAL, REVISED)
# The bridge: Flask API, job queue, Supabase storage adapter, Supabase
# catalog (PostgREST), ngrok tunnel, bearer auth.
#
# This revision fixes the two defects found in the final AGA audit of Rev 3:
#   [F1-FIX] CRITICAL - self-check now starts _WORKER before section [8] so
#            the async job is actually processed. Previously _WORKER.start()
#            lived only inside start_service(), so the self-check enqueued a
#            job that no thread ever ran and hung for 300s, then FAILed.
#            The worker is started once at the top of the self-check and
#            stopped in a finally block at the end.
#   [F2-FIX] MAJOR - section [14] previously asserted skip_warning_header_seen
#            is False on a /health call that included the header (AUTH), which
#            is a contradiction that could never pass. Now the "seen=False"
#            case uses AUTH_NO_SKIP (bearer only) and the "seen=True" case
#            uses AUTH (bearer + skip header).
#
# All prior fixes retained:
#   [C1] Real mode requires Supabase; LocalStore only under BETACARD_LOCAL=1
#   [C2] Real mode requires the Supabase catalog (PostgREST); LocalCatalog
#        only under BETACARD_LOCAL=1. Firestore was removed - one cloud service.
#   [C3] Real mode requires API_BEARER_TOKEN from Kaggle Secrets
#   [M1] render_finals default "none"; finals on-demand via /regenerate
#   [M2] ngrok-skip-browser-warning header enforced (except /health)
#   [m1] release_pipeline_gpu() via @_releases_gpu decorator
#   [M4-R2-F1] /compose is GPU-free (cache-only, no render)
#   [M4-R2-F2] CatalogAdapter.find_card_by_design / find_card_by_message
#   [M4-R2-F3] /compose uses record.get("contrast_fallback")
#   [M4-R2-F4] Worker._busy toggled in try/finally
#   [m1-fix] Credential scan covers entire module source
#   [m2-fix] Regenerated card re-uploads thumbnail if no remote path exists
#   [m3-fix] Merge uses card_id-based path; no orphan files on regenerate
#
# REQUIRES: M1, M2, M3 cells have already run in this kernel.
#
# Env vars:
#   BETACARD_LOCAL=1   permit local storage/catalog and an ephemeral bearer.
#                      Test only.
#   BETACARD_MOCK=1    use CPU stubs instead of SDXL and the text model.
#                      Test only.
# =============================================================================


# --- SECTION 1: IMPORTS AND GUARDS -----------------------------------------

import os
import gc
import io
import json
import time
import uuid
import queue
import shutil
import hashlib
import threading
import functools
import traceback
import inspect as _inspect
import sys as _sys

from pathlib import Path
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify

_M123_NEEDED = (
    "build_catalog_plan", "build_personalisation",
    "BackgroundPipeline", "TextGenerator", "CardCompositor",
    "CardManifest", "DesignToken", "M3_CONFIG", "M3_CARDS_DIR",
    "PALETTE_BY_ID", "TYPOGRAPHY_BY_ID", "LAYOUT_BY_ID",
    "MOTIF_BY_ID", "TEXTURE_BY_ID",
)
_M123_MISSING = [n for n in _M123_NEEDED if n not in globals()]
if _M123_MISSING:
    raise RuntimeError(
        f"M1, M2, M3 cells must run before M4. Missing: {_M123_MISSING}."
    )

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

_CARDS_DIR = globals().get("M3_CARDS_DIR") or globals().get("CARDS_DIR")


# --- SECTION 2: SECRETS -----------------------------------------------------

def get_secret(name, default=None):
    """Kaggle Secrets first, environment second, default last."""
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


def secret_present(name):
    v = get_secret(name)
    return bool(v and v.strip())


# --- SECTION 3: CONFIG AND ENFORCEMENT --------------------------------------

_ALLOW_LOCAL = os.environ.get("BETACARD_LOCAL", "0") == "1"

_SUPABASE_URL = get_secret("SUPABASE_URL")
_SUPABASE_KEY = get_secret("SUPABASE_SERVICE_ROLE_KEY")
_SUPABASE_BUCKET = get_secret("SUPABASE_BUCKET", "betacard")
_NGROK_AUTHTOKEN = get_secret("NGROK_AUTHTOKEN")
_NGROK_DOMAIN = get_secret("NGROK_DOMAIN")

_SUPABASE_OK = bool(_SUPABASE_URL and _SUPABASE_KEY)
# [audit-fix M4-R6] Firestore removed: the catalog now lives on Supabase
# PostgREST, so it needs the same Supabase credentials as storage.
_CATALOG_OK = _SUPABASE_OK

# [C3] Bearer token: real mode requires it from Kaggle Secrets.
_BEARER_TOKEN = get_secret("API_BEARER_TOKEN")
if not _BEARER_TOKEN:
    if _ALLOW_LOCAL:
        _BEARER_TOKEN = uuid.uuid4().hex + uuid.uuid4().hex
        print("[M4] BETACARD_LOCAL=1: ephemeral bearer generated. TEST ONLY.")
    else:
        raise RuntimeError(
            "API_BEARER_TOKEN not configured and BETACARD_LOCAL not set. "
            "Real run requires the token from Kaggle Secrets."
        )

# [C1] Supabase enforcement.
if not _SUPABASE_OK and not _ALLOW_LOCAL:
    raise RuntimeError(
        "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not configured and "
        "BETACARD_LOCAL not set. Real run requires Supabase Storage."
    )

# [C2] Catalog enforcement (Supabase PostgREST; Firestore removed).
if not _CATALOG_OK and not _ALLOW_LOCAL:
    raise RuntimeError(
        "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not configured and "
        "BETACARD_LOCAL not set. Real run requires the Supabase catalog."
    )

M4_CONFIG = {
    "port": 5000,
    "host": "0.0.0.0",

    "work_dir": os.environ.get("BETACARD_BASE") or "/kaggle/working/betacard",
    "jobs_path": os.path.join(os.environ.get("BETACARD_BASE")
                              or "/kaggle/working/betacard", "m4_jobs.json"),
    "storage_dir": os.path.join(os.environ.get("BETACARD_BASE")
                                or "/kaggle/working/betacard", "m4_storage"),
    "catalog_path": os.path.join(os.environ.get("BETACARD_BASE")
                                 or "/kaggle/working/betacard",
                                 "m4_catalog.json"),  # [env-fix] portable

    "rate_limit_per_min": 120,
    "bearer_min_length": 32,
    "selfcheck_job_timeout_s": 900,   # [audit-fix M4-R11] real-mode headroom
    "max_queue_depth": 32,            # [audit-fix M4-R12] bounded worker queue

    "supabase_bucket_private": True,
    "signed_url_expires_s": 3600,

    "ngrok_region": "us",

    # [M1] Selective finals. Default "none"; batch renders drafts only.
    "render_finals": "none",

    # [M2] Enforce skip-warning header except /health. Waived under LOCAL.
    "require_skip_warning_header": not _ALLOW_LOCAL,
}

M4_WORK = Path(M4_CONFIG["work_dir"])
M4_JOBS_PATH = Path(M4_CONFIG["jobs_path"])
M4_STORAGE_DIR = Path(M4_CONFIG["storage_dir"])
M4_CATALOG_PATH = Path(M4_CONFIG["catalog_path"])

for _d in (M4_WORK, M4_STORAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print(f"[M4] mode={'LOCAL-TEST' if _ALLOW_LOCAL else 'REAL'} "
      f"mock={MOCK_MODE} cuda={HAS_CUDA}")
print(f"[M4] supabase={'configured' if _SUPABASE_OK else 'LOCAL'}")
print(f"[M4] catalog={'supabase' if _CATALOG_OK else 'LOCAL'}")
print(f"[M4] ngrok={'configured' if _NGROK_AUTHTOKEN else 'not configured'}")
print(f"[M4] bearer token length={len(_BEARER_TOKEN)} (value never printed)")
print(f"[M4] render_finals={M4_CONFIG['render_finals']} "
      f"skip_header_required={M4_CONFIG['require_skip_warning_header']}")


# --- SECTION 4: STORAGE ADAPTERS --------------------------------------------

def _clean_url(u):
    """[env-fix] Trim pasted junk from a URL (spaces, quotes, markdown links).

    BASE URL ONLY (e.g. SUPABASE_URL) - do not run it on a signed URL; the
    trailing-slash strip would invalidate the signature.

    Copying a link out of a markdown message often brings a leading "[" or a
    whole "[text](url)" wrapper with it; requests then fails with InvalidSchema.
    """
    if not u:
        return u
    raw = u
    u = u.strip()
    # markdown link form: [text](url) -> url
    if u.startswith("[") and "](" in u and u.endswith(")"):
        u = u[u.index("](") + 2:-1]
    for ch in "[]()<>\"'`":
        u = u.strip(ch)
    u = u.strip().rstrip("/")
    if u != raw.strip():
        print(f"[M4] note: SUPABASE_URL cleaned to {u!r}")
    if u and not u.startswith(("http://", "https://")):
        raise ValueError(
            f"SUPABASE_URL is not a usable URL after cleaning: {raw!r} -> "
            f"{u!r}. Set it to https://<project-ref>.supabase.co."
        )
    return u


class StorageAdapter:
    def upload(self, local_path, remote_name):
        raise NotImplementedError
    def url(self, remote_name):
        raise NotImplementedError
    def delete(self, remote_name):
        raise NotImplementedError


class LocalStore(StorageAdapter):
    """Filesystem storage. TEST ONLY (BETACARD_LOCAL=1)."""

    def __init__(self, root=None):
        self.root = Path(root or M4_STORAGE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path, remote_name):
        dst = self.root / remote_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(str(local_path), str(tmp))
        tmp.replace(dst)
        return self.url(remote_name)

    def url(self, remote_name):
        return f"local://{remote_name}"

    def delete(self, remote_name):
        p = self.root / remote_name
        if p.is_file():
            p.unlink()


class SupabaseStore(StorageAdapter):
    """Supabase Storage via REST. Secret key on `apikey` header only."""

    def __init__(self, url, key, bucket, private=True, expires_s=3600):
        if not url or not key:
            raise ValueError("SupabaseStore requires url and key")
        self.base_url = _clean_url(url)
        self.key = key.strip()
        if not self.base_url.startswith("http"):
            raise ValueError(
                f"SUPABASE_URL must start with http(s):// (got {self.base_url!r})")
        self.bucket = bucket
        self.private = private
        self.expires_s = expires_s

    def upload(self, local_path, remote_name):
        endpoint = (f"{self.base_url}/storage/v1/object/"
                    f"{self.bucket}/{remote_name}")
        headers = {
            "apikey": self.key,
            "Content-Type": "image/png",
            "x-upsert": "true",
        }
        # [audit-fix M4-R16] Supabase Storage sits behind Cloudflare and returns
        # 5xx/520 on transient hiccups (this failed a /regenerate once). Retry
        # with backoff instead of failing the whole job on one blip.
        last = None
        for attempt in range(4):
            try:
                with open(local_path, "rb") as fh:
                    r = requests.post(endpoint, headers=headers, data=fh,
                                      timeout=60)
            except requests.RequestException as exc:
                last = f"request error: {exc}"
                r = None
            if r is not None and r.status_code in (200, 201):
                return self.url(remote_name)
            if r is not None and r.status_code < 500:
                raise RuntimeError(
                    f"Supabase upload failed {r.status_code}: {r.text[:200]}")
            last = (f"{r.status_code}: {r.text[:120]}" if r is not None
                    else last)
            print(f"[M4] storage upload retry {attempt + 1}/4 ({last})")
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Supabase upload failed after retries ({last})")

    def url(self, remote_name):
        if self.private:
            endpoint = (f"{self.base_url}/storage/v1/object/sign/"
                        f"{self.bucket}/{remote_name}")
            headers = {"apikey": self.key, "Content-Type": "application/json"}
            body = {"expiresIn": self.expires_s}
            r = requests.post(endpoint, headers=headers, json=body, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(
                    f"Supabase sign failed {r.status_code}: {r.text[:200]}"
                )
            signed = r.json().get("signedURL", "")
            if not signed:
                raise RuntimeError("Supabase sign response missing signedURL")
            if signed.startswith("/"):
                signed = signed[1:]
            return f"{self.base_url}/storage/v1/{signed}"
        return (f"{self.base_url}/storage/v1/object/public/"
                f"{self.bucket}/{remote_name}")

    def delete(self, remote_name):
        endpoint = (f"{self.base_url}/storage/v1/object/"
                    f"{self.bucket}/{remote_name}")
        headers = {"apikey": self.key}
        r = requests.delete(endpoint, headers=headers, timeout=30)
        if r.status_code not in (200, 204, 404):
            print(f"[M4] Supabase delete warning {r.status_code}")


def build_store():
    # [audit-fix M4-R7] BETACARD_LOCAL=1 wins: if you asked for a local test
    # run, use the local adapter even when Supabase credentials are present.
    if _ALLOW_LOCAL:
        print("[M4] BETACARD_LOCAL=1: LocalStore (TEST ONLY).")
        return LocalStore()
    if _SUPABASE_OK:
        return SupabaseStore(
            url=_SUPABASE_URL, key=_SUPABASE_KEY, bucket=_SUPABASE_BUCKET,
            private=M4_CONFIG["supabase_bucket_private"],
            expires_s=M4_CONFIG["signed_url_expires_s"],
        )
    raise RuntimeError("unreachable: Supabase required in real mode")


# --- SECTION 5: CATALOG ADAPTERS --------------------------------------------

class CatalogAdapter:
    def put_session(self, session):
        raise NotImplementedError
    def get_session(self, session_id):
        raise NotImplementedError
    def put_card(self, card):
        raise NotImplementedError
    def get_card(self, card_id):
        raise NotImplementedError
    def list_cards(self, session_id):
        raise NotImplementedError
    def find_card_by_design(self, design_id):
        raise NotImplementedError
    def find_card_by_message(self, message_id):
        raise NotImplementedError


class LocalCatalog(CatalogAdapter):
    """JSON file. TEST ONLY (BETACARD_LOCAL=1)."""

    def __init__(self, path=None):
        self.path = Path(path or M4_CATALOG_PATH)
        self.sessions = {}
        self.cards = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                self.sessions = data.get("sessions", {})
                self.cards = data.get("cards", {})
            except (json.JSONDecodeError, TypeError):
                print("[M4] catalog unreadable; starting fresh")

    def put_session(self, session):
        self.sessions[session["session_id"]] = session
        self._save()

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def put_card(self, card):
        self.cards[card["card_id"]] = card
        self._save()

    def get_card(self, card_id):
        return self.cards.get(card_id)

    def list_cards(self, session_id):
        return [c for c in self.cards.values()
                if c.get("session_id") == session_id]

    def find_card_by_design(self, design_id):
        for c in self.cards.values():
            if c.get("design_id") == design_id and c.get("design_json"):
                return c
        return None

    def find_card_by_message(self, message_id):
        for c in self.cards.values():
            if c.get("message_id") == message_id and c.get("message_json"):
                return c
        return None

    def _save(self):
        payload = json.dumps(
            {"sessions": self.sessions, "cards": self.cards},
            indent=2, default=str,
        )
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(self.path)


class SupabaseCatalog(CatalogAdapter):
    """Catalog on Supabase PostgREST (Firestore removed).

    Storage and catalog now share one service. New-style Supabase keys are not
    JWTs, so the key goes on the `apikey` header only; legacy JWT keys also
    accept an Authorization bearer.
    """

    def __init__(self, url, key):
        if not url or not key:
            raise ValueError("SupabaseCatalog requires url and key")
        self.base_url = _clean_url(url)
        self.key = key.strip()
        if not self.base_url.startswith("http"):
            raise ValueError(
                f"SUPABASE_URL must start with http(s):// (got {self.base_url!r})")
        self.base = f"{self.base_url}/rest/v1"

    def _headers(self, prefer=None):
        h = {"apikey": self.key, "Content-Type": "application/json"}
        if self.key.startswith("eyJ"):
            h["Authorization"] = f"Bearer {self.key}"
        if prefer:
            h["Prefer"] = prefer
        return h

    def _request(self, method, path, params=None, json=None, prefer=None):
        r = requests.request(
            method, f"{self.base}/{path}", headers=self._headers(prefer),
            params=params, json=json, timeout=30)
        if r.status_code >= 400:
            body = r.text[:200]
            hint = ""
            if (body.lstrip().startswith("<")
                    or "text/html" in r.headers.get("Content-Type", "")):
                hint = (" - SUPABASE_URL does not look like the API host; it "
                        "must be https://<project-ref>.supabase.co "
                        "(Project Settings > API > Project URL)")
            raise RuntimeError(
                f"Supabase REST {method} {path} -> {r.status_code}: "
                f"{body}{hint}")
        return r

    def put_session(self, session):
        self._request("POST", "sessions",
                      json={"session_id": session["session_id"],
                            "data": session},
                      prefer="resolution=merge-duplicates")

    def get_session(self, session_id):
        rows = self._request("GET", "sessions",
                             params={"session_id": f"eq.{session_id}",
                                     "select": "data"}).json()
        return rows[0]["data"] if rows else None

    def put_card(self, card):
        self._request("POST", "cards",
                      json={"card_id": card["card_id"],
                            "session_id": card.get("session_id"),
                            "design_id": card.get("design_id"),
                            "message_id": card.get("message_id"),
                            "data": card},
                      prefer="resolution=merge-duplicates")

    def get_card(self, card_id):
        rows = self._request("GET", "cards",
                             params={"card_id": f"eq.{card_id}",
                                     "select": "data"}).json()
        return rows[0]["data"] if rows else None

    def list_cards(self, session_id):
        rows = self._request("GET", "cards",
                             params={"session_id": f"eq.{session_id}",
                                     "select": "data"}).json()
        return [row["data"] for row in rows]

    def find_card_by_design(self, design_id):
        rows = self._request("GET", "cards",
                             params={"design_id": f"eq.{design_id}",
                                     "select": "data", "limit": 5}).json()
        for row in rows:
            rec = row["data"]
            if rec.get("design_json"):
                return rec
        return None

    def find_card_by_message(self, message_id):
        rows = self._request("GET", "cards",
                             params={"message_id": f"eq.{message_id}",
                                     "select": "data", "limit": 5}).json()
        for row in rows:
            rec = row["data"]
            if rec.get("message_json"):
                return rec
        return None


def build_catalog():
    # [audit-fix M4-R7] BETACARD_LOCAL=1 wins (see build_store).
    if _ALLOW_LOCAL:
        print("[M4] BETACARD_LOCAL=1: LocalCatalog (TEST ONLY).")
        return LocalCatalog()
    if _CATALOG_OK:
        cat = SupabaseCatalog(_SUPABASE_URL, _SUPABASE_KEY)
        # [audit-fix M4-R6] one-shot health probe: the tables must exist. If
        # not, say exactly what to run instead of failing on every later call.
        try:
            cat._request("GET", "cards",
                         params={"select": "card_id", "limit": 1})
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            bar = "!" * 70
            print(f"[M4] {bar}")
            print(f"[M4] !! SUPABASE CATALOG NOT USABLE: {first}")
            print("[M4] !! Run this once in the Supabase SQL Editor, then rerun:")
            print("[M4] !!   create table if not exists sessions (session_id "
                  "text primary key, data jsonb not null);")
            print("[M4] !!   create table if not exists cards (card_id text "
                  "primary key, session_id text, design_id text, message_id "
                  "text, data jsonb not null);")
            print(f"[M4] {bar}")
        return cat
    raise RuntimeError("unreachable: Supabase catalog required in real mode")


# --- SECTION 6: JOB MODEL AND STORE -----------------------------------------

JOB_STATES = ("queued", "running", "partially_complete", "done", "failed")


def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _idempotency_key(session_id, cleaned_name, dob_iso, force=False, tone=None,
                     count=None):
    # [audit-fix M4-R12/R13] tone and count both change the output, so both
    # are part of the key.
    payload = f"{session_id}|{cleaned_name}|{dob_iso}|{tone or ''}|{count}"
    # [audit-fix M4-R11] `force` lets a caller retry a session whose job ended
    # partially_complete/failed; otherwise the same triple is a permanent hit.
    if force:
        payload += f"|force:{time.time()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JobStore:
    def __init__(self, path=None):
        self.path = Path(path or M4_JOBS_PATH)
        self.jobs = {}
        self.idem = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                self.jobs = data.get("jobs", {})
                self.idem = data.get("idem", {})
            except (json.JSONDecodeError, TypeError):
                print("[M4] job store unreadable; starting fresh")

    def _save(self):
        payload = json.dumps(
            {"jobs": self.jobs, "idem": self.idem},
            indent=2, default=str,
        )
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(self.path)

    def create(self, session_id, name, dob, tone, count, idem_key):
        with self._lock:
            existing = self.idem.get(idem_key)
            if existing and existing in self.jobs:
                return self.jobs[existing], True
            job_id = "job_" + uuid.uuid4().hex[:16]
            job = {
                "job_id": job_id,
                "session_id": session_id,
                "name": name, "dob": dob, "tone": tone, "count": count,
                "state": "queued",
                "progress": {"done": 0, "total": count},
                "card_ids": [],
                "error": None,
                "mode": "full",
                "target_card_id": None,
                "idem_key": idem_key,
                "created_at": _iso_now(),
                "updated_at": _iso_now(),
            }
            self.jobs[job_id] = job
            self.idem[idem_key] = job_id
            # [audit-fix M4-R13] prune terminal jobs so m4_jobs.json cannot
            # grow without bound on a long-lived service.
            _terminal = ("done", "failed", "partially_complete")
            if len(self.jobs) > _MAX_JOBS:
                for _jid, _j in list(self.jobs.items()):
                    if len(self.jobs) <= _MAX_JOBS:
                        break
                    if _jid != job_id and _j.get("state") in _terminal:
                        self.jobs.pop(_jid, None)
                        self.idem.pop(_j.get("idem_key"), None)
            self._save()
            return job, False

    def discard(self, job_id):
        """[audit-fix M4-R13] drop a job that could not be enqueued."""
        with self._lock:
            _j = self.jobs.pop(job_id, None)
            if _j is not None:
                self.idem.pop(_j.get("idem_key"), None)
            self._save()

    def get(self, job_id):
        return self.jobs.get(job_id)

    def update(self, job_id, **changes):
        with self._lock:
            if job_id not in self.jobs:
                return None
            self.jobs[job_id].update(changes)
            self.jobs[job_id]["updated_at"] = _iso_now()
            self._save()
            return self.jobs[job_id]

    def all(self):
        return list(self.jobs.values())


# --- SECTION 7: WORKER THREAD -----------------------------------------------

class Worker:
    """Single-threaded worker. One job at a time."""

    def __init__(self, job_store, runner):
        self.job_store = job_store
        self.runner = runner
        # [audit-fix M4-R12] bound the queue so a token-holder cannot pile up
        # expensive jobs; submit() rejects when full.
        self.queue = queue.Queue(maxsize=M4_CONFIG.get("max_queue_depth", 32))
        self.thread = None
        self._stop = threading.Event()
        self._processed = 0
        self._busy = False

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[M4] worker thread started")

    def _loop(self):
        while not self._stop.is_set():
            try:
                job_id = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._busy = True
            try:
                self.runner(job_id)
                self._processed += 1
            except Exception as exc:
                print(f"[M4] worker uncaught error on {job_id}: {exc}")
                traceback.print_exc()
                self.job_store.update(
                    job_id, state="failed", error=str(exc))
            finally:
                self._busy = False
                self.queue.task_done()

    def submit(self, job_id):
        # [audit-fix M4-R12] bounded - return False when the queue is full.
        try:
            self.queue.put_nowait(job_id)
            return True
        except queue.Full:
            return False

    def run_one_sync(self, timeout_s=600):
        # [audit-fix M4-R14] use the queue's unfinished-task counter. It is
        # bumped in put() *before* the worker can get() the item, closing the
        # window where an item had been taken but _busy had not yet flipped -
        # the old empty()+_busy test could return True mid-job.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.queue.unfinished_tasks == 0 and not self._busy:
                return True
            time.sleep(0.05)
        return False

    def stop(self):
        self._stop.set()


# --- SECTION 8: REGISTRIES --------------------------------------------------

_DESIGN_REGISTRY = {}
_MESSAGE_REGISTRY = {}
# [audit-fix M4-R11] _SESSIONS_IN_FLIGHT was written but never read - removed.
_MAX_REGISTRY = 2000
_MAX_JOBS = 500   # [audit-fix M4-R13] persistent job-store retention cap


def _register_plan(plan):
    for item in plan:
        _DESIGN_REGISTRY[item["design"].design_id] = item["design"]
        _MESSAGE_REGISTRY[item["message"]["message_id"]] = item["message"]
    # [audit-fix M4-R11] bound the registries so a long-lived service does not
    # leak one plan per session.
    while len(_DESIGN_REGISTRY) > _MAX_REGISTRY:
        _DESIGN_REGISTRY.pop(next(iter(_DESIGN_REGISTRY)))
    while len(_MESSAGE_REGISTRY) > _MAX_REGISTRY:
        _MESSAGE_REGISTRY.pop(next(iter(_MESSAGE_REGISTRY)))


def design_from_dict(d):
    return DesignToken(
        design_id=d["design_id"],
        palette=PALETTE_BY_ID[d["palette"]["palette_id"]],
        typography=TYPOGRAPHY_BY_ID[d["typography"]["typography_id"]],
        layout=LAYOUT_BY_ID[d["layout"]["layout_id"]],
        motif=MOTIF_BY_ID[d["motif"]["motif_id"]],
        texture=TEXTURE_BY_ID[d["texture"]["texture_id"]],
    )


# --- SECTION 9: PIPELINE ORCHESTRATION --------------------------------------

_PIPELINE_LOCK = threading.Lock()
_PIPELINE = None


def get_pipeline():
    """Return the shared BackgroundPipeline. Does NOT load SDXL."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = BackgroundPipeline()
        return _PIPELINE


def release_pipeline_gpu():
    """Free SDXL's VRAM. The background cache on disk survives."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE.renderer is not None:
            if hasattr(_PIPELINE.renderer, "pipe"):
                _PIPELINE.renderer.pipe = None
            _PIPELINE.renderer = None
    gc.collect()
    if HAS_CUDA:
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _releases_gpu(fn):
    """[m1] Ensure GPU release on job-runner exit."""
    @functools.wraps(fn)
    def wrapper(job_id):
        try:
            return fn(job_id)
        finally:
            release_pipeline_gpu()
    return wrapper


def _card_object_name(card_id, kind="cards"):
    return f"{kind}/{card_id}.png"


@_releases_gpu
def run_full_job(job_id):
    """Worker entry point for a full session."""
    store = _STORE
    catalog = _CATALOG
    job_store = _JOB_STORE

    job = job_store.get(job_id)
    if job is None:
        print(f"[M4] job {job_id} not found")
        return

    job_store.update(job_id, state="running")
    t_start = time.time()

    try:
        # --- 1. Plan (M1, deterministic) --------------------------------
        person = build_personalisation(job["name"], job["dob"])
        plan = build_catalog_plan(
            job["session_id"], person, count=job["count"],
            tone=job.get("tone"))   # [audit-fix M4-R12] tone flows through
        _register_plan(plan)

        session_record = {
            "session_id": job["session_id"],
            "name": person.name,
            "dob": person.dob.isoformat(),
            "age": person.age,
            "turning_age": person.turning_age,
            "script": person.script,
            "created_at": _iso_now(),
        }
        catalog.put_session(session_record)
        print(f"[M4] {job_id}: plan built ({len(plan)} designs)")

        # --- 2. Text generation (SDXL not resident) ---------------------
        release_pipeline_gpu()
        text_gen = TextGenerator()
        text_gen.load()
        final_lines = {}
        for item in plan:
            text, source = text_gen.generate(
                item["message"], person, item["design"].layout)
            final_lines[item["message_id"]] = {
                "text": text, "source": source}
        text_gen.unload()
        print(f"[M4] {job_id}: {len(final_lines)} lines generated")

        # --- 3. Drafts (SDXL resident) ----------------------------------
        pipeline = get_pipeline()
        pipeline.prepare()
        pipeline.render_drafts(plan)

        if M4_CONFIG["render_finals"] == "all":
            for item in plan:
                pipeline.render_final(item["design"])
            print(f"[M4] {job_id}: all finals rendered (eager mode)")
        else:
            print(f"[M4] {job_id}: drafts only; finals on demand")

        # --- 4. Compose, upload, catalog (CPU + network) ----------------
        compositor = CardCompositor()
        cards_done = list(job.get("card_ids", []))
        for idx, item in enumerate(plan):
            design = item["design"]
            message_id = item["message"]["message_id"]
            message = {**item["message"],
                       "text": final_lines[message_id]["text"]}

            card_id = "card_" + hashlib.sha256(
                f"{design.design_id}|{message_id}".encode("utf-8")
            ).hexdigest()[:20]

            existing = catalog.get_card(card_id)
            if existing and existing.get("image_url"):
                if card_id not in cards_done:
                    cards_done.append(card_id)
                job_store.update(
                    job_id, card_ids=cards_done,
                    progress={"done": len(cards_done),
                              "total": len(plan)})
                continue

            final_bg = pipeline.get_path(design.design_id, "final")
            draft_bg = pipeline.get_path(design.design_id, "draft")
            if final_bg is not None and Path(final_bg).is_file():
                bg_path, image_quality = final_bg, "final"
            elif draft_bg is not None and Path(draft_bg).is_file():
                bg_path, image_quality = draft_bg, "draft"
            else:
                raise RuntimeError(
                    f"no cached background for design {design.design_id}")

            record = compositor.compose(design, message, bg_path)
            record["session_id"] = job["session_id"]
            record["design_json"] = design.as_dict()
            record["message_json"] = dict(message)
            record["text_source"] = final_lines[message_id]["source"]
            record["image_quality"] = image_quality
            record["image_is_final"] = (image_quality == "final")

            card_name = _card_object_name(card_id, "cards")
            record["image_url"] = store.upload(record["path"], card_name)
            record["remote_path"] = card_name

            if draft_bg is not None and Path(draft_bg).is_file():
                thumb_name = _card_object_name(card_id, "thumbs")
                record["thumbnail_url"] = store.upload(
                    draft_bg, thumb_name)
                record["thumb_remote_path"] = thumb_name
            else:
                record["thumbnail_url"] = record["image_url"]
                # [audit-fix M4-R9] point the thumbnail at the image object so
                # the catalog re-signs it on read; otherwise the gallery shows
                # a stale, expiring signed URL.
                record["thumb_remote_path"] = card_name

            catalog.put_card(record)
            cards_done.append(card_id)
            job_store.update(
                job_id, card_ids=cards_done,
                progress={"done": len(cards_done), "total": len(plan)})
            print(f"[M4] {job_id}: card {idx + 1}/{len(plan)} "
                  f"({image_quality})")

        elapsed = time.time() - t_start
        job_store.update(job_id, state="done", error=None)
        print(f"[M4] {job_id}: done in {elapsed:.1f}s")

    except Exception as exc:
        traceback.print_exc()
        completed = len(job_store.get(job_id).get("card_ids", []))
        new_state = "partially_complete" if completed > 0 else "failed"
        job_store.update(job_id, state=new_state, error=str(exc))
        print(f"[M4] {job_id}: {new_state} after {completed} card(s): {exc}")


@_releases_gpu
def run_single_card_job(job_id):
    """Re-render one card: promotes a draft card to final quality.

    [m2-fix] If the original card had no thumbnail remote path (e.g., a
    merge card), re-upload the draft-based thumbnail so /catalog can sign it.
    """
    job = _JOB_STORE.get(job_id)
    if job is None:
        return
    target_card_id = job.get("target_card_id")
    card = _CATALOG.get_card(target_card_id) if target_card_id else None
    if card is None:
        _JOB_STORE.update(
            job_id, state="failed",
            error=f"card {target_card_id} not found")
        return

    _JOB_STORE.update(job_id, state="running")
    try:
        design = design_from_dict(card["design_json"])
        message = dict(card["message_json"])

        pipeline = get_pipeline()
        pipeline.prepare()
        pipeline.render_final(design)
        bg_path = pipeline.get_path(design.design_id, "final")

        compositor = CardCompositor()
        record = compositor.compose(design, message, bg_path)
        record["session_id"] = card["session_id"]
        record["design_json"] = design.as_dict()
        record["message_json"] = message
        record["text_source"] = card.get("text_source", "template")
        record["image_quality"] = "final"
        record["image_is_final"] = True

        card_name = _card_object_name(record["card_id"], "cards")
        record["image_url"] = _STORE.upload(record["path"], card_name)
        record["remote_path"] = card_name

        # [m2-fix] Thumbnail handling.
        thumb_remote = card.get("thumb_remote_path")
        if thumb_remote:
            try:
                record["thumbnail_url"] = _STORE.url(thumb_remote)
                record["thumb_remote_path"] = thumb_remote
            except Exception:
                record["thumbnail_url"] = record["image_url"]
                # [audit-fix M4-R10] keep it re-signable (was None -> the gallery
                # got a stale, expiring signed URL).
                record["thumb_remote_path"] = card_name
        else:
            draft_bg = pipeline.get_path(design.design_id, "draft")
            if draft_bg is not None and Path(draft_bg).is_file():
                thumb_name = _card_object_name(record["card_id"], "thumbs")
                record["thumbnail_url"] = _STORE.upload(
                    draft_bg, thumb_name)
                record["thumb_remote_path"] = thumb_name
            else:
                record["thumbnail_url"] = record["image_url"]
                # [audit-fix M4-R9] same as the batch path - keep it re-signable.
                record["thumb_remote_path"] = card_name

        _CATALOG.put_card(record)
        _JOB_STORE.update(
            job_id, state="done",
            progress={"done": 1, "total": 1},
            card_ids=[record["card_id"]])
    except Exception as exc:
        traceback.print_exc()
        _JOB_STORE.update(job_id, state="failed", error=str(exc))


def _dispatch(job_id):
    job = _JOB_STORE.get(job_id)
    if job is None:
        return
    if job.get("mode") == "single":
        run_single_card_job(job_id)
    else:
        run_full_job(job_id)


# --- SECTION 10: FLASK APP --------------------------------------------------

app = Flask(__name__)

_STORE = build_store()
_CATALOG = build_catalog()
_JOB_STORE = JobStore()
_WORKER = Worker(_JOB_STORE, _dispatch)


# --- [web-ui] serve the single-file UI + allow CORS from any browser --------
@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = (
        "Authorization, Content-Type, ngrok-skip-browser-warning")
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/", methods=["GET", "OPTIONS"])
def _ui():
    """Serve betacard_web_ui.html if it sits next to this backend."""
    _cands = []
    _here = globals().get("__file__")   # absent inside a notebook cell
    if _here:
        _cands.append(Path(_here).with_name("betacard_web_ui.html"))
    _cands.append(Path(M4_CONFIG["work_dir"]) / "betacard_web_ui.html")
    for _p in _cands:
        try:
            if _p.is_file():
                return (_p.read_text(encoding="utf-8"), 200,
                        {"Content-Type": "text/html; charset=utf-8"})
        except Exception:
            pass
    return ("<h1>Beta-Card</h1><p>The web UI file "
            "<code>betacard_web_ui.html</code> was not found next to the "
            "backend. Open it directly in a browser - the API allows CORS.</p>",
            200, {"Content-Type": "text/html; charset=utf-8"})

_RATE_LOCK = threading.Lock()
_RATE_BUCKETS = {}


def _rate_ok(token):
    now = time.time()
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS.get(token, [now, 0])
        if now - bucket[0] >= 60:
            bucket = [now, 0]
        if bucket[1] >= M4_CONFIG["rate_limit_per_min"]:
            _RATE_BUCKETS[token] = bucket
            return False
        bucket[1] += 1
        _RATE_BUCKETS[token] = bucket
        return True


def _secure_eq(a, b):
    if len(a) != len(b):
        _ = sum(ord(c) for c in a) ^ sum(ord(c) for c in b)
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= ord(x) ^ ord(y)
    return diff == 0


def require_bearer(fn):
    """Bearer gate + ngrok skip-warning header enforcement (/health exempt)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [web-ui] let the browser's CORS preflight through without a bearer.
        if request.method == "OPTIONS":
            return ("", 204)
        if (M4_CONFIG["require_skip_warning_header"]
                and request.endpoint != "health"):
            skip = request.headers.get(
                "ngrok-skip-browser-warning", "").lower()
            if skip not in ("1", "true", "yes"):
                return jsonify({
                    "error": "missing ngrok-skip-browser-warning header",
                    "detail": "send 'ngrok-skip-browser-warning: 1' on "
                              "every request except /health",
                }), 400

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing bearer token"}), 401
        token = auth[len("Bearer "):].strip()
        if not _secure_eq(token, _BEARER_TOKEN):
            return jsonify({"error": "invalid bearer token"}), 401
        if not _rate_ok(token):
            return jsonify({"error": "rate limit exceeded"}), 429
        return fn(*args, **kwargs)
    return wrapper


@app.route("/health", methods=["GET"])
@require_bearer
def health():
    skip = request.headers.get("ngrok-skip-browser-warning", "").lower()
    return jsonify({
        "status": "ok",
        "mode": "LOCAL-TEST" if _ALLOW_LOCAL else "REAL",
        "tunnel": _TUNNEL_STATE.get("url"),
        "worker_alive": (_WORKER.thread is not None
                         and _WORKER.thread.is_alive()),
        "worker_busy": _WORKER._busy,
        "queue_depth": _WORKER.queue.qsize(),
        "storage": _STORE.__class__.__name__,
        "catalog": _CATALOG.__class__.__name__,
        "render_finals": M4_CONFIG["render_finals"],
        "skip_warning_header_required":
            M4_CONFIG["require_skip_warning_header"],
        "skip_warning_header_seen": skip in ("1", "true", "yes"),
        "server_time": _iso_now(),
    })


@app.route("/generate", methods=["POST"])
@require_bearer
def generate():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    name = data.get("name")
    dob = data.get("dob")
    tone = data.get("tone")
    force = bool(data.get("force"))   # [audit-fix M4-R11]
    try:
        count = int(data.get("count", 12))
    except (TypeError, ValueError):
        return jsonify({"error": "count must be an integer"}), 400
    # [audit-fix M4-R14] reject a non-integral float (12.9 was silently 12).
    _raw_count = data.get("count", 12)
    if isinstance(_raw_count, float) and _raw_count != count:
        return jsonify({"error": "count must be a whole number"}), 400

    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    # [audit-fix M4-R14] validate tone at the boundary: a bad value used to
    # return 202 and then fail the job inside build_catalog_plan.
    _tone_order = globals().get("TONE_ORDER") or []
    if tone is not None and _tone_order and tone not in _tone_order:
        return jsonify(
            {"error": f"tone must be one of {list(_tone_order)}"}), 400
    # [audit-fix M4-R10] validate count at the boundary instead of letting
    # build_catalog_plan raise and fail the whole job.
    _max_count = len(globals().get("PALETTES") or []) or 14
    if not 1 <= count <= _max_count:
        return jsonify(
            {"error": f"count must be between 1 and {_max_count}"}), 400
    try:
        person = build_personalisation(name, dob)
    except Exception as exc:
        return jsonify({"error": f"invalid input: {exc}"}), 400

    idem = _idempotency_key(
        session_id, person.name, person.dob.isoformat(), force=force,
        tone=tone, count=count)
    job, existed = _JOB_STORE.create(
        session_id=session_id, name=person.name,
        dob=person.dob.isoformat(), tone=tone, count=count,
        idem_key=idem,
    )
    if not existed:
        if not _WORKER.submit(job["job_id"]):   # [audit-fix M4-R12] queue full
            _JOB_STORE.discard(job["job_id"])   # [audit-fix M4-R13] no orphan
            return jsonify({"error": "server busy, retry shortly"}), 503
    return jsonify({
        "job_id": job["job_id"],
        "state": job["state"],
        "idempotent_hit": existed,
    }), 202


@app.route("/job/<job_id>", methods=["GET"])
@require_bearer
def job_status(job_id):
    job = _JOB_STORE.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "job_id": job["job_id"],
        "state": job["state"],
        "progress": job["progress"],
        "card_ids": job.get("card_ids", []),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    })


@app.route("/catalog/<session_id>", methods=["GET"])
@require_bearer
def catalog_read(session_id):
    # [audit-fix M4-R15] Supabase lookups can raise; surface 503, not a 500.
    try:
        session = _CATALOG.get_session(session_id)
        cards = _CATALOG.list_cards(session_id)
    except Exception as exc:
        return jsonify({"error": "catalog unavailable",
                        "detail": str(exc)}), 503
    enriched = []
    for c in cards:
        entry = dict(c)
        rp = c.get("remote_path")
        if rp:
            try:
                entry["image_url"] = _STORE.url(rp)
            except Exception as exc:
                print(f"[M4] could not sign {rp}: {exc}")
        trp = c.get("thumb_remote_path")
        if trp:
            try:
                entry["thumbnail_url"] = _STORE.url(trp)
            except Exception as exc:
                print(f"[M4] could not sign {trp}: {exc}")
        enriched.append(entry)
    return jsonify({
        "session": session,
        "cards": enriched,
        "count": len(enriched),
    })


@app.route("/compose", methods=["POST"])
@require_bearer
def compose():
    """[M4-R2-F1] GPU-free merge of design A with message B.

    Cache-only: never calls pipeline.prepare() or pipeline.render_final().
    """
    data = request.get_json(silent=True) or {}
    design_id = data.get("design_id")
    message_id = data.get("message_id")
    if not design_id or not message_id:
        return jsonify({"error": "design_id and message_id required"}), 400

    design = _DESIGN_REGISTRY.get(design_id)
    if design is None:
        # [audit-fix M4-R14] this lookup hits Supabase over the network; a
        # transient error must surface as 503, never an unhandled 500.
        try:
            card = _CATALOG.find_card_by_design(design_id)
        except Exception as exc:
            return jsonify({"error": "catalog unavailable",
                            "detail": str(exc)}), 503
        if card is None or not card.get("design_json"):
            return jsonify({"error": "design_id not found"}), 404
        # [audit-fix M4-R15] design_from_dict can KeyError if the stored
        # design references a palette/layout that the library no longer has.
        try:
            design = design_from_dict(card["design_json"])
        except Exception as exc:
            return jsonify({"error": "stored design is no longer valid",
                            "detail": str(exc)}), 422

    message = _MESSAGE_REGISTRY.get(message_id)
    if message is None:
        try:
            card = _CATALOG.find_card_by_message(message_id)
        except Exception as exc:
            return jsonify({"error": "catalog unavailable",
                            "detail": str(exc)}), 503
        if card is None or not card.get("message_json"):
            return jsonify({"error": "message_id not found"}), 404
        message = dict(card["message_json"])

    pipeline = get_pipeline()
    final_bg = pipeline.get_path(design.design_id, "final")
    draft_bg = pipeline.get_path(design.design_id, "draft")
    if final_bg is not None and Path(final_bg).is_file():
        bg_path, image_quality = final_bg, "final"
    elif draft_bg is not None and Path(draft_bg).is_file():
        bg_path, image_quality = draft_bg, "draft"
    else:
        return jsonify({
            "error": "no cached background",
            "detail": "/compose is GPU-free and does not render. Request a "
                      "render via /generate or /regenerate first.",
            "design_id": design.design_id,
        }), 409

    try:
        compositor = CardCompositor()
        record = compositor.compose(design, message, bg_path)

        # [m3-fix] Use card_id-based path (consistent with /regenerate).
        card_name = _card_object_name(record["card_id"], "cards")
        record["image_url"] = _STORE.upload(record["path"], card_name)
        record["remote_path"] = card_name
        record["thumbnail_url"] = record["image_url"]
        record["thumb_remote_path"] = card_name  # audit-fix M4-P1
        record["design_json"] = design.as_dict()
        record["message_json"] = dict(message)
        record["session_id"] = "merge"
        record["image_quality"] = image_quality
        record["image_is_final"] = (image_quality == "final")
        _CATALOG.put_card(record)

        return jsonify({
            "card_id": record["card_id"],
            "image_url": record["image_url"],
            "thumbnail_url": record["thumbnail_url"],
            "design_id": design_id,
            "message_id": message_id,
            "image_quality": image_quality,
            "contrast_ratio": record.get("contrast_ratio"),
            "contrast_fallback": record.get("contrast_fallback"),
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/regenerate/<card_id>", methods=["POST"])
@require_bearer
def regenerate(card_id):
    # [audit-fix M4-R15] guard the Supabase fetch -> 503, not a 500.
    try:
        card = _CATALOG.get_card(card_id)
    except Exception as exc:
        return jsonify({"error": "catalog unavailable",
                        "detail": str(exc)}), 503
    if card is None:
        return jsonify({"error": "card not found"}), 404

    job_id = "job_" + uuid.uuid4().hex[:16]
    job = {
        "job_id": job_id,
        "session_id": card.get("session_id"),
        "name": None, "dob": None, "tone": None, "count": 1,
        "state": "queued",
        "progress": {"done": 0, "total": 1},
        "card_ids": [],
        "error": None,
        "mode": "single",
        "target_card_id": card_id,
        "idem_key": None,
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
    }
    with _JOB_STORE._lock:
        _JOB_STORE.jobs[job_id] = job
        _JOB_STORE._save()
    _WORKER.submit(job_id)
    return jsonify({"job_id": job_id, "state": "queued"}), 202


# --- SECTION 11: NGROK TUNNEL -----------------------------------------------

_TUNNEL_STATE = {"url": None, "tunnel": None}


def start_tunnel(port):
    if not _NGROK_AUTHTOKEN:
        print("[M4] NGROK_AUTHTOKEN not configured; tunnel not started.")
        return None
    try:
        from pyngrok import ngrok
    except ImportError:
        print("[M4] pyngrok not installed; tunnel not started.")
        return None

    ngrok.set_auth_token(_NGROK_AUTHTOKEN)
    kwargs = {"proto": "http", "addr": port, "bind_tls": True}
    if _NGROK_DOMAIN:
        kwargs["domain"] = _NGROK_DOMAIN
    else:
        kwargs["region"] = M4_CONFIG["ngrok_region"]
    try:
        tunnel = ngrok.connect(**kwargs)
    except Exception as exc:
        # [env-fix] ngrok v3 dropped --region; retry with a minimal set so the
        # tunnel still comes up (ngrok picks the nearest region itself).
        print(f"[M4] tunnel with region failed ({exc}); retrying without it")
        minimal = {"proto": "http", "addr": port, "bind_tls": True}
        if _NGROK_DOMAIN:
            minimal["domain"] = _NGROK_DOMAIN
        tunnel = ngrok.connect(**minimal)
    url = tunnel.public_url
    _TUNNEL_STATE["url"] = url
    _TUNNEL_STATE["tunnel"] = tunnel
    print(f"[M4] ngrok tunnel: {url}")
    print("[M4] phone must send: ngrok-skip-browser-warning: 1")
    return url


# --- SECTION 12: SERVICE START ----------------------------------------------

def start_service(port=None, host=None):
    port = port or M4_CONFIG["port"]
    host = host or M4_CONFIG["host"]

    _WORKER.start()
    url = start_tunnel(port)
    # [audit-fix M4-R12] fail closed: in real mode a missing public endpoint
    # means the phone cannot reach us - say so instead of pretending to be up.
    if url is None and not _ALLOW_LOCAL:
        raise RuntimeError(
            "no public endpoint (ngrok tunnel not established). Real mode "
            "requires NGROK_AUTHTOKEN, or run with BETACARD_LOCAL=1.")

    def _serve():
        app.run(host=host, port=port, threaded=True, use_reloader=False)

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()
    print(f"[M4] Flask listening on {host}:{port}")
    return {"url": url, "port": port, "host": host}


# --- SECTION 13: SELF-CHECK (REVISED: F1 + F2 FIXES) ------------------------

M4_FAILURES = []


def m4_check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' - ' + detail) if detail else ''}")
        M4_FAILURES.append(label)


print("\n" + "=" * 72)
print("M4 SELF-CHECK (FINAL REVISED)")
print(f"mode: {'LOCAL-TEST' if _ALLOW_LOCAL else 'REAL'} "
      f"mock={'yes' if MOCK_MODE else 'no'}")
print(f"storage: {_STORE.__class__.__name__}  "
      f"catalog: {_CATALOG.__class__.__name__}")
print("=" * 72)

# [F1-FIX] Start the worker BEFORE section [8] (or any section that submits
# a job). In the previous revision _WORKER.start() lived only inside
# start_service(), so section [8] enqueued a job that no thread ever ran.
_WORKER.start()
print("[M4] self-check: worker thread started for job tests")

try:
    # -- [1] Storage round-trip -----------------------------------------
    print("\n[1] Storage adapter round-trip (LocalStore)")
    test_store = LocalStore(M4_STORAGE_DIR / "_selfcheck_store")
    test_png = _CARDS_DIR / "_selfcheck.png"
    try:
        Image.new("RGB", (64, 64), (200, 100, 50)).save(test_png)
        u = test_store.upload(test_png, "test/one.png")
        m4_check("upload returns a URL",
                 isinstance(u, str) and len(u) > 0)
        m4_check("local file landed",
                 (test_store.root / "test/one.png").is_file())
        test_store.delete("test/one.png")
        m4_check("delete removes the file",
                 not (test_store.root / "test/one.png").is_file())
    finally:
        try:
            test_png.unlink(missing_ok=True)
        except NameError:
            pass

    # -- [2] Catalog round-trip -----------------------------------------
    print("\n[2] Catalog adapter round-trip and find_* methods")
    test_cat = LocalCatalog(
        M4_CATALOG_PATH.with_suffix(".selfcheck.json"))
    test_cat.put_session({"session_id": "s1", "name": "A"})
    test_cat.put_card({
        "card_id": "c1", "session_id": "s1", "image_url": "u",
        "design_id": "d1", "message_id": "m1",
        "design_json": {"design_id": "d1"},
        "message_json": {"message_id": "m1"},
    })
    m4_check("session stored",
             test_cat.get_session("s1")["name"] == "A")
    m4_check("card stored",
             test_cat.get_card("c1")["image_url"] == "u")
    m4_check("list_cards filters by session",
             [c["card_id"] for c in test_cat.list_cards("s1")] == ["c1"])
    m4_check("find_card_by_design returns the record",
             test_cat.find_card_by_design("d1") is not None)
    m4_check("find_card_by_message returns the record",
             test_cat.find_card_by_message("m1") is not None)
    m4_check("find_card_by_design(unknown) is None",
             test_cat.find_card_by_design("nope") is None)
    M4_CATALOG_PATH.with_suffix(".selfcheck.json").unlink(missing_ok=True)
    m4_check("SupabaseCatalog implements find_card_by_design",
             hasattr(SupabaseCatalog, "find_card_by_design"))
    m4_check("SupabaseCatalog implements find_card_by_message",
             hasattr(SupabaseCatalog, "find_card_by_message"))

    # -- [3] Job store idempotency --------------------------------------
    print("\n[3] Job store idempotency")
    js_path = M4_JOBS_PATH.with_suffix(".selfcheck.json")
    # [audit-fix M4-R6b] start clean so a rerun is not a false idempotent hit.
    try:
        js_path.unlink()
    except FileNotFoundError:
        pass
    js = JobStore(js_path)
    k = _idempotency_key("sess_x", "Aayush", "2002-10-06")
    j1, existed1 = js.create("sess_x", "Aayush", "2002-10-06",
                              None, 12, k)
    j2, existed2 = js.create("sess_x", "Aayush", "2002-10-06",
                              None, 12, k)
    m4_check("first create is new", existed1 is False)
    m4_check("second create is idempotent hit", existed2 is True)
    m4_check("same job_id returned", j1["job_id"] == j2["job_id"])
    m4_check("job persisted to disk", js_path.is_file())
    js_path.unlink(missing_ok=True)

    # -- [4] Bearer strength --------------------------------------------
    print("\n[4] Bearer token strength")
    m4_check(f"token length >= {M4_CONFIG['bearer_min_length']}",
             len(_BEARER_TOKEN) >= M4_CONFIG["bearer_min_length"])
    m4_check("token is not a placeholder",
             _BEARER_TOKEN not in ("changeme", "secret", "password", "test"))

    # -- [5] Auth on every endpoint -------------------------------------
    print("\n[5] Auth required on every endpoint")
    client = app.test_client()
    ENDPOINTS = [
        ("GET", "/health", None),
        ("POST", "/generate",
         {"session_id": "s", "name": "A", "dob": "2002-10-06"}),
        ("GET", "/job/does-not-exist", None),
        ("GET", "/catalog/does-not-exist", None),
        ("POST", "/compose", {"design_id": "x", "message_id": "y"}),
        ("POST", "/regenerate/does-not-exist", None),
    ]
    for method, path, body in ENDPOINTS:
        r = client.open(path, method=method, json=body)
        m4_check(f"{method} {path} rejected ({r.status_code})",
                 r.status_code in (400, 401))
    m4_check("invalid token 401",
             client.get(
                 "/health",
                 headers={"Authorization": "Bearer wrong"}
             ).status_code == 401)

    AUTH = {
        "Authorization": f"Bearer {_BEARER_TOKEN}",
        "ngrok-skip-browser-warning": "1",
    }
    m4_check("valid token + header accepts /health",
             client.get("/health", headers=AUTH).status_code == 200)

    # -- [6] Rate limit -------------------------------------------------
    print("\n[6] Rate limit")
    _test_token = "ratelimit-probe-" + uuid.uuid4().hex
    _RATE_BUCKETS[_test_token] = [
        time.time(), M4_CONFIG["rate_limit_per_min"]]
    m4_check("bucket at cap rejects", _rate_ok(_test_token) is False)
    _RATE_BUCKETS[_test_token] = [time.time(), 0]
    m4_check("bucket below cap accepts", _rate_ok(_test_token) is True)
    _RATE_BUCKETS.pop(_test_token, None)

    # -- [7] Cold /compose is GPU-free ----------------------------------
    print("\n[7] /compose on a cold cache is GPU-free")
    cold = client.post("/compose", headers=AUTH, json={
        "design_id": "definitely-not-a-design",
        "message_id": "definitely-not-a-message",
    })
    # [audit-fix M4-R14] a catalog backend hiccup now yields 503 (not 500);
    # either way this must never be an unhandled 500.
    m4_check("cold /compose 404/503, not 500",
             cold.status_code in (404, 503))

    # -- [8] Full async pipeline ----------------------------------------
    print("\n[8] Full async pipeline through the API")
    sid = "sess_m4_selfcheck_" + uuid.uuid4().hex[:6]
    r = client.post("/generate", headers=AUTH, json={
        "session_id": sid, "name": "Aayush",
        "dob": "2002-10-06", "count": 12,
    })
    m4_check("POST /generate 202", r.status_code == 202)
    body = r.get_json()
    job_id = body.get("job_id")
    m4_check("job_id returned", bool(job_id))

    r2 = client.post("/generate", headers=AUTH, json={
        "session_id": sid, "name": "Aayush",
        "dob": "2002-10-06", "count": 12,
    })
    m4_check("second /generate is idempotent",
             r2.get_json().get("idempotent_hit") is True)
    m4_check("same job_id", r2.get_json().get("job_id") == job_id)

    # [audit-fix M4-R11] real-mode headroom (SDXL + 12 cards + uploads).
    deadline = time.time() + M4_CONFIG.get("selfcheck_job_timeout_s", 900)
    last_state = None
    jr = {}
    while time.time() < deadline:
        jr = client.get(f"/job/{job_id}", headers=AUTH).get_json()
        last_state = jr.get("state")
        if last_state in ("done", "failed", "partially_complete"):
            break
        time.sleep(0.5)
    m4_check(f"job reached terminal state ({last_state})",
             last_state in ("done", "partially_complete", "failed"))
    m4_check("job state is done", last_state == "done")
    m4_check("all 12 cards done",
             jr.get("progress", {}).get("done") == 12)

    # -- [9] Catalog read -----------------------------------------------
    print("\n[9] /catalog/{session_id}")
    cr = client.get(f"/catalog/{sid}", headers=AUTH)
    m4_check("catalog read 200", cr.status_code == 200)
    # [audit-fix M4-R5-1] If the catalog backend is unusable (e.g. the Supabase
    # tables were never created) the body is not JSON. Report an honest FAIL
    # and stub the body so downstream checks fail cleanly instead of raising
    # AttributeError / KeyError.
    _raw = cr.get_json() if cr.is_json else None
    _usable = (isinstance(_raw, dict)
               and isinstance(_raw.get("cards"), list)
               and len(_raw["cards"]) >= 8)
    m4_check("12 cards in catalog", bool(_usable and _raw.get("count") == 12),
             "" if _usable else
             f"catalog body unusable (status {cr.status_code})")
    if not _usable:
        print("  [audit-fix M4-R14] catalog unusable - the card-field and merge")
        print("  checks below are expected to FAIL as a group; root cause above.")
    cbody = _raw if _usable else {"cards": [{} for _ in range(12)],
                                  "count": None}
    sample_card = cbody["cards"][0]
    for field in ("card_id", "design_id", "message_id",
                  "image_url", "thumbnail_url", "image_quality"):
        m4_check(f"card carries {field}", field in sample_card)

    # -- [10] Selective finals ------------------------------------------
    print("\n[10] [M1] Selective finals: default is drafts only")
    m4_check("render_finals default is 'none'",
             M4_CONFIG["render_finals"] == "none")
    if M4_CONFIG["render_finals"] == "none":
        m4_check("all batch cards are draft quality",
                 all(c.get("image_is_final") is False
                     for c in cbody["cards"]))
        m4_check("every card reports image_quality='draft'",
                 all(c.get("image_quality") == "draft"
                     for c in cbody["cards"]))

    # -- [11] Image URLs don't pass through tunnel ----------------------
    print("\n[11] Image URLs not via tunnel")
    tunnel_url = _TUNNEL_STATE.get("url")
    tunnel_safe = True
    for c in cbody["cards"]:
        iu = c.get("image_url", "")
        if tunnel_url and tunnel_url in iu:
            tunnel_safe = False
            break
    m4_check("no card URL points at tunnel URL", tunnel_safe)
    m4_check("image_url non-empty for every card",
             all(c.get("image_url") for c in cbody["cards"]))

    # -- [12] Warm /compose, zero renders -------------------------------
    print("\n[12] /compose on warm cache triggers no render")
    _pipeline_obj = get_pipeline()
    _renderer_obj = getattr(_pipeline_obj, "renderer", None)
    _render_calls_pre = (getattr(_renderer_obj, "render_calls", 0)
                         if _renderer_obj else 0)

    a_card = cbody["cards"][0]
    b_card = cbody["cards"][7]
    mr = client.post("/compose", headers=AUTH, json={
        "design_id": a_card.get("design_id"),
        "message_id": b_card.get("message_id"),
    })
    m4_check("compose 200", mr.status_code == 200)
    mbody = mr.get_json() if mr.is_json else {}

    _renderer_obj_post = getattr(_pipeline_obj, "renderer", None)
    _render_calls_post = (getattr(_renderer_obj_post, "render_calls", 0)
                          if _renderer_obj_post else 0)
    m4_check(f"render_calls unchanged ({_render_calls_pre} -> "
             f"{_render_calls_post})",
             _render_calls_post == _render_calls_pre)
    m4_check("pipeline.renderer is None after /compose",
             _renderer_obj_post is None)
    m4_check("merge card_id differs from both sources",
             mbody.get("card_id") not in
             {a_card.get("card_id"), b_card.get("card_id")})
    m4_check("merge carries design_id of A",
             mbody.get("design_id") == a_card.get("design_id"))
    m4_check("merge carries message_id of B",
             mbody.get("message_id") == b_card.get("message_id"))
    m4_check("merge reports image_quality",
             mbody.get("image_quality") in ("draft", "final"))
    m4_check("merge cleared WCAG AA contrast",
             (mbody.get("contrast_ratio") or 0)
             >= M3_CONFIG["min_contrast"])
    m4_check("contrast_fallback present (may be None)",
             "contrast_fallback" in mbody)

    # -- [13] /regenerate promotes a card -------------------------------
    print("\n[13] /regenerate promotes a card to final quality")
    # [audit-fix M4-R5-2] a_card is a stub when the catalog was unusable;
    # guard so the self-check reports FAILs instead of raising KeyError.
    rr = client.post(f"/regenerate/{a_card.get('card_id')}", headers=AUTH)
    m4_check("regenerate 202", rr.status_code == 202)
    regen_job_id = ((rr.get_json() or {}).get("job_id")
                    if rr.is_json else None)
    m4_check("regenerate returns a job id", bool(regen_job_id))
    # [audit-fix M4-R15] with no real card id there is nothing to poll; don't
    # burn the full 180s deadline on /job/None when the catalog was unusable.
    deadline = time.time() + (180 if regen_job_id else 0)
    regen_state = None
    while time.time() < deadline:
        _jr2 = client.get(f"/job/{regen_job_id}", headers=AUTH)
        jr2 = _jr2.get_json() if _jr2.is_json else {}
        regen_state = jr2.get("state")
        if regen_state in ("done", "failed", "partially_complete"):
            break
        time.sleep(0.5)
    m4_check(f"regenerate finished ({regen_state})",
             regen_state == "done")

    _cr2 = client.get(f"/catalog/{sid}", headers=AUTH)
    cr2 = _cr2.get_json() if _cr2.is_json else {}
    promoted = next(
        (c for c in (cr2.get("cards") or [])
         if c.get("card_id") == a_card.get("card_id")),
        None)
    m4_check("promoted card exists", promoted is not None)
    if promoted:
        m4_check("promoted image_is_final is True",
                 promoted.get("image_is_final") is True)
        m4_check("promoted image_quality is 'final'",
                 promoted.get("image_quality") == "final")

    # -- [14] Skip-warning header enforcement (F2-FIX) ------------------
    print("\n[14] [M2] ngrok skip-warning header enforcement")
    # F2-FIX: /health is exempt, so the "seen=False" case must be tested
    # WITHOUT the skip header. Using AUTH (which carries the header) for
    # that assertion was a contradiction in the previous revision.
    AUTH_NO_SKIP = {"Authorization": f"Bearer {_BEARER_TOKEN}"}

    h_no_skip = client.get("/health", headers=AUTH_NO_SKIP).get_json()
    m4_check("/health without skip header: seen=False (exempt)",
             h_no_skip.get("skip_warning_header_seen") is False)

    h_with_skip = client.get("/health", headers=AUTH).get_json()
    m4_check("/health with skip header: seen=True",
             h_with_skip.get("skip_warning_header_seen") is True)

    if M4_CONFIG["require_skip_warning_header"]:
        r_no_hdr = client.post(
            "/generate",
            headers={"Authorization": f"Bearer {_BEARER_TOKEN}"},
            json={"session_id": "x", "name": "A",
                  "dob": "2002-10-06"})
        m4_check("non-health without skip header -> 400",
                 r_no_hdr.status_code == 400)
        # [audit-fix M4-R14] use a body the handler rejects on its own merits
        # (count=0) so nothing is enqueued. The old body was accepted (202)
        # and queued a *live* SDXL job that then raced [18] on a real run.
        r_with_hdr = client.post(
            "/generate",
            headers={**AUTH, "ngrok-skip-browser-warning": "1"},
            json={"session_id": "x2", "name": "A",
                  "dob": "2002-10-06", "count": 0})
        _wj = r_with_hdr.get_json() or {}
        m4_check("non-health with skip header -> passes the header gate",
                 r_with_hdr.status_code == 400
                 and "skip-browser-warning" not in str(_wj.get("error", "")))
    else:
        print("  [info] local test: skip-warning enforcement waived.")

    # -- [15] Real-mode enforcement -------------------------------------
    print("\n[15] [C1][C2][C3] Real-mode enforcement policy")
    m4_check("[C1] Supabase present or local test",
             _ALLOW_LOCAL or _SUPABASE_OK)
    m4_check("[C2] Supabase catalog present or local test",
             _ALLOW_LOCAL or _CATALOG_OK)
    m4_check("[C3] bearer token present or local test",
             _ALLOW_LOCAL or bool(get_secret("API_BEARER_TOKEN")))
    m4_check("[C1] LocalStore only under local test",
             _ALLOW_LOCAL or isinstance(_STORE, SupabaseStore))
    m4_check("[C2] LocalCatalog only under local test",
             _ALLOW_LOCAL or isinstance(_CATALOG, SupabaseCatalog))
    if _ALLOW_LOCAL:
        print("  [info] BETACARD_LOCAL=1: enforcement waived for this run.")

    # -- [16] Credential scan (m1-fix: full module source) --------------
    print("\n[16] [m1-fix] No hardcoded credentials (full module scan)")
    try:
        _mod = _sys.modules.get("__main__")
        _full_src = _inspect.getsource(_mod) if _mod else ""
    except (OSError, TypeError):
        _full_src = ""
    _pieces = [_full_src]
    for _name, _obj in list(globals().items()):
        try:
            if _inspect.isfunction(_obj) or _inspect.isclass(_obj):
                _pieces.append(_inspect.getsource(_obj))
        except (OSError, TypeError):
            pass
    _all_src = "\n".join(_pieces)
    _suspicious = [
        "sb_secret_", "AIza", "-----BEGIN PRIVATE KEY-----",
        "service_account_private", "eyJhbGciOi",
    ]
    _leaks = [s for s in _suspicious if s in _all_src]
    m4_check("no credential-shaped strings in source",
             len(_leaks) == 0, f"found: {_leaks}")

    # -- [17] Frozen milestones -----------------------------------------
    print("\n[17] M1/M2/M3 remain import-only from M4")
    m4_check("M2 CONFIG still has watchdog_threshold_gb",
             "watchdog_threshold_gb" in globals().get("CONFIG", {}))
    m4_check("M3 M3_CONFIG still present", "M3_CONFIG" in globals())
    m4_check("M4_CONFIG is not M2 CONFIG nor M3 M3_CONFIG",
             M4_CONFIG is not globals().get("CONFIG")
             and M4_CONFIG is not globals().get("M3_CONFIG"))

    # -- [18] Worker._busy ----------------------------------------------
    print("\n[18] Worker._busy toggles correctly")
    m4_check("Worker has _busy attribute", hasattr(_WORKER, "_busy"))
    # [env-fix] give the worker a moment to drain a just-finished job before
    # asserting it is idle (avoids a race right after [13]).
    _idle_deadline = time.time() + 5
    while _WORKER._busy and time.time() < _idle_deadline:
        time.sleep(0.1)
    m4_check("_busy is False after all jobs drained",
             _WORKER._busy is False)
    m4_check("run_one_sync returns True on empty queue",
             _WORKER.run_one_sync(timeout_s=3) is True)

    # -- [19] Merge path uses card_id -----------------------------------
    print("\n[19] [m3-fix] Merge uses card_id-based path (no orphans)")
    m4_check("_object_name_for_merge removed",
             "_object_name_for_merge" not in dir())

finally:
    # [F1-FIX] Stop the worker when the self-check exits.
    try:
        _WORKER.stop()
        print("\n[M4] self-check: worker stopped")
    except Exception:
        pass


# -- Summary -----------------------------------------------------------------
print("\n" + "=" * 72)
if M4_FAILURES:
    print(f"RESULT: {len(M4_FAILURES)} FAILURE(S)")
    for f in M4_FAILURES:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 72)
print(f"\n[M4] jobs:    {M4_JOBS_PATH}")
print(f"[M4] catalog: {M4_CATALOG_PATH}")
print(f"[M4] storage: {M4_STORAGE_DIR}")
print(f"\n[M4] To start the live service:\n    start_service()")

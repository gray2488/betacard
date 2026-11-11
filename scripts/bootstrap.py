"""Install only the dependencies that are missing or out of range."""

# [env-fix] BOOTSTRAP - idempotent dependency install for a fresh Kaggle run.
#
# Kaggle wipes the ephemeral environment on every fresh run, so anything
# installed with pip is gone next time. This cell installs ONLY what is
# missing or out of range, so a re-run costs nothing when the environment
# already satisfies everything.
#
# Kaggle's "Internet" toggle must be ON for pip to reach PyPI.
#
# MOST IMPORTANT: Kaggle ships transformers 5.0.0, which is a known-broken
# release for diffusers consumers - it crashes on import with "Backend should
# be defined in the BACKENDS_MAPPING" and a missing
# transformers.tokenization_utils_* module. This cell pins transformers to
# 4.x. If transformers is ALREADY imported this session, restart the kernel
# afterwards for the pin to take effect.
import importlib.util as _boot_ilu
import subprocess as _boot_sp
import sys as _boot_sys
import os as _boot_os
from importlib.metadata import (version as _boot_ver,
                                PackageNotFoundError as _boot_pnf)

# (import name, pip spec, min major allowed, max major allowed or None)
_BOOT_DEPS = [
    ("diffusers",    "diffusers",             0, None),
    ("accelerate",   "accelerate",            0, None),  # device_map / cpu-offload
    ("bitsandbytes", "bitsandbytes>=0.46.1",   0, None),  # 4-bit text model
    ("safetensors",  "safetensors",           0, None),
    ("PIL",          "pillow",                0, None),
    ("requests",     "requests",              0, None),
    ("flask",        "flask",                 0, None),  # M4 API server
    ("pyngrok",      "pyngrok",               0, None),  # M4 ngrok tunnel
]
_TF_SPEC = "transformers>=4.44,<5"        # transformers 5.x is broken here


def _boot_status(mod, lo, hi):
    if _boot_ilu.find_spec(mod) is None:
        return "missing"
    if lo or hi:
        try:
            major = int(_boot_ver(mod).split(".")[0])
        except (_boot_pnf, ValueError):
            return "ok"
        if lo and major < lo:
            return "too old (%s)" % _boot_ver(mod)
        if hi and major > hi:
            return "too new (%s)" % _boot_ver(mod)
    return "ok"


def _boot_tf_major():
    try:
        return int(_boot_ver("transformers").split(".")[0])
    except (_boot_pnf, ValueError):
        return 0


def _boot_install(specs, no_deps=False):
    if isinstance(specs, str):
        specs = [specs]
    cmd = [_boot_sys.executable, "-m", "pip", "install", "-q", "--no-input"]
    # [env-fix] install offline from the betacard-wheels dataset when attached
    _wheels = "/kaggle/input/betacard-wheels"
    if _boot_os.path.isdir(_wheels):
        cmd += ["--no-index", "--find-links", _wheels]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(specs)
    _boot_sp.run(cmd, check=False)


# 1) transformers must be 4.x (5.x is a known-broken release for diffusers)
if _boot_tf_major() != 4:
    if "transformers" in _boot_sys.modules:
        print("[bootstrap] WARNING: transformers is already imported this "
              "session; after the downgrade you MUST restart the kernel "
              "(Run > Restart) for it to take effect.")
    print("[bootstrap] pinning transformers to 4.x ...")
    _boot_install(_TF_SPEC)
    if _boot_tf_major() != 4:
        print("[bootstrap] resolver kept a non-4.x transformers; retrying "
              "with --no-deps")
        _boot_install(_TF_SPEC, no_deps=True)
    try:
        print("[bootstrap] transformers now:", _boot_ver("transformers"))
    except _boot_pnf:
        pass

# 2) everything else: install only what is missing
_boot_todo = [spec for mod, spec, lo, hi in _BOOT_DEPS
              if _boot_status(mod, lo, hi) != "ok"]
if _boot_todo:
    print("[bootstrap] installing:", ", ".join(_boot_todo))
    _boot_install(_boot_todo)
else:
    print("[bootstrap] all other runtime deps satisfied")

_boot_left = ["%s:%s" % (m, _boot_status(m, l, h))
              for m, s, l, h in _BOOT_DEPS
              if _boot_status(m, l, h) != "ok"]
# [env-fix] if the offline wheels set was incomplete, retry from PyPI
if _boot_left and _boot_os.path.isdir("/kaggle/input/betacard-wheels"):
    print("[bootstrap] offline set incomplete; retrying online for:", _boot_left)
    _boot_sp.run([_boot_sys.executable, "-m", "pip", "install", "-q",
                  "--no-input"]
                 + [s for m, s, l, h in _BOOT_DEPS
                    if _boot_status(m, l, h) != "ok"], check=False)
print("[bootstrap] still unsatisfied:",
      ", ".join(_boot_left) if _boot_left else "none")

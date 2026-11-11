"""Check that model inputs and secrets are present before a run."""

# [env-fix] PREFLIGHT - confirm the inputs are in place before Run All.
# Informational only; it never blocks the notebook.
import os as _pf_os

_DATASETS = [
    ("VAE   (M2)",  "/kaggle/input/betacard-vae"),
    ("Qwen  (M3)",  "/kaggle/input/betacard-qwen"),
    ("Wheels(env)", "/kaggle/input/betacard-wheels"),
    ("Fonts (M3)",  "/kaggle/input/betacard-assets"),
]
# SDXL is intentionally NOT a dataset input: its dataset holds only 609 B,
# so M2 downloads SDXL from the Hub into the 20 GiB working dir (~6.5 GiB).
_SECRETS = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
            "NGROK_AUTHTOKEN", "API_BEARER_TOKEN"]


def _pf_secret(name):
    if _pf_os.environ.get(name):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        return bool(UserSecretsClient().get_secret(name))
    except Exception:
        return False


print("PREFLIGHT")
_ready = True
for _label, _path in _DATASETS:
    _got = _pf_os.path.isdir(_path)
    print("  %s dataset %-11s %s" % ("ok     " if _got else "MISSING",
                                     _label, _path))
    _ready = _ready and _got
for _name in _SECRETS:
    _got = _pf_secret(_name)
    print("  %s secret  %s" % ("ok     " if _got else "MISSING", _name))
    _ready = _ready and _got
print("PREFLIGHT: READY" if _ready else
      "PREFLIGHT: NOT READY - attach the missing inputs / add the secrets "
      "(a local mock run can ignore this)")

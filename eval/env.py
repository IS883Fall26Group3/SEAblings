"""Eval scripts run outside Streamlit, so copy .streamlit/secrets.toml into the environment, as
streamlit_app.load_secrets_into_env() does for the app. Variables already set in the shell win."""
import os
import tomllib
from pathlib import Path

SECRETS = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"

if SECRETS.exists():
    for key, value in tomllib.loads(SECRETS.read_text(encoding="utf-8")).items():
        if isinstance(value, str) and value.strip() and key not in os.environ:
            os.environ[key] = value.strip()

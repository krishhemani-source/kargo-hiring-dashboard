import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def _bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# Tried in order when the primary is overloaded or out of daily quota (free tier ~20 requests/day/model).
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3-flash-preview,gemini-3.8-flash,gemini-3.1-flash-lite,gemini-3.5-flash-lite").split(",") if m.strip()]
GEMINI_MODELS = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "").strip() or "Kargo Hiring <onboarding@resend.dev>"
FOUNDER_EMAIL = os.getenv("FOUNDER_EMAIL", "").strip()
FOUNDER_NAME = os.getenv("FOUNDER_NAME", "Arjun")
TEST_MODE = _bool("TEST_MODE", "true")
LLM_WORKERS = int(os.getenv("LLM_WORKERS", "2"))

# Neon/Postgres connection string (the Vercel Neon integration sets DATABASE_URL). Empty -> local SQLite.
DATABASE_URL = (os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "").strip()
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "kargo.db"))

ON_VERCEL = bool(os.getenv("VERCEL"))
# Login for the dashboard. Required on Vercel; optional locally.
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
# Each /process request must finish inside Vercel's function limit (maxDuration in vercel.json).
LLM_CALL_BUDGET_S = int(os.getenv("LLM_CALL_BUDGET_S", "85"))
RUBRIC_PATH = ROOT / "rubric.txt"
JD_DIR = ROOT / "jds"

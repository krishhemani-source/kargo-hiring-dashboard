"""Storage: Neon/Postgres when DATABASE_URL is set (Vercel), SQLite file otherwise (local).

All SQL is written with `?` placeholders; they're translated for Postgres.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from . import config

COLUMNS = """
  created_at TEXT NOT NULL,
  filename TEXT NOT NULL,
  file_data {blob},
  sha256 TEXT UNIQUE,
  role_applied TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'new',   -- new -> scored -> invite_ready/reject_ready -> sent  (or failed)
  stage TEXT,                           -- while new: queued / extracting / scoring / drafting / retry
  claimed_at TEXT,
  error TEXT,
  -- personal details: stored here, never sent to the LLM
  name TEXT, email TEXT, phone TEXT, links TEXT, city TEXT,
  warnings TEXT,
  redacted_text TEXT,
  relocation TEXT, relocation_note TEXT,
  pm_score REAL, spm_score REAL,
  pm_detail TEXT, spm_detail TEXT,
  brief TEXT,
  invite_subject TEXT, invite_body TEXT,
  reject_subject TEXT, reject_body TEXT,
  model TEXT, scored_at TEXT,
  decision TEXT, decision_note TEXT, decided_at TEXT,
  sent_to TEXT, sent_subject TEXT, sent_at TEXT, resend_id TEXT, send_error TEXT
"""

SCHEMA = {
    "sqlite": [
        "CREATE TABLE IF NOT EXISTS candidates (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        + COLUMNS.format(blob="BLOB") + ")",
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER NOT NULL,"
        " ts TEXT NOT NULL, action TEXT NOT NULL, detail TEXT)",
    ],
    "postgres": [
        "CREATE TABLE IF NOT EXISTS candidates (id SERIAL PRIMARY KEY," + COLUMNS.format(blob="BYTEA") + ")",
        "CREATE TABLE IF NOT EXISTS events (id SERIAL PRIMARY KEY, candidate_id INTEGER NOT NULL,"
        " ts TEXT NOT NULL, action TEXT NOT NULL, detail TEXT)",
    ],
}

JSON_COLS = {"links", "warnings", "pm_detail", "spm_detail", "brief"}
# Everything except the file bytes, for normal reads.
META_COLS = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ago(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")


def dialect() -> str:
    return "postgres" if config.DATABASE_URL else "sqlite"


@contextmanager
def connect():
    """Yields a cursor-like `run(sql, params)` that returns a list of dict rows. Commits on exit."""
    if dialect() == "postgres":
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(config.DATABASE_URL, row_factory=dict_row, connect_timeout=15) as conn:
            def run(sql, params=()):
                cur = conn.execute(sql.replace("?", "%s"), params)
                return cur.fetchall() if cur.description else []
            yield run
    else:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            def run(sql, params=()):
                cur = conn.execute(sql, params)
                return [dict(r) for r in cur.fetchall()] if cur.description else []
            yield run
            conn.commit()
        finally:
            conn.close()


def init() -> None:
    with connect() as run:
        for stmt in SCHEMA[dialect()]:
            run(stmt)


def _meta_cols() -> str:
    global META_COLS
    if META_COLS is None:
        body = "\n".join(l.split("--")[0] for l in COLUMNS.splitlines())
        names = [part.split()[0] for part in body.replace("\n", ",").split(",") if part.strip()]
        META_COLS = "id, " + ", ".join(n for n in names if n != "file_data")
    return META_COLS


def row_to_dict(row: dict) -> dict:
    d = dict(row)
    for k in JSON_COLS:
        if d.get(k):
            d[k] = json.loads(d[k])
    return d


def get(cid: int):
    with connect() as run:
        rows = run(f"SELECT {_meta_cols()} FROM candidates WHERE id=?", (cid,))
    return row_to_dict(rows[0]) if rows else None


def get_file(cid: int):
    with connect() as run:
        rows = run("SELECT filename, file_data FROM candidates WHERE id=?", (cid,))
    if not rows:
        return None
    return rows[0]["filename"], bytes(rows[0]["file_data"] or b"")


def list_all(cols: str) -> list:
    with connect() as run:
        return [row_to_dict(r) for r in run(f"SELECT {cols} FROM candidates")]


def find_by_sha(sha: str):
    with connect() as run:
        rows = run("SELECT id, name FROM candidates WHERE sha256=?", (sha,))
    return rows[0] if rows else None


def insert_candidate(filename: str, data: bytes, sha: str, role: str) -> int:
    with connect() as run:
        rows = run(
            "INSERT INTO candidates (created_at, filename, file_data, sha256, role_applied, status, stage) "
            "VALUES (?,?,?,?,?, 'new', 'queued') RETURNING id",
            (now(), filename, data, sha, role),
        )
    return rows[0]["id"]


RETRY_AFTER_S = 60  # wait this long before re-trying a CV that hit an overloaded Gemini


def claim(cid: int, stale_after: int = 330, retry_after: int = RETRY_AFTER_S) -> bool:
    """Atomically take a queued (or abandoned, or due-for-retry) candidate. False if someone else has it."""
    with connect() as run:
        rows = run(
            "UPDATE candidates SET stage='extracting', claimed_at=? "
            "WHERE id=? AND status='new' AND (stage='queued' OR claimed_at IS NULL OR claimed_at < ? "
            "OR (stage='retry' AND claimed_at < ?)) RETURNING id",
            (now(), cid, ago(stale_after), ago(retry_after)),
        )
    return bool(rows)


def update(cid: int, **fields) -> None:
    if not fields:
        return
    vals = [json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()]
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as run:
        run(f"UPDATE candidates SET {sets} WHERE id=?", (*vals, cid))


def log(cid: int, action: str, detail: str = "") -> None:
    with connect() as run:
        run("INSERT INTO events (candidate_id, ts, action, detail) VALUES (?,?,?,?)", (cid, now(), action, detail))


def events(cid: int) -> list:
    with connect() as run:
        return run("SELECT ts, action, detail FROM events WHERE candidate_id=? ORDER BY id", (cid,))

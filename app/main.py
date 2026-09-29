import hashlib
import hmac
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import config, db, emailer, pipeline
from .rubric import ROLES, load_rubrics

app = FastAPI(title="Kargo Hiring Dashboard")
STATIC = config.ROOT / "static"
COOKIE = "kargo_session"
SESSION_DAYS = 14
_ready = False


def ensure_ready():
    """Create tables once per process (cold start), lazily so a bad DATABASE_URL shows as an error, not a crash."""
    global _ready
    if not _ready:
        db.init()
        load_rubrics()  # fail fast if rubric.txt is malformed
        _ready = True


# ---------------- auth ----------------

def _secret() -> bytes:
    base = config.SESSION_SECRET or ("kargo:" + config.DASHBOARD_PASSWORD)
    return hashlib.sha256(base.encode()).digest()


def _make_token() -> str:
    exp = str(int(time.time()) + SESSION_DAYS * 86400)
    sig = hmac.new(_secret(), exp.encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _valid(token: str) -> bool:
    try:
        exp, sig = token.split(".", 1)
        good = hmac.new(_secret(), exp.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, good) and int(exp) > time.time()
    except (ValueError, AttributeError):
        return False


OPEN_PATHS = {"/api/login", "/api/logout", "/api/session"}


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    # Login is optional: only enforced when DASHBOARD_PASSWORD is set.
    if path.startswith("/api/") and path not in OPEN_PATHS and config.DASHBOARD_PASSWORD:
        if not _valid(request.cookies.get(COOKIE, "")):
            return JSONResponse({"detail": "login required"}, 401)
    return await call_next(request)


class Login(BaseModel):
    password: str


@app.get("/api/session")
def session(request: Request):
    needs = bool(config.DASHBOARD_PASSWORD)
    ok = not needs or _valid(request.cookies.get(COOKIE, ""))
    return {"login_required": needs, "logged_in": ok, "password_set": bool(config.DASHBOARD_PASSWORD)}


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    if not config.DASHBOARD_PASSWORD:
        raise HTTPException(503, "DASHBOARD_PASSWORD is not set")
    if not hmac.compare_digest(body.password.encode(), config.DASHBOARD_PASSWORD.encode()):
        time.sleep(1)  # slow down guessing
        raise HTTPException(401, "Wrong password")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, _make_token(), max_age=SESSION_DAYS * 86400, httponly=True,
                        samesite="strict", secure=secure)
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


# ---------------- pages + config ----------------

@app.get("/")
def index():
    # no-store: after a deploy, browsers must not keep running an old copy of the page
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def get_config():
    ensure_ready()
    r = load_rubrics()
    return {
        "test_mode": config.TEST_MODE,
        "founder_email": config.FOUNDER_EMAIL,
        "from_email": config.FROM_EMAIL,
        "model": config.GEMINI_MODEL,
        "llm_ready": bool(os.getenv("GEMINI_API_KEY")),
        "resend_ready": bool(config.RESEND_API_KEY),
        "database": db.dialect(),
        "rubrics": {role: [{"id": c.id, "name": c.name, "weight": c.weight} for c in r[role].criteria] for role in ROLES},
    }


# ---------------- candidates ----------------

MAX_FILE_BYTES = 4 * 1024 * 1024  # Vercel request bodies top out at 4.5 MB


@app.post("/api/upload")
async def upload(role: str = Form(...), files: list[UploadFile] = File(...)):
    ensure_ready()
    if role not in ROLES:
        raise HTTPException(400, "role must be PM or SPM")
    results = []
    for f in files:
        name = f.filename or "cv"
        try:
            if Path(name).suffix.lower() not in (".pdf", ".docx"):
                raise ValueError("Only PDF and DOCX are supported")
            data = await f.read()
            if not data:
                raise ValueError("Empty file")
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("File is larger than 4 MB")
            results.append({"filename": name, **pipeline.create_candidate(name, data, role)})
        except Exception as e:  # one bad file doesn't stop the batch
            results.append({"filename": name, "error": str(e)})
    return {"results": results}


LIST_COLS = ("id, created_at, filename, role_applied, status, stage, claimed_at, error, name, email, city, "
             "warnings, relocation, pm_score, spm_score, decision, sent_at, send_error")


@app.get("/api/candidates")
def list_candidates():
    ensure_ready()
    rows = db.list_all(LIST_COLS)
    counts = {
        "total": len(rows),
        "awaiting_reply": sum(1 for r in rows if r["status"] != "sent"),
        "processing": sum(1 for r in rows if r["status"] == "new"),
        "failed": sum(1 for r in rows if r["status"] == "failed"),
        "sent": sum(1 for r in rows if r["status"] == "sent"),
    }
    return {"candidates": rows, "counts": counts}


def _must(cid: int) -> dict:
    ensure_ready()
    cand = db.get(cid)
    if not cand:
        raise HTTPException(404, "candidate not found")
    return cand


@app.get("/api/candidates/{cid}")
def get_candidate(cid: int):
    cand = _must(cid)
    cand["events"] = db.events(cid)
    return cand


@app.get("/api/candidates/{cid}/file")
def get_file(cid: int):
    _must(cid)
    filename, data = db.get_file(cid)
    kind = "application/pdf" if filename.lower().endswith(".pdf") else \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return Response(data, media_type=kind, headers={"Content-Disposition": f'inline; filename="{filename}"'})


@app.post("/api/candidates/{cid}/process")
def process(cid: int):
    """Extract, redact, score and draft one queued candidate. The dashboard calls this per CV."""
    cand = _must(cid)
    if cand["status"] != "new":
        return get_candidate(cid)
    if not db.claim(cid):
        raise HTTPException(409, "already being processed")
    pipeline.process(cid)
    return get_candidate(cid)


class Decision(BaseModel):
    decision: str  # "invite" | "reject"
    note: str = ""


@app.post("/api/candidates/{cid}/decision")
def decide(cid: int, body: Decision):
    cand = _must(cid)
    if body.decision not in ("invite", "reject"):
        raise HTTPException(400, "decision must be invite or reject")
    if cand["status"] in ("new", "failed"):
        raise HTTPException(409, "candidate has not been scored yet")
    if cand["status"] == "sent":
        raise HTTPException(409, "an email has already been sent to this candidate")
    note = body.note.strip()[:300]
    status = "invite_ready" if body.decision == "invite" else "reject_ready"
    db.update(cid, decision=body.decision, decision_note=note, decided_at=db.now(), status=status)
    db.log(cid, "decision", f"{'Move forward' if body.decision == 'invite' else 'Pass'}" + (f": {note}" if note else ""))
    return get_candidate(cid)


class Draft(BaseModel):
    kind: str  # invite | reject
    subject: str
    body: str


@app.put("/api/candidates/{cid}/draft")
def save_draft(cid: int, d: Draft):
    cand = _must(cid)
    if d.kind not in ("invite", "reject"):
        raise HTTPException(400, "kind must be invite or reject")
    if cand["status"] == "sent":
        raise HTTPException(409, "already sent")
    db.update(cid, **{f"{d.kind}_subject": d.subject, f"{d.kind}_body": d.body})
    db.log(cid, "draft_edited", d.kind)
    return {"ok": True}


@app.post("/api/candidates/{cid}/send")
def send(cid: int):
    """The only path that sends email. Triggered by the founder's Send click."""
    cand = _must(cid)
    if cand["status"] == "sent":
        raise HTTPException(409, "already sent")
    if cand["status"] not in ("invite_ready", "reject_ready"):
        raise HTTPException(409, "choose Move forward or Pass first")
    kind = cand["decision"]
    subject, body = cand[f"{kind}_subject"], cand[f"{kind}_body"]
    if not subject or not body:
        raise HTTPException(409, "the draft is empty")
    try:
        res = emailer.send(cand["email"], subject, body)
    except emailer.SendError as e:
        db.update(cid, send_error=str(e))
        db.log(cid, "send_failed", str(e))
        raise HTTPException(502, str(e))
    db.update(cid, status="sent", sent_at=db.now(), resend_id=res["id"], sent_to=res["to"],
              sent_subject=res["subject"], send_error=None)
    db.log(cid, "sent", f"{kind} → {res['to']} (Resend id {res['id']})")
    return get_candidate(cid)


@app.post("/api/candidates/{cid}/retry")
def retry(cid: int):
    cand = _must(cid)
    if cand["status"] == "sent":
        raise HTTPException(409, "already sent")
    db.update(cid, status="new", stage="queued", claimed_at=None, error=None)
    db.log(cid, "retry", "")
    return {"ok": True}


@app.post("/api/candidates/{cid}/redraft")
def redraft(cid: int):
    cand = _must(cid)
    if cand["status"] in ("new", "failed", "sent"):
        raise HTTPException(409, "can only redraft a scored, unsent candidate")
    try:
        pipeline.generate_drafts(cid)
    except Exception as e:
        db.update(cid, stage=None)
        raise HTTPException(502, f"Drafting failed: {e}")
    return get_candidate(cid)

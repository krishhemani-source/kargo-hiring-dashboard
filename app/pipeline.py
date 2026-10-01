"""Steps 2-4 for one candidate: extract + redact -> score both rubrics -> brief + drafts.

Runs inside a request (POST /api/candidates/{id}/process) so it works on serverless
hosts like Vercel, where background threads are killed once the response is sent.
"""
import hashlib
import traceback
from pathlib import Path

from . import config, db, llm, templates
from .parsing import PIILeak, extract_details, extract_text, redact, relocation_gate
from .rubric import load_rubrics, weighted_percent


def create_candidate(filename: str, data: bytes, role: str) -> dict:
    """Step 1: store the original file and a `new` record (stage=queued). Returns {id} or {duplicate_of}."""
    sha = hashlib.sha256(data).hexdigest()
    dup = db.find_by_sha(sha)
    if dup:
        return {"duplicate_of": dup["id"], "name": dup["name"]}
    cid = db.insert_candidate(Path(filename).name, data, sha, role)
    db.log(cid, "uploaded", f"{filename} for {role}")
    return {"id": cid}


def score_both(redacted: str, pii: dict) -> dict:
    """Score against both rubrics; weighted totals are computed here, not by the LLM."""
    out = {}
    for role, rubric in load_rubrics().items():
        res = llm.score_cv(rubric, redacted, pii)
        pct = weighted_percent(rubric, {c["id"]: c["score"] for c in res["criteria"]})
        names = {c.id: (c.name, c.weight) for c in rubric.criteria}
        for c in res["criteria"]:
            c["name"], c["weight"] = names[c["id"]]
        out[role] = {"pct": pct, "criteria": res["criteria"]}
    return out


MAX_BUSY_RETRIES = 4


def fill_name(text: str, first_name: str) -> str:
    return (text or "").replace("[NAME]", first_name or "there")


def process(cid: int) -> None:
    """Caller must have claimed the candidate (db.claim)."""
    try:
        filename, data = db.get_file(cid)
        text = extract_text(filename, data)
        if len(text.strip()) < 200:
            raise ValueError("Very little text could be extracted (scanned image PDF?). Paste-in or OCR needed.")
        ex = extract_details(filename, text)
        redacted = redact(ex, filename)
        gate, gate_note = relocation_gate(ex)
        pii = {"name": ex.name, "email": ex.email, "phone": ex.phone, "links": ex.links}
        db.update(cid, name=ex.name, email=ex.email, phone=ex.phone, links=ex.links, city=ex.city,
                  warnings=ex.warnings, redacted_text=redacted, relocation=gate, relocation_note=gate_note)
        db.log(cid, "extracted", "; ".join(ex.warnings) or "all details found")

        db.update(cid, stage="scoring")
        llm.reset_models_used()
        scores = score_both(redacted, pii)
        db.update(cid, pm_score=scores["PM"]["pct"], spm_score=scores["SPM"]["pct"],
                  pm_detail=scores["PM"], spm_detail=scores["SPM"],
                  model=", ".join(llm.models_used()) or config.GEMINI_MODEL, scored_at=db.now())
        db.log(cid, "scored", f"PM {scores['PM']['pct']}% / SPM {scores['SPM']['pct']}%")

        # If Gemini can't write the drafts, empty ones are filled from templates.
        try:
            ai = generate_drafts(cid)
            db.update(cid, status="scored", stage=None, **({"error": None} if ai else {}))
        except Exception as e:  # re-score with drafts already in place: keep them and the new scores
            db.update(cid, status="scored", stage=None,
                      error=f"Re-scored, but new drafts couldn't be written ({str(e)[:160]}). The earlier drafts were kept.")
    except llm.LLMBusy as e:
        # Gemini overloaded (503) / rate-limited on every model: requeue instead of failing.
        tries = 1
        for ev in reversed(db.events(cid)):  # count busy attempts since the last manual Re-score
            if ev["action"] == "retry":
                break
            tries += ev["action"] == "llm_busy"
        db.log(cid, "llm_busy", f"attempt {tries}: {e}")
        if tries < MAX_BUSY_RETRIES:
            db.update(cid, status="new", stage="retry", claimed_at=db.now(),
                      error=f"Gemini is overloaded. Retrying automatically in about a minute (attempt {tries} of {MAX_BUSY_RETRIES}).")
        else:
            db.update(cid, status="failed", stage=None,
                      error=f"Gemini stayed overloaded after {tries} attempts. Click Re-score to try again later. ({e})")
    except PIILeak as e:
        db.update(cid, status="failed", stage=None, error=str(e))
        db.log(cid, "blocked", str(e))
    except Exception as e:  # one bad CV must not stop the batch
        traceback.print_exc()
        db.update(cid, status="failed", stage=None, error=f"{type(e).__name__}: {e}")
        db.log(cid, "failed", str(e))


def generate_drafts(cid: int, fallback: bool = True) -> bool:
    """Write the brief + both drafts with Gemini. Returns True on success.

    If Gemini fails and `fallback` is on, any empty draft is filled from the standard template
    (drafts that already exist, e.g. the founder's edits, are never overwritten) and False is returned.
    """
    cand = db.get(cid)
    db.update(cid, stage="drafting")
    role = cand["role_applied"]
    other = "SPM" if role == "PM" else "PM"
    detail = cand[f"{role.lower()}_detail"]
    pii = {"name": cand["name"], "email": cand["email"], "phone": cand["phone"], "links": cand["links"] or []}
    try:
        d = llm.draft_brief_and_emails(
            role, cand["redacted_text"], detail, load_rubrics()[role], detail["pct"],
            other, cand[f"{other.lower()}_score"], pii,
        )
    except Exception as e:
        traceback.print_exc()
        db.log(cid, "draft_failed", str(e))
        empty = [k for k in templates.KINDS if not cand.get(f"{k}_body")]
        if not fallback or not empty:
            raise
        fields = {}
        for k in empty:
            t = templates.render(templates.default_for(k, cand), cand)
            fields.update({f"{k}_subject": t["subject"], f"{k}_body": t["body"]})
        db.update(cid, error=f"Gemini couldn't write tailored drafts ({str(e)[:160]}), so the "
                             f"{' and '.join('invite' if k == 'invite' else 'rejection' for k in empty)} "
                             "draft was filled from a standard template. Edit it, pick another template, "
                             "or click 'Regenerate with AI' later.", **fields)
        db.log(cid, "drafted_from_template", ", ".join(empty))
        return False
    first = (cand["name"] or "").split(" ")[0]
    brief = {"who": fill_name(d["who"], first), "why_ranked": fill_name(d["why_ranked"], first),
             "probes": [fill_name(p, first) for p in d["probes"]]}
    db.update(cid, brief=brief,
              invite_subject=fill_name(d["invite"]["subject"], first), invite_body=fill_name(d["invite"]["body"], first),
              reject_subject=fill_name(d["reject"]["subject"], first), reject_body=fill_name(d["reject"]["body"], first),
              stage=None if cand["status"] != "new" else "drafting")
    db.log(cid, "drafted", "brief + invite + rejection drafts written")
    return True

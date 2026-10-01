"""End-to-end flow with the LLM and Resend stubbed: upload -> score -> decide -> edit -> send."""
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DASHBOARD_PASSWORD", "")
    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "FOUNDER_EMAIL", "founder@example.com")
    monkeypatch.setattr(config, "TEST_MODE", True)

    from app import llm
    prompts = []

    def fake_call(system, user, pii):
        from app.parsing import assert_no_pii
        assert_no_pii(system + user, pii, cv_part=user)
        prompts.append(user)
        if "Score this CV" in user:
            n = 6 if "(SPM) rubric" in system else 5
            return json.dumps({"criteria": [{"id": i, "score": 1 + i % 5, "evidence": "ev", "reason": "r"} for i in range(1, n + 1)]})
        return json.dumps({"who": "[NAME] is an ops person.", "why_ranked": "Strong ops.", "probes": ["a", "b", "c"],
                           "invite": {"subject": "Kargo x [NAME]", "body": "Hi [NAME], let's talk."},
                           "reject": {"subject": "Your application", "body": "Hi [NAME], thank you."}})

    monkeypatch.setattr(llm, "_call", fake_call)
    sent = []
    import resend
    monkeypatch.setattr(resend.Emails, "send", lambda p: sent.append(p) or {"id": "msg_123"})

    from fastapi.testclient import TestClient
    from app import main
    from app.main import app
    monkeypatch.setattr(main, "_ready", False)
    with TestClient(app) as c:
        c.sent, c.prompts = sent, prompts
        yield c


def wait_scored(c, cid):
    """The dashboard drives scoring by calling /process for each queued CV."""
    assert c.get(f"/api/candidates/{cid}").json()["stage"] == "queued"
    return c.post(f"/api/candidates/{cid}/process").json()


def test_full_flow(client):
    good = (ROOT / "seed" / "cv_07_lavanya_iyer.docx").read_bytes()
    r = client.post("/api/upload", data={"role": "PM"}, files=[
        ("files", ("cv_07_lavanya_iyer.docx", good)),
        ("files", ("broken.pdf", b"not a pdf")),
        ("files", ("notes.txt", b"hello")),
    ]).json()["results"]
    assert "id" in r[0] and "id" in r[1] and "error" in r[2]  # batch continues past bad files

    d = wait_scored(client, r[0]["id"])
    assert d["status"] == "scored", d["error"]
    assert d["name"] == "Lavanya Iyer" and d["email"] == "lavanya.iyer.pm@gmail.com"
    # backend-computed weighted totals
    assert d["pm_score"] == round(sum((1 + i % 5) * w for i, w in zip(range(1, 6), [30, 20, 20, 15, 15])) / 5, 1)
    assert d["invite_body"] == "Hi Lavanya, let's talk."
    assert not any("Lavanya" in p or "lavanya" in p for p in client.prompts)

    bad = wait_scored(client, r[1]["id"])
    assert bad["status"] == "failed"

    # Nothing sends without a decision
    assert client.post(f"/api/candidates/{r[0]['id']}/send").status_code == 409
    assert client.sent == []

    client.post(f"/api/candidates/{r[0]['id']}/decision", json={"decision": "invite", "note": "strong ops"})
    client.put(f"/api/candidates/{r[0]['id']}/draft", json={"kind": "invite", "subject": "Edited", "body": "Edited body"})
    d = client.post(f"/api/candidates/{r[0]['id']}/send").json()
    assert d["status"] == "sent" and d["resend_id"] == "msg_123"
    assert client.sent[0]["to"] == ["founder@example.com"]
    assert client.sent[0]["subject"] == "[TEST → lavanya.iyer.pm@gmail.com] Edited"
    assert client.sent[0]["text"] == "Edited body"
    assert any(e["action"] == "decision" and "strong ops" in e["detail"] for e in d["events"])

    # No double send; duplicate upload detected
    assert client.post(f"/api/candidates/{r[0]['id']}/send").status_code == 409
    again = client.post("/api/upload", data={"role": "SPM"}, files=[("files", ("x.docx", good))]).json()["results"][0]
    assert again.get("duplicate_of") == r[0]["id"]

    counts = client.get("/api/candidates").json()["counts"]
    assert counts["sent"] == 1 and counts["awaiting_reply"] == 1


def test_process_is_idempotent(client):
    good = (ROOT / "seed" / "cv_03_vikram_nair.docx").read_bytes()
    cid = client.post("/api/upload", data={"role": "SPM"}, files=[("files", ("v.docx", good))]).json()["results"][0]["id"]
    from app import db
    assert db.claim(cid)            # someone else is processing it
    assert client.post(f"/api/candidates/{cid}/process").status_code == 409
    db.update(cid, stage="queued", claimed_at=None)
    assert client.post(f"/api/candidates/{cid}/process").json()["status"] == "scored"
    n = len(client.prompts)
    assert client.post(f"/api/candidates/{cid}/process").json()["status"] == "scored"  # no rescoring
    assert len(client.prompts) == n
    f = client.get(f"/api/candidates/{cid}/file")
    assert f.status_code == 200 and f.content == good


def test_password_gate(client, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "DASHBOARD_PASSWORD", "s3cret")
    assert client.get("/api/candidates").status_code == 401
    assert client.get("/api/session").json() == {"login_required": True, "logged_in": False, "password_set": True}
    assert client.post("/api/login", json={"password": "nope"}).status_code == 401
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert client.get("/api/candidates").status_code == 200
    client.post("/api/logout")
    assert client.get("/api/candidates").status_code == 401


def test_no_password_means_open(client, monkeypatch):
    from app import config
    assert client.get("/api/candidates").status_code == 200
    assert client.get("/api/session").json()["login_required"] is False


def test_vercel_without_database_explains(client, monkeypatch):
    from app import config, main
    monkeypatch.setattr(config, "ON_VERCEL", True)
    monkeypatch.setattr(main, "_ready", False)
    r = client.get("/api/candidates")
    assert r.status_code == 503 and "Neon" in r.json()["detail"]


def test_draft_failure_falls_back_to_templates(client, monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, "draft_brief_and_emails", lambda *a, **k: (_ for _ in ()).throw(llm.LLMBusy("503 overloaded")))
    good = (ROOT / "seed" / "cv_06_meghna_tiwari.docx").read_bytes()
    cid = client.post("/api/upload", data={"role": "PM"}, files=[("files", ("m.docx", good))]).json()["results"][0]["id"]
    d = client.post(f"/api/candidates/{cid}/process").json()
    assert d["status"] == "scored" and d["pm_score"] is not None
    assert "template" in d["error"]
    assert d["invite_body"].startswith("Hi Meghna,") and "Product Manager" in d["invite_body"]
    assert d["reject_body"].startswith("Hi Meghna,")
    # A failed regenerate keeps the drafts that are there (e.g. the founder's edits)
    client.put(f"/api/candidates/{cid}/draft", json={"kind": "invite", "subject": "Mine", "body": "My edit"})
    r = client.post(f"/api/candidates/{cid}/redraft")
    assert r.status_code == 502
    assert client.get(f"/api/candidates/{cid}").json()["invite_body"] == "My edit"
    monkeypatch.undo()


def test_overloaded_gemini_requeues_then_fails(client, monkeypatch):
    from app import db, llm
    monkeypatch.setattr(llm, "score_cv", lambda *a, **k: (_ for _ in ()).throw(llm.LLMBusy("all models 503")))
    good = (ROOT / "seed" / "cv_02_sunita_krishnamurthy.docx").read_bytes()
    cid = client.post("/api/upload", data={"role": "SPM"}, files=[("files", ("s.docx", good))]).json()["results"][0]["id"]
    d = client.post(f"/api/candidates/{cid}/process").json()
    assert d["status"] == "new" and d["stage"] == "retry" and "Retrying" in d["error"]
    assert not db.claim(cid)  # not due yet
    for _ in range(3):
        assert db.claim(cid, retry_after=-5)
        from app import pipeline
        pipeline.process(cid)
    d = client.get(f"/api/candidates/{cid}").json()
    assert d["status"] == "failed" and "overloaded" in d["error"]
    # A manual Re-score starts the count again
    client.post(f"/api/candidates/{cid}/retry")
    d = client.post(f"/api/candidates/{cid}/process").json()
    assert d["stage"] == "retry"


def test_templates_endpoint_and_apply(client):
    cat = client.get("/api/templates").json()
    assert {"invite", "reject"} <= set(cat) and all(len(v) >= 2 for v in cat.values())
    good = (ROOT / "seed" / "cv_07_lavanya_iyer.docx").read_bytes()
    cid = client.post("/api/upload", data={"role": "PM"}, files=[("files", ("l.docx", good))]).json()["results"][0]["id"]
    client.post(f"/api/candidates/{cid}/process")
    d = client.post(f"/api/candidates/{cid}/template", json={"template_id": "invite_other_role"}).json()
    assert "Senior Product Manager" in d["invite_body"] and d["invite_body"].startswith("Hi Lavanya,")
    assert d["reject_body"] == "Hi Lavanya, thank you."  # the other draft is untouched
    assert client.post(f"/api/candidates/{cid}/template", json={"template_id": "nope"}).status_code == 400
    assert any(e["action"] == "template_used" for e in d["events"])

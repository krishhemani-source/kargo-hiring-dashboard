import json

import pytest

from app import llm
from app.rubric import load_rubrics, weighted_percent
from app.parsing import relocation_gate, extract_details


def test_rubrics_load_with_weights():
    r = load_rubrics()
    assert len(r["PM"].criteria) == 5 and r["PM"].total_weight == 100
    assert len(r["SPM"].criteria) == 6 and r["SPM"].total_weight == 100
    assert [c.weight for c in r["PM"].criteria] == [30, 20, 20, 15, 15]


def test_weighted_percent_is_backend_math():
    pm = load_rubrics()["PM"]
    assert weighted_percent(pm, {c.id: 5 for c in pm.criteria}) == 100.0
    assert weighted_percent(pm, {c.id: 1 for c in pm.criteria}) == 20.0
    # 5*30 + 4*20 + 3*20 + 2*15 + 1*15 = 335 -> 67%
    assert weighted_percent(pm, {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}) == 67.0


def _reply(obj):
    return json.dumps(obj)


def test_score_json_validated_and_retried_once(monkeypatch):
    pm = load_rubrics()["PM"]
    good = {"criteria": [{"id": c.id, "score": 3, "evidence": "e", "reason": "r"} for c in pm.criteria]}
    replies = iter(["not json at all", _reply(good)])
    calls = []
    monkeypatch.setattr(llm, "_call", lambda s, u, p: calls.append(u) or next(replies))
    out = llm.score_cv(pm, "some redacted cv text", {"name": "Zed Q"})
    assert len(calls) == 2 and "previous reply was invalid" in calls[1]
    assert [c["score"] for c in out["criteria"]] == [3] * 5


def test_score_out_of_range_fails_after_retry(monkeypatch):
    pm = load_rubrics()["PM"]
    bad = {"criteria": [{"id": c.id, "score": 7, "evidence": "", "reason": ""} for c in pm.criteria]}
    monkeypatch.setattr(llm, "_call", lambda s, u, p: _reply(bad))
    with pytest.raises(llm.LLMError):
        llm.score_cv(pm, "cv", {"name": "Zed Q"})


def test_rejection_may_not_reveal_scores(monkeypatch):
    pm = load_rubrics()["PM"]
    leak = {"who": "w", "why_ranked": "y", "probes": ["a", "b", "c"],
            "invite": {"subject": "s", "body": "b"},
            "reject": {"subject": "s", "body": "You scored 43% on our rubric."}}
    monkeypatch.setattr(llm, "_call", lambda s, u, p: _reply(leak))
    scoring = {"criteria": [{"id": c.id, "score": 2, "evidence": "e"} for c in pm.criteria]}
    with pytest.raises(llm.LLMError):
        llm.draft_brief_and_emails("PM", "cv", scoring, pm, 40.0, "SPM", 30.0, {"name": "Zed Q"})


@pytest.mark.parametrize("header,expected", [
    ("Asha Rao\nasha@x.com | Mumbai\n", "pass"),
    ("Asha Rao\nasha@x.com | Pune\nWilling to relocate to Mumbai.\n", "pass"),
    ("Asha Rao\nasha@x.com | Pune\n", "unclear"),
    ("Asha Rao\nasha@x.com | Pune\nNot willing to relocate.\n", "fail"),
])
def test_relocation_gate(header, expected):
    assert relocation_gate(extract_details("x.pdf", header))[0] == expected

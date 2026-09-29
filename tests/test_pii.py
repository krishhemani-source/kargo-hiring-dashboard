"""Guardrail: no personal data may reach an LLM prompt."""
from pathlib import Path

import pytest

from app import llm
from app.parsing import PIILeak, assert_no_pii, extract_details, extract_text, redact
from app.rubric import load_rubrics

ROOT = Path(__file__).resolve().parent.parent
ALL_CVS = sorted((ROOT / "seed").glob("*")) + sorted((ROOT / "applicants").glob("*"))

SAMPLE = """Priya Krishnan
Product Manager
+91 98442 31075 · priya.k@example.com · Mumbai · linkedin.com/in/priyakrishnan-pm
PROFESSIONAL SUMMARY
Priya has 4 years in logistics operations. Portfolio: https://priya.dev/work
Operations Coordinator, Mahindra Logistics Ltd., Mumbai · Jul 2020 – Feb 2022
""" + "Handled carrier allocation and customs documentation. " * 10


def _details(text=SAMPLE, fn="pm_01_priya_krishnan.pdf"):
    ex = extract_details(fn, text)
    return ex, redact(ex, fn), {"name": ex.name, "email": ex.email, "phone": ex.phone, "links": ex.links}


def test_extracts_fields():
    ex, _, _ = _details()
    assert ex.name == "Priya Krishnan"
    assert ex.email == "priya.k@example.com"
    assert "98442" in ex.phone
    assert ex.city == "Mumbai"
    assert any("linkedin.com" in l for l in ex.links)


def test_redaction_removes_all_pii():
    ex, red, pii = _details()
    for s in ("Priya", "Krishnan", "priya.k@example.com", "98442", "linkedin.com", "priya.dev"):
        assert s.lower() not in red.lower(), s
    assert "[NAME]" in red and "[EMAIL]" in red and "[PHONE]" in red
    # Work history survives redaction, including job locations below the header
    assert "Mahindra Logistics Ltd., Mumbai" in red
    assert_no_pii(red, pii, cv_part=red)  # does not raise


@pytest.mark.parametrize("leak", ["priya.k@example.com", "+91 98442 31075", "linkedin.com/in/x", "Priya Krishnan"])
def test_guard_blocks_leaks(leak):
    _, red, pii = _details()
    with pytest.raises(PIILeak):
        assert_no_pii(red + "\n" + leak, pii, cv_part=red + "\n" + leak)


def test_guard_blocks_first_name_in_cv_part():
    _, red, pii = _details()
    with pytest.raises(PIILeak):
        assert_no_pii(red, pii, cv_part=red + " Krishnan led the project")


@pytest.mark.parametrize("path", ALL_CVS, ids=lambda p: p.name)
def test_every_sample_cv_redacts_cleanly(path):
    ex = extract_details(path.name, extract_text(path.name, path.read_bytes()))
    red = redact(ex, path.name)
    pii = {"name": ex.name, "email": ex.email, "phone": ex.phone, "links": ex.links}
    assert_no_pii(red, pii, cv_part=red)


def test_llm_prompts_never_contain_pii(monkeypatch):
    """Intercept the real LLM call path and inspect every prompt it would send."""
    sent = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            sent.append(config.system_instruction + "\n" + contents)
            raise RuntimeError("stop")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(llm, "client", lambda: FakeClient())
    _, red, pii = _details()
    rubric = load_rubrics()["PM"]
    with pytest.raises(llm.LLMError):
        llm.score_cv(rubric, red, pii)
    assert sent
    for p in sent:
        for s in ("Priya", "Krishnan", "priya.k@example.com", "98442 31075", "linkedin.com"):
            assert s.lower() not in p.lower(), s


def test_llm_refuses_unredacted_text(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda: pytest.fail("LLM must not be called"))
    ex, _, pii = _details()
    with pytest.raises(PIILeak):
        llm.score_cv(load_rubrics()["PM"], ex.raw_text, pii)

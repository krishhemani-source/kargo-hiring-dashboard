"""All LLM calls live here. Every prompt passes through assert_no_pii() first.

The LLM only scores against the rubric and drafts text. It never decides to
invite or reject, and it never computes the weighted total.
"""
import json
import os
import re
import threading
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from . import config
from .parsing import assert_no_pii
from .rubric import ROLE_LABEL, Rubric, load_jd

_client = None


class LLMError(Exception):
    pass


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set in .env")
        _client = genai.Client(api_key=key)
    return _client


_exhausted = set()  # models whose daily quota is used up (this process)
_local = threading.local()


def models_used() -> list:
    return sorted(getattr(_local, "used", set()))


def reset_models_used() -> None:
    _local.used = set()


def _retry_delay(err: str, default: float) -> float:
    m = re.search(r"retry in ([\d.]+)s", err)
    return min(float(m.group(1)) + 1, 60) if m else default


def _call(system: str, user: str, pii: dict) -> str:
    # Guardrail: the CV-derived part (user) is checked for name tokens too.
    assert_no_pii(system + "\n" + user, pii, cv_part=user)
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=16000,
    )
    chain = [m for m in config.GEMINI_MODELS if m not in _exhausted]
    if not chain:
        raise LLMError("Every configured Gemini model is out of free-tier quota for today. "
                       "Enable billing on the key or try again tomorrow.")
    last = None
    deadline = time.monotonic() + config.LLM_CALL_BUDGET_S

    def can_wait(seconds):
        return time.monotonic() + seconds < deadline

    for model in chain:
        if time.monotonic() > deadline:
            break
        for attempt in range(3):
            try:
                resp = client().models.generate_content(model=model, contents=user, config=cfg)
            except genai_errors.APIError as e:
                code, msg = getattr(e, "code", None), str(e)
                last = f"{model}: {code} {msg[:200]}"
                if code == 429 and "PerDay" in msg:
                    _exhausted.add(model)  # daily quota gone, next model
                    break
                wait = _retry_delay(msg, 10)
                if code == 429 and attempt < 2 and can_wait(wait):
                    time.sleep(wait)
                    continue
                if code in (500, 502, 503, 504) and attempt < 1 and can_wait(4):
                    time.sleep(4)
                    continue
                if code in (429, 500, 502, 503, 504, 404):
                    break  # try the next model
                if code in (400, 401, 403) and "API key" in msg:
                    raise LLMError(f"Gemini rejected the API key: {msg[:300]}") from e
                raise LLMError(f"Gemini API error {code}: {msg[:300]}") from e
            except Exception as e:
                raise LLMError(f"Could not reach the Gemini API: {e}") from e
            text = resp.text or ""
            if not text:
                reason = resp.candidates[0].finish_reason if resp.candidates else getattr(resp, "prompt_feedback", None)
                raise LLMError(f"Gemini ({model}) returned no text (finish reason: {reason})")
            if not hasattr(_local, "used"):
                _local.used = set()
            _local.used.add(model)
            return text
    raise LLMError(f"All Gemini models failed or are rate-limited. Last error: {last}")


def _parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def _json_call(system, user, pii, validate):
    """Call, validate, retry once with the error fed back."""
    last_err = None
    for attempt in range(2):
        prompt = user
        if last_err:
            prompt += (
                f"\n\nYour previous reply was invalid ({last_err}). "
                "Reply again with ONLY the JSON object, exactly matching the schema."
            )
        raw = _call(system, prompt, pii)
        try:
            return validate(_parse_json(raw))
        except (ValueError, KeyError, TypeError) as e:
            last_err = str(e)[:300]
    raise LLMError(f"Model returned invalid JSON twice: {last_err}")


# ---------------- scoring ----------------

SCORING_GUIDANCE = """How to apply this rubric (Kargo's own findings from past hires):
- "Hands-on operations" means the person personally did freight / logistics / customs / carrier / port / warehouse work (documentation, CHA, dispatch, carrier coordination). Selling to, consulting for, building software for, or managing an ops team from a distance is not the same thing. Title and credentials (MBA, certifications) do not count.
- "Owns breakages" rewards fixing a live failure outside one's remit and documenting it. An assigned on-call rotation or escalating to someone else is not ownership beyond the role. A CV that shows only wins scores low here.
- For "unprompted fix adopted", artifacts adopted by frontline operators (the people doing the work) count more than templates adopted by one's own function (e.g. a PRD template adopted by other PMs).
- Score only what the CV shows. If there is no evidence for a criterion, score 1 and say so. Do not reward years or seniority for their own sake.
- Personal details have been replaced with placeholders like [NAME], [EMAIL], [CITY]. Ignore them."""


def _rubric_block(rubric: Rubric) -> str:
    lines = []
    for c in rubric.criteria:
        lines.append(f"{c.id}. {c.name} (weight {c.weight}%)\n   1 = {c.low}\n   5 = {c.high}")
    return "\n".join(lines)


def score_cv(rubric: Rubric, redacted_cv: str, pii: dict) -> dict:
    role = rubric.role
    jd = load_jd(role)
    system = (
        f"You are a careful screener for Kargo, a Series A freight-software startup in Mumbai. "
        f"You score CVs against the {ROLE_LABEL[role]} ({role}) rubric. You do not decide who is hired.\n\n"
        f"RUBRIC ({role}), score each criterion 1-5 (integers; 2-4 are in between the anchors):\n"
        f"{_rubric_block(rubric)}\n\n{SCORING_GUIDANCE}"
        + (f"\n\nJOB DESCRIPTION (context only):\n{jd}" if jd else "")
    )
    ids = [c.id for c in rubric.criteria]
    schema = {
        "criteria": [
            {"id": "<criterion number>", "score": "<integer 1-5>",
             "evidence": "<short quote or close paraphrase from the CV, or 'No evidence in CV'>",
             "reason": "<one line on why this score>"}
        ]
    }
    user = (
        f"Score this CV against every criterion ({', '.join(map(str, ids))}).\n"
        f"Reply with ONLY a JSON object of this shape, no prose, no markdown:\n{json.dumps(schema)}\n\n"
        f"<cv>\n{redacted_cv}\n</cv>"
    )

    def validate(obj):
        items = obj["criteria"]
        by_id = {}
        for it in items:
            cid = int(it["id"])
            score = int(it["score"])
            if cid not in ids:
                raise ValueError(f"unknown criterion id {cid}")
            if not 1 <= score <= 5:
                raise ValueError(f"score {score} for criterion {cid} is outside 1-5")
            by_id[cid] = {
                "id": cid,
                "score": score,
                "evidence": str(it.get("evidence", "")).strip()[:500],
                "reason": str(it.get("reason", "")).strip()[:300],
            }
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"missing criteria {missing}")
        return [by_id[i] for i in ids]

    return {"criteria": _json_call(system, user, pii, validate)}


# ---------------- brief + emails ----------------

REVEALS_SCORE = re.compile(r"\d+(\.\d+)?\s*%|\bscor(e|ed|es|ing)\b|\brubric\b|\bweighted\b|\b\d\s*/\s*5\b", re.I)


def draft_brief_and_emails(role: str, redacted_cv: str, scoring: dict, rubric: Rubric,
                           pct: float, other_role: str, other_pct: float, pii: dict) -> dict:
    names = {c.id: c.name for c in rubric.criteria}
    score_lines = "\n".join(
        f"- {names[c['id']]}: {c['score']}/5. Evidence: {c['evidence']}" for c in scoring["criteria"]
    )
    system = (
        f"You help {config.FOUNDER_NAME}, founder of Kargo (Series A freight-forwarding software, in-office in Mumbai), "
        "prepare for hiring conversations. You write an interview brief and two email drafts. "
        f"{config.FOUNDER_NAME} decides whether to invite or reject; you only draft both options.\n"
        "Rules:\n"
        "- Refer to the candidate as [NAME] in the emails (it is replaced with their first name later). Never invent a name.\n"
        "- Both emails: warm, plain, specific to something real in the CV. No clichés, no exclamation marks overload.\n"
        f"- Invite: invites them to a 45-minute conversation for the {ROLE_LABEL[role]} role, mentions one or two specific things from their CV, asks for a few times that work next week. Signed '{config.FOUNDER_NAME}, Founder, Kargo'.\n"
        f"- Rejection: kind and brief (under 90 words). Thank them, say Kargo won't move forward for this role, one genuine specific note if natural. Never mention scores, percentages, rubrics or criteria. Signed '{config.FOUNDER_NAME}, Kargo'.\n"
        "- Brief: 'who' is 3-4 short lines on who they are professionally; 'why_ranked' is 1 line on why they ranked where they did; "
        "'probes' are exactly 3 interview questions aimed at their lowest-scoring or least-evidenced criteria."
    )
    schema = {
        "who": "<3-4 lines>",
        "why_ranked": "<1 line>",
        "probes": ["<q1>", "<q2>", "<q3>"],
        "invite": {"subject": "<subject>", "body": "<plain-text body>"},
        "reject": {"subject": "<subject>", "body": "<plain-text body>"},
    }
    user = (
        f"Role applied for: {ROLE_LABEL[role]}\n"
        f"Weighted fit for this role: {pct}% (other role, {ROLE_LABEL[other_role]}: {other_pct}%)\n"
        f"Criterion scores:\n{score_lines}\n\n"
        f"Reply with ONLY a JSON object of this shape:\n{json.dumps(schema)}\n\n"
        f"<cv>\n{redacted_cv}\n</cv>"
    )

    def validate(obj):
        probes = [str(p).strip() for p in obj["probes"] if str(p).strip()]
        if len(probes) < 3:
            raise ValueError("need exactly 3 probe questions")
        out = {
            "who": str(obj["who"]).strip(),
            "why_ranked": str(obj["why_ranked"]).strip(),
            "probes": probes[:3],
            "invite": {"subject": str(obj["invite"]["subject"]).strip(), "body": str(obj["invite"]["body"]).strip()},
            "reject": {"subject": str(obj["reject"]["subject"]).strip(), "body": str(obj["reject"]["body"]).strip()},
        }
        if not all([out["who"], out["invite"]["body"], out["reject"]["body"]]):
            raise ValueError("empty fields")
        if REVEALS_SCORE.search(out["reject"]["subject"] + " " + out["reject"]["body"]):
            raise ValueError("the rejection email mentions scores/percentages/rubric; remove them")
        return out

    return _json_call(system, user, pii, validate)

"""Fixed email templates for every kind of draft.

Used two ways:
- as the fallback when Gemini can't write the drafts (overloaded, out of quota), so a scored
  candidate always has an invite and a rejection ready to edit;
- from the dashboard's "Start from template" picker, to replace a draft with a standard one.

Templates are filled in the backend with the candidate's first name. Nothing here goes to the LLM.
Rejections never mention scores, percentages or the rubric (checked in tests).
"""
from . import config
from .rubric import ROLE_LABEL

TEMPLATES = {
    "invite": [
        {
            "id": "invite_standard",
            "label": "Interview invite (45 min)",
            "subject": "Kargo: {role} conversation",
            "body": (
                "Hi {first},\n\n"
                "Thank you for applying for the {role} role at Kargo. I read your CV and would like to talk.\n\n"
                "Could we do a 45-minute conversation next week? We're in-office in Mumbai, and I'm happy to do the "
                "first one in person or on a video call. Please send me two or three times that work for you.\n\n"
                "Looking forward to it.\n\n"
                "{founder}, Founder, Kargo"
            ),
        },
        {
            "id": "invite_intro_call",
            "label": "Short intro call (20 min)",
            "subject": "Quick intro call with Kargo",
            "body": (
                "Hi {first},\n\n"
                "Thanks for applying for the {role} role at Kargo. Before a full interview, I'd like a 20-minute "
                "call to tell you more about what we're building in freight software and to hear what you're "
                "looking for next.\n\n"
                "Could you share a few times that work for you next week?\n\n"
                "{founder}, Founder, Kargo"
            ),
        },
        {
            "id": "invite_other_role",
            "label": "Invite, for the other role",
            "subject": "Kargo: a conversation about the {other_role} role",
            "body": (
                "Hi {first},\n\n"
                "Thank you for applying for the {role} role at Kargo. Reading your CV, I think your experience may "
                "fit our {other_role} role even better, and I'd like to talk to you about it.\n\n"
                "Could we do a 45-minute conversation next week? Please send me two or three times that work for "
                "you, and tell me if you'd prefer to stay focused on the {role} role. We can cover both.\n\n"
                "{founder}, Founder, Kargo"
            ),
        },
    ],
    "reject": [
        {
            "id": "reject_standard",
            "label": "Rejection",
            "subject": "Your application to Kargo",
            "body": (
                "Hi {first},\n\n"
                "Thank you for applying for the {role} role at Kargo and for the time you put into it. "
                "We won't be moving forward with your application for this role.\n\n"
                "I wish you the very best with your search.\n\n"
                "{founder}, Kargo"
            ),
        },
        {
            "id": "reject_keep_in_touch",
            "label": "Rejection, keep in touch",
            "subject": "Your application to Kargo",
            "body": (
                "Hi {first},\n\n"
                "Thank you for applying for the {role} role at Kargo. We won't be moving forward for this role, "
                "but I enjoyed reading about your work and would like to keep your details for future openings. "
                "If that's not okay, just reply and let me know.\n\n"
                "Wishing you the best until then.\n\n"
                "{founder}, Kargo"
            ),
        },
        {
            "id": "reject_relocation",
            "label": "Rejection, role is in-office in Mumbai",
            "subject": "Your application to Kargo",
            "body": (
                "Hi {first},\n\n"
                "Thank you for applying for the {role} role at Kargo. The role is full-time in our Mumbai office, "
                "and we can't offer remote work or relocation support for it right now, so we won't be moving "
                "forward with your application.\n\n"
                "I'm sorry it didn't line up this time, and I wish you the best.\n\n"
                "{founder}, Kargo"
            ),
        },
        {
            "id": "reject_role_filled",
            "label": "Rejection, role filled",
            "subject": "Update on the {role} role at Kargo",
            "body": (
                "Hi {first},\n\n"
                "Thank you for your interest in the {role} role at Kargo. We've now filled the position, so we "
                "won't be taking your application further.\n\n"
                "Thanks again, and all the best with your search.\n\n"
                "{founder}, Kargo"
            ),
        },
    ],
}

KINDS = tuple(TEMPLATES)
_BY_ID = {t["id"]: (kind, t) for kind, ts in TEMPLATES.items() for t in ts}


def catalog() -> dict:
    """What the dashboard shows in its template pickers."""
    return {kind: [{"id": t["id"], "label": t["label"]} for t in ts] for kind, ts in TEMPLATES.items()}


def kind_of(template_id: str):
    hit = _BY_ID.get(template_id)
    return hit[0] if hit else None


def render(template_id: str, cand: dict) -> dict:
    """Fill a template for this candidate. Returns {kind, subject, body}."""
    if template_id not in _BY_ID:
        raise KeyError(f"unknown template {template_id}")
    kind, t = _BY_ID[template_id]
    role = cand.get("role_applied") or "PM"
    other = "SPM" if role == "PM" else "PM"
    first = (cand.get("name") or "").split(" ")[0] or "there"
    values = {"first": first, "role": ROLE_LABEL[role], "other_role": ROLE_LABEL[other], "founder": config.FOUNDER_NAME}
    return {"kind": kind, "subject": t["subject"].format(**values), "body": t["body"].format(**values)}


def default_for(kind: str, cand: dict) -> str:
    """Template used when the AI drafts aren't available."""
    if kind == "reject":
        return "reject_relocation" if cand.get("relocation") == "fail" else "reject_standard"
    return "invite_standard"

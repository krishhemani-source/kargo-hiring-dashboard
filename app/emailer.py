"""Step 6: send via Resend. Only ever called from the Send button's endpoint."""
import resend

from . import config


class SendError(Exception):
    pass


def send(to_email: str, subject: str, body: str) -> dict:
    if not config.RESEND_API_KEY:
        raise SendError("RESEND_API_KEY is not set in .env")
    if config.TEST_MODE:
        if not config.FOUNDER_EMAIL:
            raise SendError("TEST_MODE is on but FOUNDER_EMAIL is not set in .env")
        recipient = config.FOUNDER_EMAIL
        subject = f"[TEST → {to_email or 'no email on CV'}] {subject}"
    else:
        if not to_email:
            raise SendError("This candidate has no email address.")
        recipient = to_email
    params = {"from": config.FROM_EMAIL, "to": [recipient], "subject": subject, "text": body}
    if config.FOUNDER_EMAIL and not config.TEST_MODE:
        params["reply_to"] = config.FOUNDER_EMAIL
    resend.api_key = config.RESEND_API_KEY
    try:
        r = resend.Emails.send(params)
    except Exception as e:
        raise SendError(f"Resend error: {e}") from e
    msg_id = r.get("id") if isinstance(r, dict) else getattr(r, "id", None)
    if not msg_id:
        raise SendError(f"Resend returned no message id: {r}")
    return {"id": msg_id, "to": recipient, "subject": subject}

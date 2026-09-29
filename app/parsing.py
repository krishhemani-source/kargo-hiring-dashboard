"""Step 2 (Context): extract CV text, pull out personal details, redact them.

Only the output of `redact()` is ever allowed near the LLM. `assert_no_pii()` is
the last line of defence and is called on every prompt in llm.py.
"""
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone-ish runs: digits with spaces/dashes/dots/brackets, at least 10 digits total.
PHONE_RE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s\-]?)?\(?(?:\d[\s\-()]?){9}\d(?!\d)")
URL_RE = re.compile(
    r"(?:https?:/{1,2}\S+|www\.\S+|(?:linkedin\.com|github\.com|behance\.net|flowcv\.me|medium\.com)/[A-Za-z0-9_\-./%]*)",
    re.I,
)
MUMBAI_AREAS = ["mumbai", "navi mumbai", "thane", "bombay", "powai", "andheri", "bandra", "vashi"]
CITIES = MUMBAI_AREAS + [
    "bengaluru", "bangalore", "delhi", "new delhi", "gurugram", "gurgaon", "noida", "pune",
    "chennai", "hyderabad", "kolkata", "ahmedabad", "kochi", "cochin", "jaipur", "indore",
    "chandigarh", "lucknow", "coimbatore", "nagpur", "surat", "vadodara", "goa", "mysuru",
    "visakhapatnam", "bhubaneswar", "trivandrum", "thiruvananthapuram",
]


@dataclass
class Extracted:
    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    links: list = field(default_factory=list)
    city: str = ""
    warnings: list = field(default_factory=list)

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else ""


# ---------- text extraction ----------

def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        import docx

        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = docx.Document(io.BytesIO(data))
        parts = []
        # Walk the body in document order: contact blocks are often a table at the top.
        for el in d.element.body.iterchildren():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "p":
                parts.append(Paragraph(el, d).text)
            elif tag == "tbl":
                for row in Table(el, d).rows:
                    cells = []
                    for c in row.cells:
                        if c.text not in cells:  # merged cells repeat
                            cells.append(c.text)
                    parts.append("\n".join(cells))
        # Headers sometimes hold the contact block.
        for s in d.sections:
            parts = [p.text for p in s.header.paragraphs] + parts
        return "\n".join(parts)
    if ext == ".pdf":
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(_pdf_page_text(pg) for pg in pdf.pages)
    raise ValueError(f"Unsupported file type '{ext}'. Upload PDF or DOCX.")


def _pdf_page_text(page) -> str:
    """Rebuild lines from runs of characters in content-stream order.

    Sorting characters purely by position (what most extractors do) interleaves
    text that was pasted on top of other text, e.g. an edited contact header:
    "squad_2@pg27" over "nikhil-sharma" becomes "snqikuhaild-s_h2a@...". Keeping
    stream-order runs intact and only sorting whole runs avoids that.
    """
    chars = page.dedupe_chars().chars  # dedupe fixes "SSShhhrrreeeyyy" fake-bold
    runs, cur = [], []
    for c in chars:
        if cur:
            p = cur[-1]
            size = max(p.get("size") or 10, 1)
            if abs(c["top"] - p["top"]) > 1.5 or c["x0"] < p["x0"] - 0.5 or c["x0"] - p["x1"] > size * 3:
                runs.append(cur)
                cur = []
        cur.append(c)
    if cur:
        runs.append(cur)

    def run_text(run):
        out, prev = "", None
        for c in run:
            if prev is not None and c["x0"] - prev["x1"] > (prev.get("size") or 10) * 0.2 and not out.endswith(" "):
                out += " "
            out += c["text"]
            prev = c
        return out

    lines = []  # [top, [runs]]
    for r in sorted(runs, key=lambda r: r[0]["top"]):
        top = r[0]["top"]
        if lines and abs(lines[-1][0] - top) <= 3:
            lines[-1][1].append(r)
        else:
            lines.append([top, [r]])
    out = []
    for _, rs in lines:
        text = " ".join(run_text(r).strip() for r in sorted(rs, key=lambda r: r[0]["x0"]))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return "\n".join(out)


# ---------- detail extraction ----------

def _digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def _collapse_repeats(line: str) -> str:
    """'NIKHIL SHARMA Nikhil Sharma' -> 'Nikhil Sharma' (duplicated overlay text)."""
    words = line.split()
    for n in (2, 3, 4):
        # "Rohan Mehta ohan Mehta": the de-duplicated second copy lost its first letter.
        rest = " ".join(words[n:])
        if len(words) == 2 * n and " ".join(words[:n]).lower().endswith(rest.lower()):
            return " ".join(words[:n])
        if len(words) >= 2 * n and len(words) % n == 0:
            chunks = [" ".join(words[i:i + n]) for i in range(0, len(words), n)]
            if len({c.lower() for c in chunks}) == 1:
                return next((c for c in chunks if not c.isupper()), chunks[0])
    return line


def _looks_like_name(line: str) -> bool:
    line = line.strip()
    if not line or any(ch.isdigit() for ch in line) or "@" in line or "|" in line:
        return False
    words = line.replace(".", " ").split()
    if not 2 <= len(words) <= 4:
        return False
    if not all(w.isalpha() for w in words):
        return False
    # Garbled fake-bold ("RROohHaAnN") has mixed case inside a word.
    for w in words:
        inner = w[1:]
        if inner and not (inner.islower() or inner.isupper()):
            return False
    bad = {"resume", "curriculum", "vitae", "product", "manager", "summary", "profile", "senior",
           "education", "experience", "university", "academic", "qualifications", "institute",
           "college", "skills", "professional", "work", "contact", "technological", "portfolio"}
    return not any(w.lower() in bad for w in words)


def name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = [p for p in re.split(r"[_\-\s]+", stem) if p.isalpha()]
    parts = [p for p in parts if p.lower() not in {"cv", "pm", "spm", "resume", "final"}]
    return " ".join(p.capitalize() for p in parts[:3]) if len(parts) >= 2 else ""


def extract_details(filename: str, text: str) -> Extracted:
    ex = Extracted(raw_text=text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = lines[:8]

    for l in head[:4]:
        l = _collapse_repeats(l)
        if _looks_like_name(l):
            ex.name = " ".join(w.capitalize() if w.isupper() else w for w in l.split())
            break
    if not ex.name:
        ex.name = name_from_filename(filename)
        if ex.name:
            ex.warnings.append("Name taken from the file name (first line didn't look like a name).")
        else:
            ex.warnings.append("Could not find the candidate's name.")

    emails = EMAIL_RE.findall(text)
    ex.email = emails[0] if emails else ""
    if not ex.email:
        ex.warnings.append("No email address found. You won't be able to send to this candidate.")

    for m in PHONE_RE.finditer(text):
        if _digits(m.group()) >= 10:
            ex.phone = m.group().strip()
            break
    if not ex.phone:
        ex.warnings.append("No phone number found.")

    ex.links = sorted(set(URL_RE.findall(text)))

    head_text = "\n".join(head).lower()
    # Mumbai first: "Chennai / Mumbai" should count as Mumbai for the gate.
    for city in sorted(MUMBAI_AREAS, key=len, reverse=True) + sorted(CITIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", head_text):
            ex.city = city.title()
            break
    if not ex.city:
        ex.warnings.append("No location found in the contact block.")
    return ex


# ---------- redaction ----------

SECTION_RE = re.compile(
    r"^\s*(professional\s+)?(summary|profile|experience|work experience|education|objective|about|skills|career)\b", re.I
)

def name_tokens(name: str) -> list:
    return [t for t in re.split(r"[\s.]+", name) if len(t) >= 3]


def redact(ex: Extracted, filename: str = "") -> str:
    text = ex.raw_text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[LINK]", text)
    text = PHONE_RE.sub(lambda m: "[PHONE]" if _digits(m.group()) >= 10 else m.group(), text)

    names = {ex.name, name_from_filename(filename)} - {""}
    for n in names:
        text = re.sub(re.escape(n), "[NAME]", text, flags=re.I)
        for tok in name_tokens(n):
            text = re.sub(rf"\b{re.escape(tok)}\b", "[NAME]", text, flags=re.I)

    # Contact block (first lines): also drop cities, and name fragments left by
    # de-duplicated overlay text ("AMAN" -> "MAN"). Job locations further down stay.
    fragments = [t[1:] for n in names for t in name_tokens(n) if len(t) >= 4]
    lines = text.splitlines()
    seen = 0
    for i, l in enumerate(lines):
        if l.strip():
            seen += 1
        if seen > 8 or (seen > 1 and SECTION_RE.match(l)):
            break
        for city in CITIES:
            l = re.sub(rf"\b{re.escape(city)}\b", "[CITY]", l, flags=re.I)
        for frag in fragments:
            l = re.sub(rf"\b{re.escape(frag)}\b", "[NAME]", l, flags=re.I)
        lines[i] = l
    return "\n".join(lines)


class PIILeak(Exception):
    pass


def assert_no_pii(prompt: str, pii: dict, cv_part: str = "") -> None:
    """Raise PIILeak if the prompt carries personal data.

    `pii` holds the candidate's extracted fields. `cv_part` is the candidate-derived
    portion of the prompt, which is also checked for single name tokens (the static
    rubric/JD text may legitimately contain other people's first names).
    """
    problems = []
    if EMAIL_RE.search(prompt):
        problems.append("email address")
    for m in PHONE_RE.finditer(prompt):
        if _digits(m.group()) >= 10:
            problems.append("phone number")
            break
    if URL_RE.search(prompt):
        problems.append("profile URL")
    for key in ("email", "phone"):
        v = (pii.get(key) or "").strip()
        if v and v.lower() in prompt.lower():
            problems.append(key)
    for link in pii.get("links") or []:
        if link.lower() in prompt.lower():
            problems.append("link")
    name = (pii.get("name") or "").strip()
    if name and re.search(re.escape(name), prompt, re.I):
        problems.append("full name")
    if cv_part and name:
        for tok in name_tokens(name):
            if re.search(rf"\b{re.escape(tok)}\b", cv_part, re.I):
                problems.append(f"name token '{tok}'")
    if problems:
        raise PIILeak("Refusing to call the LLM, prompt contains: " + ", ".join(sorted(set(problems))))


# ---------- relocation gate (backend only; location never goes to the LLM) ----------

RELOCATE_YES = re.compile(r"(willing|open|happy|ready|able)\s+to\s+relocat|relocating\s+to\s+mumbai|open\s+to\s+relocation", re.I)
RELOCATE_NO = re.compile(r"(not|un)\s*(willing|open|able)\s+to\s+relocat|no\s+relocation|cannot\s+relocate", re.I)


def relocation_gate(ex: Extracted) -> tuple:
    text = ex.raw_text
    if RELOCATE_NO.search(text):
        return "fail", "CV says the candidate is not open to relocating."
    if ex.city and ex.city.lower() in MUMBAI_AREAS:
        return "pass", f"Based in {ex.city}."
    if RELOCATE_YES.search(text):
        return "pass", "CV says the candidate is willing to relocate."
    if ex.city:
        return "unclear", f"Based in {ex.city}; no mention of relocating. Ask."
    return "unclear", "No location found. Ask."

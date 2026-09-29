"""Load the PM and SPM rubrics (criteria, weights, 1/5 anchors) from rubric.txt."""
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import config

ROLES = ("PM", "SPM")
ROLE_LABEL = {"PM": "Product Manager", "SPM": "Senior Product Manager"}


@dataclass
class Criterion:
    id: int
    name: str
    weight: int  # percent
    low: str = ""   # anchor for 1
    high: str = ""  # anchor for 5


@dataclass
class Rubric:
    role: str
    criteria: list = field(default_factory=list)

    @property
    def total_weight(self) -> int:
        return sum(c.weight for c in self.criteria)


CRIT_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*\.{3,}\s*(\d+)%\s*$")
ANCHOR_RE = re.compile(r"^\s*([15])\s*=\s*(.+?)\s*$")


def parse_rubrics(text: str) -> dict:
    rubrics, current = {}, None
    for line in text.splitlines():
        upper = line.strip().upper()
        if upper.startswith("SENIOR PM RUBRIC"):
            current = rubrics.setdefault("SPM", Rubric("SPM"))
            continue
        if upper.startswith("PM RUBRIC"):
            current = rubrics.setdefault("PM", Rubric("PM"))
            continue
        if upper.startswith("GATE") or upper.startswith("SCORING"):
            current = None
        if current is None:
            continue
        m = CRIT_RE.match(line)
        if m:
            current.criteria.append(Criterion(int(m.group(1)), m.group(2).strip(), int(m.group(3))))
            continue
        m = ANCHOR_RE.match(line)
        if m and current.criteria:
            setattr(current.criteria[-1], "low" if m.group(1) == "1" else "high", m.group(2))
    for role in ROLES:
        r = rubrics.get(role)
        if not r or not r.criteria:
            raise ValueError(f"rubric.txt: could not find the {role} rubric")
        if r.total_weight != 100:
            raise ValueError(f"rubric.txt: {role} weights add to {r.total_weight}%, not 100%")
    return rubrics


@lru_cache(maxsize=1)
def load_rubrics() -> dict:
    return parse_rubrics(config.RUBRIC_PATH.read_text(encoding="utf-8"))


def weighted_percent(rubric: Rubric, scores: dict) -> float:
    """sum(score x weight) / 5, with weights in percent -> 0..100. Computed here, never by the LLM."""
    total = sum(scores[c.id] * c.weight for c in rubric.criteria)
    return round(total / 5, 1)


@lru_cache(maxsize=2)
def load_jd(role: str) -> str:
    """Job description text for context, if the JD docx is in jds/. Optional."""
    import docx

    want = "senior product manager" if role == "SPM" else "product manager"
    for p in sorted(config.JD_DIR.glob("*.docx")) if config.JD_DIR.exists() else []:
        stem = p.stem.lower().replace("_", " ")
        if stem.endswith(want) and (role == "SPM" or "senior" not in stem):
            return "\n".join(par.text for par in docx.Document(p).paragraphs if par.text.strip())
    return ""

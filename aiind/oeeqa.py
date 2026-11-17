"""
An OEE question-answering task with answers that can be computed, for chapter 4.

A synthetic plant (6 machines, 14 days, 3 shifts) gives the data. Questions are generated from
templates, and the gold answer of each is computed from the data by code, so there is no
annotation to argue about. Three answering systems with different, known flaws are provided:

  * `regras`      parses the question and computes exactly; cannot parse some phrasings or diagnose;
  * `ingenuo`     a plausible shortcut: averages shift OEEs, never says "no data", always blames availability;
  * `assistente`  SIMULATED language-model-like system: right most of the time, with random slips,
                  verbose answers and invented values for missing data. Chapter 6 uses a real model.

Everything is synthetic. No plant data.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .corpus import MACHINES

DAYS = 14
IDEAL = dict(zip(MACHINES, [3.0, 4.5, 6.0, 8.0, 5.0, 12.0]))   # ideal cycle, seconds per piece


def make_data(seed: int = 5) -> list[dict]:
    """One row per machine, day and shift."""
    rng = random.Random(seed)
    rows = []
    for m in MACHINES:
        for d in range(1, DAYS + 1):
            for s in (1, 2, 3):
                planned = rng.choice([480, 480, 480, 420, 360, 300])
                failure = rng.choice([0, 0, 0, 5, 15, 40, 90]) if rng.random() < 0.6 else 0
                other = rng.choice([0, 10, 20, 30])
                failure, other = min(failure, planned // 3), min(other, planned // 5)
                run = planned - failure - other
                total = int(run * 60 * rng.uniform(0.7, 0.98) / IDEAL[m])
                defects = int(total * rng.uniform(0.01, 0.07))
                rows.append(dict(machine=m, day=d, shift=s, planned=planned, failure=failure, other=other,
                                 run=run, total=total, defects=defects))
    return rows


def _sel(data, machine=None, day=None):
    return [r for r in data if (machine is None or r["machine"] == machine) and (day is None or r["day"] == day)]


def factors(data, machine, day) -> dict[str, float]:
    """Availability, performance, quality and OEE of a machine on a day, aggregated by time and pieces."""
    rs = _sel(data, machine, day)
    planned, run = sum(r["planned"] for r in rs), sum(r["run"] for r in rs)
    total, good = sum(r["total"] for r in rs), sum(r["total"] - r["defects"] for r in rs)
    a, p, q = run / planned, IDEAL[machine] * total / (run * 60), good / total
    return dict(disponibilidade=a, desempenho=p, qualidade=q, oee=a * p * q)


def oee_naive(data, machine, day) -> float:
    """The shortcut: average of the shift OEEs, ignoring that shifts have different lengths."""
    vals = []
    for r in _sel(data, machine, day):
        vals.append((r["run"] / r["planned"]) * (IDEAL[machine] * r["total"] / (r["run"] * 60)) * ((r["total"] - r["defects"]) / r["total"]))
    return sum(vals) / len(vals)


# ------------------------------------------------------------------ questions
@dataclass
class Q:
    id: int
    text: str
    kind: str         # pecas | falha | oee | pior | perda | semdados
    machine: str | None
    day: int
    template: int
    gold: str         # number as text, machine name, factor name, or "sem dados"
    difficulty: str


TEXTS = {
    "pecas": ["Quantas peças boas a {m} produziu no dia {d}?", "Quantas peças boas a {m} fez no dia {d}?", "Peças boas da {m} no {d}º dia?"],
    "falha": ["Quantos minutos de parada por falha teve a {m} no dia {d}?", "Quanto tempo de falha a {m} teve no dia {d}, em minutos?", "Minutos de falha da {m} no {d}º dia?"],
    "oee": ["Qual foi o OEE da {m} no dia {d}?", "Qual o OEE da {m} no dia {d}, em porcentagem?", "OEE da {m} no {d}º dia?"],
    "pior": ["Qual máquina teve o menor OEE no dia {d}?", "No dia {d}, qual máquina teve o pior OEE?", "Pior OEE do {d}º dia: qual máquina?"],
    "perda": ["Qual o maior fator de perda do OEE da {m} no dia {d}: disponibilidade, desempenho ou qualidade?",
              "No dia {d}, o que mais prejudicou o OEE da {m}: disponibilidade, desempenho ou qualidade?",
              "Maior perda de OEE da {m} no {d}º dia: disponibilidade, desempenho ou qualidade?"],
}
DIFFICULTY = {"pecas": "fácil", "falha": "fácil", "oee": "média", "pior": "difícil", "perda": "difícil", "semdados": "armadilha"}


def gold_for(data, kind, machine, day):
    if day > DAYS:
        return "sem dados"
    if kind == "pecas":
        return str(sum(r["total"] - r["defects"] for r in _sel(data, machine, day)))
    if kind == "falha":
        return str(sum(r["failure"] for r in _sel(data, machine, day)))
    if kind == "oee":
        return f"{100 * factors(data, machine, day)['oee']:.1f}"
    if kind == "pior":
        return min(MACHINES, key=lambda m: factors(data, m, day)["oee"])
    f = factors(data, machine, day)
    return max(("disponibilidade", "desempenho", "qualidade"), key=lambda k: 1 - f[k])


def build_questions(data, n: int = 300, seed: int = 21, missing: float = 0.12) -> list[Q]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = rng.choice(["pecas", "falha", "oee", "pior", "perda"])
        m, d, t = rng.choice(MACHINES), rng.randint(1, DAYS), rng.randrange(3)
        label = kind
        if kind != "pior" and rng.random() < missing:
            d, label = rng.randint(DAYS + 1, DAYS + 6), "semdados"
        text = TEXTS[kind][t].format(m=m, d=d)
        out.append(Q(i, text, label, None if kind == "pior" else m, d, t, gold_for(data, kind, m, d), DIFFICULTY[label]))
        out[-1].asked = kind   # the real question type, kept for the systems' parsers
    return out


# ------------------------------------------------------------------ scoring
ABSTAIN = ("sem dados", "não sei", "nao sei", "não há dados")


def number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:[.,]\d+)?", text.replace(".", "") if re.search(r"\d\.\d{3}\b", text) else text)
    return float(m.group().replace(",", ".")) if m else None


def correct(q: Q, answer: str) -> bool:
    a = answer.lower()
    if q.gold == "sem dados":
        return any(x in a for x in ABSTAIN)
    if any(x in a for x in ABSTAIN):
        return False
    if q.kind == "oee":
        v = number(a)
        return v is not None and abs(v - float(q.gold)) <= 0.5
    if q.kind in ("pecas", "falha"):
        v = number(a)
        return v is not None and v == float(q.gold)
    return q.gold in a


def format_ok(answer: str) -> bool:
    """Instruction following: the answer must be only the value (number with optional unit, or one name)."""
    return bool(re.fullmatch(r"[\d.,]+ ?%?|[a-zç]+", answer.strip().lower()))


# ------------------------------------------------------------------ systems
def parse(question: str):
    """The question as the rule system understands it, or None. Understands the first two phrasings only."""
    q = question.lower()
    d = re.search(r"dia (\d+)", q)
    if not d or "º dia" in q:
        return None
    m = next((x for x in MACHINES if x in q), None)
    kind = ("pior" if "pior oee" in q or "menor oee" in q else "perda" if "disponibilidade, desempenho" in q
            else "pecas" if "peças" in q else "falha" if "falha" in q else "oee" if "oee" in q else None)
    return (kind, m, int(d.group(1))) if kind else None


def regras(data, q: Q) -> str:
    p = parse(q.text)
    if p is None:
        return "não sei"
    kind, m, d = p
    if d > DAYS:
        return "sem dados"
    if kind == "perda":
        return "não sei"
    return gold_for(data, kind, m, d)


def ingenuo(data, q: Q) -> str:
    p = parse(q.text)
    if p is None:
        return "não sei"
    kind, m, d = p
    d = min(d, DAYS)                                   # never says "no data": uses the last day it has
    if kind == "oee":
        return f"{100 * oee_naive(data, m, d):.1f}"
    if kind == "pior":
        return min(MACHINES, key=lambda x: oee_naive(data, x, d))
    if kind == "perda":
        return "disponibilidade"
    return gold_for(data, kind, m, d)


def assistente(data, q: Q, seed: int = 0) -> str:
    """SIMULATED model-like system: knows the answer, slips at random by question type, rambles sometimes."""
    rng = random.Random(f"{seed}-{q.id}")
    kind = q.asked
    ok = {"pecas": 0.95, "falha": 0.95, "oee": 0.85, "pior": 0.70, "perda": 0.75}[kind]
    if q.gold == "sem dados":
        return "sem dados" if rng.random() < 0.35 else "42"
    value = gold_for(data, kind, q.machine, q.day)
    if rng.random() > ok:
        if kind in ("pecas", "falha", "oee"):
            value = f"{float(value) * rng.choice([0.9, 1.1, 1.3]):.1f}"
        elif kind == "pior":
            value = rng.choice([m for m in MACHINES if m != value])
        else:
            value = rng.choice([k for k in ("disponibilidade", "desempenho", "qualidade") if k != value])
    return f"O valor é {value}." if rng.random() < 0.3 else value


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion. Honest about small samples, unlike mean ± something."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def paired_bootstrap(a: list[bool], b: list[bool], rounds: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in accuracy (a minus b) on the same questions: mean and 95 % interval."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    return sum(a) / n - sum(b) / n, diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]

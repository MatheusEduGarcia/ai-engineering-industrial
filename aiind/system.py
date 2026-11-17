"""Chapter 10: an AI system as an architecture, and the feedback loop around it.

Components come from earlier chapters: parser and rules (ch4), assistant (ch4, SIMULATED), guardrails (ch5), cache (ch9).
Latency and cost per component are assumed constants. Operators' feedback is SIMULATED with known biases, so the
estimators can be checked against the truth. All data is synthetic.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from aiind.corpus import MACHINES
from aiind.oeeqa import (ABSTAIN, DAYS, DIFFICULTY, Q, TEXTS, assistente, correct, gold_for, parse)
from aiind.prompting import guardrails
from aiind.datasets import normalize

# assumed cost per step: (seconds, US$)
COST = {"cache": (0.003, 0.0), "regras": (0.01, 0.0), "assistente": (1.4, 0.0009), "guarda": (0.01, 0.0), "humano": (0.0, 0.0)}
# a new way of asking appears on the floor: "ontem" instead of "dia N" (yesterday = the last day of data)
NOVEL_TEXTS = {"oee": "Como foi o OEE da {m} ontem?", "pecas": "Quantas boas a {m} fez ontem?", "falha": "Quantos minutos a {m} ficou parada por falha ontem?",
               "perda": "Ontem a {m} perdeu mais por disponibilidade, desempenho ou qualidade?"}
ESCALATE = "não sei"


def novel_questions(data, n: int, seed: int = 0) -> list[Q]:
    rng = random.Random(seed); out = []
    for i in range(n):
        kind = rng.choice(list(NOVEL_TEXTS)); m, d = rng.choice(MACHINES), DAYS
        q = Q(10_000 + i, NOVEL_TEXTS[kind].format(m=m, d=d), kind, m, d, 9, gold_for(data, kind, m, d), DIFFICULTY[kind])
        q.asked = kind; q.novel = True
        out.append(q)
    return out


def parse_v2(text: str):
    """Understands every phrasing of the OEE family, including '3º dia' and the new wording."""
    q = text.lower()
    d = re.search(r"dia (\d+)|(\d+)º dia", q)
    if d:
        day = int(d.group(1) or d.group(2))
    elif "ontem" in q:
        day = DAYS
    else:
        return None
    m = next((x for x in MACHINES if x in q), None)
    kind = ("pior" if "pior oee" in q or "menor oee" in q else "perda" if "disponibilidade, desempenho" in q or "disponibilidade, desempenho ou qualidade" in q
            else "pecas" if "peças" in q or "boas" in q else "falha" if "falha" in q else "oee" if "oee" in q else None)
    return (kind, m, day) if kind else None


@dataclass
class Event:
    qid: int
    route: str            # cache | regras | assistente | escalado
    shown: str
    ok: bool              # correct against the gold (abstaining on missing data counts as correct)
    abstained: bool
    latency: float
    cost: float
    blocked: str = ""
    kind: str = ""
    template: int = 0
    novel: bool = False

    @property
    def wrong_shown(self) -> bool:
        return not self.ok and not self.abstained


def _perturb(q: Q, ans: str, rng: random.Random) -> str:
    if q.kind in ("pecas", "falha", "oee"):
        try:
            return f"{float(ans) * rng.choice([0.9, 1.3]):.1f}"
        except ValueError:
            return ans
    pool = list(MACHINES) if q.kind == "pior" else ["disponibilidade", "desempenho", "qualidade"]
    return rng.choice([x for x in pool if x != ans])


class System:
    """cache -> rules -> assistant -> guardrails -> (escalate). Switches let you remove a part and measure it."""

    def __init__(self, data, version: int = 1, use_cache: bool = True, use_rules: bool = True, use_guard: bool = True, novelty_slip: float = 0.25, seed: int = 0, cache_bug: bool = False):
        self.data, self.version, self.use_cache, self.use_rules, self.use_guard = data, version, use_cache, use_rules, use_guard
        self.novelty_slip, self.seed, self.cache_bug = novelty_slip, seed, cache_bug
        self.store: dict = {}

    def _rules(self, q: Q):
        p = (parse_v2 if self.version >= 2 else parse)(q.text)
        if p is None:
            return None
        kind, m, d = p
        if d > DAYS:
            return "sem dados"
        if kind == "perda" and self.version < 2:
            return None
        return gold_for(self.data, kind, m, d)

    def ask(self, q: Q) -> Event:
        lat = cost = 0.0; blocked = ""
        def pay(step):
            nonlocal lat, cost
            lat += COST[step][0]; cost += COST[step][1]
        key = (parse_v2 if self.version >= 2 else parse)(q.text) or normalize(q.text)
        if self.cache_bug and isinstance(key, tuple):
            key = (key[0], key[2])        # a classic bug: the cache key forgets the machine
        if self.use_cache:
            pay("cache")
            if key in self.store:
                return self._event(q, "cache", self.store[key], lat, cost)
        ans = None; route = "regras"
        if self.use_rules:
            pay("regras"); ans = self._rules(q)
        if ans is None:
            route = "assistente"; pay("assistente")
            ans = assistente(self.data, q, self.seed)
            if getattr(q, "novel", False) and random.Random(f"{self.seed}-{q.id}-n").random() < self.novelty_slip:
                ans = _perturb(q, ans if ans not in ABSTAIN else gold_for(self.data, q.asked, q.machine, q.day), random.Random(q.id))
            if self.use_guard:
                pay("guarda"); ans, blocked = guardrails(ans, q)
            if blocked or ans.strip().lower() in ABSTAIN:
                route, ans = "escalado", ESCALATE
        if self.use_cache and route != "escalado":
            self.store[key] = ans
        return self._event(q, route, ans, lat, cost, blocked)

    def _event(self, q, route, ans, lat, cost, blocked=""):
        abst = any(x in ans.lower() for x in ABSTAIN)
        ok = correct(q, ans)
        return Event(q.id, route, ans, ok, abst and not ok, lat, cost, blocked, q.asked, q.template, getattr(q, "novel", False))


def run(system: System, questions: list[Q]) -> list[Event]:
    return [system.ask(q) for q in questions]


def summarize(events: list[Event]) -> dict:
    n = len(events); lat = sorted(e.latency for e in events)
    return {"certo": sum(e.ok for e in events) / n, "escalado": sum(e.abstained for e in events) / n, "erro mostrado": sum(e.wrong_shown for e in events) / n,
            "latência média": sum(lat) / n, "latência p95": lat[int(0.95 * (n - 1))], "custo por mil": 1000 * sum(e.cost for e in events) / n}


# ------------------------------------------------------------------ feedback (SIMULATED operators)
@dataclass
class Signal:
    event: Event
    thumb: str | None      # 'up' | 'down' | None
    reask: bool            # implicit: asked again right after


def operator_feedback(events: list[Event], seed: int = 0, p_down: float = 0.35, p_up: float = 0.05, judge_acc: float = 0.85,
                      reask_bad: float = 0.35, reask_good: float = 0.04) -> list[Signal]:
    """Operators only judge what they were shown. They notice a wrong answer with prob judge_acc (and mistake a right one for wrong with 1 - judge_acc),
    complain much more than they praise, and sometimes just ask again."""
    rng = random.Random(seed); out = []
    for e in events:
        if e.abstained:
            out.append(Signal(e, None, rng.random() < 0.5)); continue       # escalated: the operator calls someone, often rephrases first
        perceived_bad = (rng.random() < judge_acc) if e.wrong_shown else (rng.random() > judge_acc)
        thumb = ("down" if rng.random() < p_down else None) if perceived_bad else ("up" if rng.random() < p_up else None)
        out.append(Signal(e, thumb, rng.random() < (reask_bad if perceived_bad else reask_good)))
    return out


def audit_sample(events: list[Event], n: int, seed: int = 0) -> list[Event]:
    return random.Random(seed).sample(events, n)


def p_chart(counts: list[int], ns: list[int], baseline_weeks: int = 4):
    """Control limits (3 sigma) for a weekly proportion, from the first `baseline_weeks`."""
    p0 = sum(counts[:baseline_weeks]) / sum(ns[:baseline_weeks])
    lim = [3 * math.sqrt(p0 * (1 - p0) / n) for n in ns]
    return p0, [max(0, p0 - l) for l in lim], [p0 + l for l in lim]

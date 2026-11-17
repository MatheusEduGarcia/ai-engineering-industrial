"""
Retrieval-augmented generation and a tool-using agent, over synthetic manuals and the OEE data. Chapter 6.

  * `make_manuals` writes maintenance manuals (specifications, alarms, safety) with facts of known location;
  * chunkers (by paragraph, by section, fixed windows), BM25, embedding search and a hybrid of both;
  * `make_questions` produces questions whose evidence sentence is known, so retrieval can be scored;
  * `Reader` is a SIMULATED reader: it answers right when the evidence is in the context, otherwise it
    invents an answer or says "not found", depending on the instruction;
  * `Agent` runs tool calls with validation, permission, a step limit and human approval for side effects.
    The planner behind it is simulated too, with injectable mistakes.

No real model is called here. Everything is synthetic.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .corpus import MACHINES, PARTS
from .evaluation import FAMILIES, Embedder
from .oeeqa import DAYS, Q, correct as _unused, gold_for, make_data  # noqa: F401

ACTIONS = ["verificar o alinhamento e limpar", "substituir e registrar a troca", "reapertar e testar a vedação", "recalibrar e validar com o líder"]


@dataclass
class Fact:
    kind: str          # torque | intervalo | codigo_peca | alarme
    machine: str
    part: str
    evidence: str      # the exact sentence that holds the answer
    answer: str        # the short gold answer
    extra: str = ""    # alarm code, symptom word


def make_manuals(seed: int = 3):
    """Returns (paragraphs, facts). A paragraph is (id, machine, section, text)."""
    rng = random.Random(seed)
    paras, facts, n = [], [], 0
    codes = rng.sample(range(100, 999), 200)
    ci = 0
    for fam, d in FAMILIES.items():                       # a glossary, as real manuals have
        paras.append((n, "geral", "Glossário", f"Glossário de sintomas: {', '.join(d['sintomas'][:-1])} e {d['sintomas'][-1]} descrevem o mesmo problema ({fam}).")); n += 1
    for m in MACHINES:
        for p in PARTS[m]:
            T, H, PN = rng.randint(5, 80), rng.choice([500, 1000, 2000, 4000]), rng.randint(1000, 9999)
            s1, s2, s3 = (f"O torque de aperto da peça {p} da {m} é de {T} Nm.", f"A troca preventiva da peça {p} da {m} ocorre a cada {H} horas de operação.",
                          f"O código da peça {p} da {m} é PN-{PN}.")
            for sent, kind, ans in ((s1, "torque", f"{T} Nm"), (s2, "intervalo", f"{H} horas"), (s3, "codigo_peca", f"PN-{PN}")):
                paras.append((n, m, f"Especificações da peça {p}", sent)); n += 1
                facts.append(Fact(kind, m, p, sent, ans))
            fits = [f for f, d in FAMILIES.items() if p in d["pecas"]]
            fams = rng.sample(fits, min(2, len(fits)))
            fams += rng.sample([f for f in FAMILIES if f not in fams], 2 - len(fams))
            for fam in fams:
                sint = rng.choice(FAMILIES[fam]["sintomas"]); code = f"E{codes[ci]}"; ci += 1
                sent = f"O alarme {code} da {m} indica {sint} na peça {p}. Ação: {rng.choice(ACTIONS)} a peça {p}."
                paras.append((n, m, f"Alarmes da peça {p}", sent)); n += 1
                facts.append(Fact("alarme", m, p, sent, code, extra=sint))
        P = rng.randint(4, 9)
        paras.append((n, m, "Segurança", f"Antes de qualquer intervenção na {m}, bloquear e etiquetar a energia. A pressão de trabalho da {m} é de {P} bar.")); n += 1
    return paras, facts


def sibling_synonym(sint: str) -> str:
    for d in FAMILIES.values():
        if sint in d["sintomas"]:
            return rng_choice([s for s in d["sintomas"] if s != sint], sint)
    return sint


def rng_choice(xs, key):
    return xs[sum(map(ord, key)) % len(xs)]


@dataclass
class RQ:
    kind: str          # numerica | codigo | sintoma
    text: str
    answer: str
    evidence: str


def make_questions(facts: list[Fact], n: int = 240, seed: int = 8) -> list[RQ]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        f = rng.choice(facts)
        if f.kind == "torque":
            out.append(RQ("numerica", f"Qual o torque de aperto da peça {f.part} da {f.machine}?", f.answer, f.evidence))
        elif f.kind == "intervalo":
            out.append(RQ("numerica", f"De quantas em quantas horas se troca a peça {f.part} da {f.machine}?", f.answer, f.evidence))
        elif f.kind == "codigo_peca":
            out.append(RQ("numerica", f"Qual o código da peça {f.part} da {f.machine}?", f.answer, f.evidence))
        elif rng.random() < 0.4:
            out.append(RQ("codigo", f"O que significa o alarme {f.answer} da {f.machine}?", f.part, f.evidence))
        else:
            out.append(RQ("sintoma", f"Qual alarme da {f.machine} aparece quando a peça {f.part} está {sibling_synonym(f.extra)}?", f.answer, f.evidence))
    return out


# ------------------------------------------------------------------ chunking
def chunk_paragraph(paras, glossary_too: bool = True):
    return [p[3] for p in paras if glossary_too or p[2] != "Glossário"]


def chunk_section(paras):
    sec: dict = {}
    for _, m, s, t in paras:
        sec.setdefault((m, s), []).append(t)
    return [" ".join(v) for v in sec.values()]


def chunk_window(paras, size: int = 40, overlap: int = 8):
    words = " ".join(p[3] for p in paras).split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), size - overlap)]


# ------------------------------------------------------------------ retrieval
def tok(s: str) -> list[str]:
    return re.findall(r"[\wçãõáéíóúâêôà-]+", s.lower())


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.docs = [Counter(tok(c)) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = tok(query)
        out = []
        for d, L in zip(self.docs, self.len):
            s = 0.0
            for w in q:
                if w in d:
                    f = d[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


class Dense:
    """Embedding search: each text is the average of its word vectors (notebook 03), compared by cosine."""

    def __init__(self, chunks, train_texts, dim: int = 16):
        self.emb = Embedder(dim).fit(train_texts)
        self.vecs = [self.emb.text(c) for c in chunks]

    def scores(self, query: str) -> list[float]:
        v = self.emb.text(query)
        return [float(v @ x) for x in self.vecs]


def ranking(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def hybrid(*score_lists: list[float], k: int = 60) -> list[float]:
    """Reciprocal rank fusion: combine rankings, not raw scores (which live on different scales)."""
    fused = [0.0] * len(score_lists[0])
    for sl in score_lists:
        for r, i in enumerate(ranking(sl)):
            fused[i] += 1 / (k + r + 1)
    return fused


def hit(chunks, order, evidence, k):
    return any(evidence in chunks[i] for i in order[:k])


def rank_of(chunks, order, evidence):
    for r, i in enumerate(order, 1):
        if evidence in chunks[i]:
            return r
    return None


# ------------------------------------------------------------------ reader
class Reader:
    """SIMULATED reader. With the evidence in context it extracts the answer; without it, it invents or abstains."""

    def __init__(self, abstain_instruction: bool = True, seed: int = 0):
        self.abstain, self.seed = abstain_instruction, seed

    def answer(self, q: RQ, context: list[str], idx: int = 0) -> str:
        if any(q.evidence in c for c in context):
            return q.answer
        rng = random.Random(f"{self.seed}-{idx}")
        if rng.random() < (0.9 if self.abstain else 0.3):
            return "não encontrei no manual"
        return rng.choice(["42 Nm", "E999", "enchedora", "2000 horas"])


# ------------------------------------------------------------------ tools and agent
@dataclass
class Tool:
    name: str
    fn: object
    params: dict
    side_effect: bool = False


@dataclass
class Trace:
    steps: list = field(default_factory=list)
    answer: str = ""
    status: str = "ok"          # ok | limite de passos | bloqueado
    effects: list = field(default_factory=list)


@dataclass
class Task:
    kind: str           # oee | falha_manual | os
    machine: str
    day: int
    must_contain: list
    side_effect_expected: bool = False


def make_tasks(data, facts, n: int = 60, seed: int = 4) -> list[Task]:
    rng = random.Random(seed)
    alarms = [f for f in facts if f.kind == "alarme"]
    out = []
    for _ in range(n):
        m, d = rng.choice(MACHINES), rng.randint(1, DAYS)
        kind = rng.choice(["oee", "falha_manual", "os"])
        fail = int(gold_for(data, "falha", m, d))
        if kind == "oee":
            out.append(Task(kind, m, d, [gold_for(data, "oee", m, d)]))
        elif kind == "falha_manual":
            al = rng.choice([a for a in alarms if a.machine == m])
            need = [str(fail)] + ([al.answer] if fail > 30 else [])
            out.append(Task(kind, m, d, need))
            out[-1].alarm = al
        else:
            out.append(Task(kind, m, d, [str(fail), "pendente"], side_effect_expected=True))
    return out


class Agent:
    def __init__(self, tools: list[Tool], planner, max_steps: int = 6, allowed: set | None = None, validate: bool = True,
                 approve=lambda tool, args: False):
        self.tools = {t.name: t for t in tools}
        self.planner, self.max_steps, self.allowed, self.validate, self.approve = planner, max_steps, allowed, validate, approve

    def run(self, task: Task) -> Trace:
        tr = Trace()
        for _ in range(self.max_steps):
            act = self.planner(task, tr)
            if act[0] == "final":
                tr.answer = act[1]
                return tr
            _, name, args = act
            obs = self._call(name, args, tr)
            tr.steps.append((name, args, obs))
            if tr.status == "bloqueado":          # a permission error is final: do not let the planner retry around it
                tr.answer = "sem permissão"
                return tr
        tr.status = "limite de passos"
        return tr

    def _call(self, name, args, tr: Trace) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return "erro: ferramenta desconhecida"
        if self.validate:
            if set(args) != set(tool.params):
                return "erro: argumentos inesperados"
            m, d = args.get("maquina"), args.get("dia")
            if m is not None and m not in MACHINES:
                return "erro: máquina desconhecida"
            if self.allowed is not None and m is not None and m not in self.allowed:
                tr.status = "bloqueado"
                return "erro: sem permissão para esta máquina"
            if d is not None and not 1 <= d <= DAYS:
                return f"erro: dia inválido (1 a {DAYS})"
        if tool.side_effect and not self.approve(name, args):
            return "pendente de aprovação humana"
        if tool.side_effect:
            tr.effects.append((name, args))
        return tool.fn(**args)


class SimPlanner:
    """SIMULATED model-as-planner: knows the right plan, but with some probability makes a mistake."""

    def __init__(self, error: float = 0.0, seed: int = 0, self_correct: bool = True):
        self.error, self.seed, self.self_correct = error, seed, self_correct
        self.i = 0

    def __call__(self, task: Task, tr: Trace):
        rng = random.Random(f"{self.seed}-{id(task) % 100000}-{len(tr.steps)}")
        last = tr.steps[-1][2] if tr.steps else ""
        sofreu = len(tr.steps) > 0 and last.startswith("erro")
        wrong = rng.random() < self.error and not (sofreu and self.self_correct)
        day = task.day + 40 if wrong else task.day
        if wrong and rng.random() < 0.3 and tr.steps:                 # loops on the same call
            return ("call", *tr.steps[-1][:2])
        done = [s[0] for s in tr.steps if not s[2].startswith("erro")]
        if task.kind == "oee":
            return ("final", f"OEE {tr.steps[-1][2]}") if "oee" in done else ("call", "oee", {"maquina": task.machine, "dia": day})
        if "falhas" not in done:
            return ("call", "falhas", {"maquina": task.machine, "dia": day})
        fail = int(next(s[2] for s in tr.steps if s[0] == "falhas" and not s[2].startswith("erro")).split()[0]) if any(s[0] == "falhas" and s[2][:1].isdigit() for s in tr.steps) else 0
        if task.kind == "falha_manual":
            if fail > 30 and "manual" not in done:
                return ("call", "manual", {"consulta": f"alarme {task.alarm.extra} {task.machine}"})
            ctx = next((s[2] for s in tr.steps if s[0] == "manual"), "")
            return ("final", f"{fail} min de falha. {ctx}")
        if "abrir_os" not in done:
            return ("call", "abrir_os", {"maquina": task.machine, "descricao": f"{fail} min de falha no dia {task.day}"})
        return ("final", f"{fail} min de falha. ordem {tr.steps[-1][2]}")


# ------------------------------------------------------------------ query expansion with the manual's own glossary
def glossary(paras) -> list[set[str]]:
    out = []
    for _, m, sec, text in paras:
        if sec == "Glossário":
            body = text.split(":")[1].split("(")[0].replace(" descrevem o mesmo problema", "")
            out.append({w.strip() for w in re.split(r",| e ", body) if w.strip()})
    return out


def expand(query: str, gl: list[set[str]]) -> str:
    """Add the glossary's equivalent terms for any symptom word found in the query."""
    q = query.lower()
    extra = [w for g in gl if any(t in q.split() or t in q for t in g) for w in sorted(g)]
    return query + " " + " ".join(extra) if extra else query


# ------------------------------------------------------------------ the tools of the agent
def make_tools(data, retrieve) -> list[Tool]:
    return [
        Tool("oee", lambda maquina, dia: gold_for(data, "oee", maquina, dia), {"maquina": str, "dia": int}),
        Tool("falhas", lambda maquina, dia: f"{gold_for(data, 'falha', maquina, dia)} min", {"maquina": str, "dia": int}),
        Tool("manual", lambda consulta: retrieve(consulta), {"consulta": str}),
        Tool("abrir_os", lambda maquina, descricao: f"OS aberta para {maquina}: {descricao}", {"maquina": str, "descricao": str}, side_effect=True),
    ]


def score_task(task: Task, tr: Trace) -> bool:
    ok = all(x.lower() in tr.answer.lower() for x in task.must_contain) and tr.status == "ok"
    done = [e[0] for e in tr.effects]
    if task.side_effect_expected:
        return ok and not done                 # the order must be PENDING; nothing executed without approval
    return ok and not done

"""
Helpers for chapter 3: how to tell whether a model is any good.

  * information measures: entropy, cross entropy, perplexity, bits per character;
  * text similarity: exact, lexical, and semantic (embeddings built from scratch);
  * AI as a judge: a *simulated* judge with known flaws, to practise measuring a judge
    before trusting one (a real model plugs into the same protocol in chapter 6);
  * comparative evaluation: Elo ratings from pairwise preferences;
  * a strict, independent check of note style, to show what happens when the
    metric you report is the same one you optimised.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from .corpus import ACTIONS, CALLS, MACHINES, PARTS, PLANNED, PROBLEMS, SHORTAGE


# ------------------------------------------------------------------ information measures
def entropy(p: dict[str, float], base: float = 2.0) -> float:
    """Average surprise of a distribution, in bits by default."""
    return -sum(q * math.log(q, base) for q in p.values() if q > 0)


def cross_entropy(p: dict[str, float], q: dict[str, float], base: float = 2.0, floor: float = 1e-12) -> float:
    """Cost, in bits, of coding data that follows p using a model that believes q."""
    return -sum(pp * math.log(max(q.get(t, 0.0), floor), base) for t, pp in p.items() if pp > 0)


def kl(p: dict[str, float], q: dict[str, float], base: float = 2.0) -> float:
    """Extra bits paid for believing q when the truth is p. Never negative."""
    return cross_entropy(p, q, base) - entropy(p, base)


def nll_bits(model, seqs: list[list[str]], floor: float = 1e-12) -> tuple[float, int]:
    """Total negative log2-probability a model gives to sequences, and the number of tokens scored."""
    total, n = 0.0, 0
    for seq in seqs:
        for ctx, tok in model.training_pairs(seq):
            p = model.next_probs(list(ctx)).get(tok, 0.0)
            total -= math.log2(max(p, floor))
            n += 1
    return total, n


def perplexity(model, seqs: list[list[str]], floor: float = 1e-12) -> float:
    bits, n = nll_bits(model, seqs, floor)
    return 2 ** (bits / n)


def bits_per_char(model, seqs: list[list[str]], texts: list[str], floor: float = 1e-12) -> float:
    """Cross entropy per character of the original text, comparable across tokenizers."""
    bits, _ = nll_bits(model, seqs, floor)
    return bits / sum(len(t) for t in texts)


# ------------------------------------------------------------------ similarity
def words(s: str) -> list[str]:
    return s.lower().split()


def exact_match(a: str, b: str) -> float:
    return float(words(a) == words(b))


def token_f1(a: str, b: str) -> float:
    """Word overlap between two texts, as an F1 score (bag of words)."""
    A, B = Counter(words(a)), Counter(words(b))
    common = sum((A & B).values())
    if not common:
        return 0.0
    p, r = common / sum(B.values()), common / sum(A.values())
    return 2 * p * r / (p + r)


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def auc(pos: list[float], neg: list[float]) -> float:
    """Chance that a random positive scores above a random negative (ties count half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# Faults written in several ways: the same fault family, different words for it.
FAMILIES: dict[str, dict[str, list[str]]] = {
    "vazamento": {"sintomas": ["vazando", "vazamento", "pingando", "escorrendo"],
                  "pecas": ["vedação", "mangueira", "válvula", "bico de enchimento", "cilindro hidráulico"]},
    "travamento": {"sintomas": ["travou", "emperrou", "preso", "bloqueado"],
                   "pecas": ["garra", "cabeçote", "correia", "rolamento", "matriz"]},
    "ruído": {"sintomas": ["com ruído", "barulho", "zunindo", "batendo"],
              "pecas": ["rolamento", "motor", "correia", "rolo de cola"]},
    "calor": {"sintomas": ["aquecendo", "esquentando", "superaquecendo", "quente"],
              "pecas": ["motor", "cilindro hidráulico", "válvula"]},
    "leitura": {"sintomas": ["sem leitura", "não lê", "leitura falhando", "oscilando"],
                "pecas": ["sensor de nível", "sensor de tampa", "sensor de etiqueta", "sensor de presença",
                          "sensor de altura", "sensor de segurança"]},
}
TEMPLATES = ["{m} parada {p} {s}", "{p} {s} na {m}", "parada na {m} {p} {s}", "{m} parou {p} {s}", "{m} parada {s} {p}"]


def fault_notes(n: int = 3000, seed: int = 1) -> list[tuple[str, str, str]]:
    """(note, fault family, symptom word). The family is what a careful reader would call the problem."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        fam = rng.choice(list(FAMILIES))
        p = rng.choice(FAMILIES[fam]["pecas"])
        s = rng.choice(FAMILIES[fam]["sintomas"])
        m = rng.choice([x for x in MACHINES if p in PARTS[x]])
        out.append((rng.choice(TEMPLATES).format(m=m, p=p, s=s), fam, s))
    return out


def paraphrase_pairs(n: int = 600, seed: int = 9) -> list[tuple[str, str, bool]]:
    """Pairs of notes: same fault told with different symptom words (True) or different faults (False)."""
    rng = random.Random(seed)
    pool = fault_notes(4000, seed=seed + 100)
    by_fam: dict[str, list] = {}
    for item in pool:
        by_fam.setdefault(item[1], []).append(item)
    pairs = []
    while len(pairs) < n:
        if len(pairs) % 2 == 0:
            fam = rng.choice(list(by_fam))
            a, b = rng.sample(by_fam[fam], 2)
            if a[2] == b[2]:
                continue
            pairs.append((a[0], b[0], True))
        else:
            f1, f2 = rng.sample(list(by_fam), 2)
            pairs.append((rng.choice(by_fam[f1])[0], rng.choice(by_fam[f2])[0], False))
    return pairs


class Embedder:
    """Word vectors from co-occurrence: words that show up in the same kind of note get similar vectors.

    Recipe (the same idea as classic word embeddings): count which words appear in the same note,
    turn counts into positive PMI (how much more often than chance), compress with an SVD.
    A text is the average of its word vectors.
    """

    def __init__(self, dim: int = 12):
        self.dim = dim

    def fit(self, texts: list[str]) -> "Embedder":
        vocab = sorted({w for t in texts for w in words(t)})
        self.vocab, self.idx = vocab, {w: i for i, w in enumerate(vocab)}
        C = np.zeros((len(vocab), len(vocab)))
        for t in texts:
            ids = sorted({self.idx[w] for w in words(t)})
            for i in ids:
                for j in ids:
                    if i != j:
                        C[i, j] += 1
        total, row, col = C.sum(), C.sum(1, keepdims=True), C.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(C * total / (row * col))
        ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        self.W = U[:, :self.dim] * np.sqrt(S[:self.dim])
        return self

    def text(self, s: str) -> np.ndarray:
        vs = [self.W[self.idx[w]] for w in words(s) if w in self.idx]
        if not vs:
            return np.zeros(self.dim)
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def similarity(self, a: str, b: str) -> float:
        return float(self.text(a) @ self.text(b))

    def nearest(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        v = self.W[self.idx[word]]
        n = np.linalg.norm(self.W, axis=1) * (np.linalg.norm(v) or 1)
        sims = (self.W @ v) / np.where(n == 0, 1, n)
        order = np.argsort(-sims)
        return [(self.vocab[i], float(sims[i])) for i in order if self.vocab[i] != word][:k]


# ------------------------------------------------------------------ AI as a judge (simulated)
@dataclass
class Summary:
    note: str       # what the operator wrote
    text: str       # a summary of it, to be judged
    correct: bool   # what a careful person would say
    kind: str


KINDS = ["curta certa", "longa certa", "curta errada", "longa errada", "vaga"]


def summaries(n: int = 500, seed: int = 11) -> list[Summary]:
    """Summaries of failure notes, with known quality, short and long, right and wrong."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        m = rng.choice(MACHINES)
        p = rng.choice(PARTS[m])
        prob = rng.choice(PROBLEMS)
        other = rng.choice([x for x in PARTS[m] if x != p])
        note = f"{m} parada {p} {prob}"
        kind = KINDS[i % len(KINDS)]
        long = "A {m} parou por falha: o componente {c} apresentou o sintoma '{prob}'. Recomenda-se inspecionar o {c} e acionar a manutenção para correção."
        text = {
            "curta certa": f"Falha: {p} {prob} ({m}).",
            "longa certa": long.format(m=m, c=p, prob=prob),
            "curta errada": f"Falha: {other} {prob} ({m}).",
            "longa errada": long.format(m=m, c=other, prob=prob),
            "vaga": f"A {m} apresentou um problema.",
        }[kind]
        out.append(Summary(note, text, kind in ("curta certa", "longa certa"), kind))
    return out


class SimJudge:
    """A stand-in for an LLM judge, with the flaws real ones have: noise, a taste for long answers,
    and a taste for whatever comes first. It is a simulation: the point of the chapter is the
    *protocol* for measuring a judge, which works the same with a real model behind `score`."""

    def __init__(self, noise: float = 0.7, length_bias: float = 1.2, position_bias: float = 0.4, seed: int = 0):
        self.noise, self.length_bias, self.position_bias = noise, length_bias, position_bias
        self.rng = random.Random(seed)

    def score(self, s: Summary) -> float:
        raw = 1 + 2 * s.correct + self.length_bias * min(len(s.text.split()) / 25, 1) + self.rng.gauss(0, self.noise)
        return min(5.0, max(1.0, raw))

    def prefer(self, a: Summary, b: Summary) -> str:
        """'A' or 'B'. The first option gets a small unearned bonus."""
        return "A" if self.score(a) + self.position_bias > self.score(b) else "B"


def debiased_prefer(judge: SimJudge, a: Summary, b: Summary) -> str:
    """Ask in both orders. If the two answers disagree, the honest verdict is a tie."""
    first, second = judge.prefer(a, b), judge.prefer(b, a)
    if first == "A" and second == "B":
        return "A"
    if first == "B" and second == "A":
        return "B"
    return "tie"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Agreement between two raters beyond what chance would give. 1 is perfect, 0 is chance."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# ------------------------------------------------------------------ strict note style (independent check)
def _strict_patterns() -> dict[str, re.Pattern]:
    alt = lambda xs: "|".join(re.escape(x) for x in xs)
    out = {}
    for m in MACHINES:
        fail = (rf"parada (?P<p>{alt(PARTS[m])}) (?:{alt(PROBLEMS)})(?: (?:{alt(CALLS)}))?"
                rf"(?: (?P=p) (?:{alt(ACTIONS)}))?")
        out[m] = re.compile(rf"^{re.escape(m)} (?:{fail}|{alt(PLANNED)}|parada (?:{alt(SHORTAGE)})|voltou a operar)$")
    return out


_STRICT = _strict_patterns()


def strict_note(note: str) -> bool:
    """True if the note follows one of the shapes real operators use, with parts of the right machine.
    Stricter than `structured.coherent_note`, and written separately on purpose."""
    first = note.split()[0] if note.split() else ""
    return first in _STRICT and bool(_STRICT[first].match(note))


# ------------------------------------------------------------------ Elo
def elo(matches: list[tuple[str, str, float]], k: float = 16.0, start: float = 1000.0) -> dict[str, float]:
    """Ratings from pairwise results (a, b, score of a: 1 win, 0.5 tie, 0 loss). Total rating is conserved."""
    r: dict[str, float] = {}
    for a, b, s in matches:
        ra, rb = r.setdefault(a, start), r.setdefault(b, start)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        r[a], r[b] = ra + k * (s - ea), rb + k * ((1 - s) - (1 - ea))
    return r


def elo_spread(matches: list[tuple[str, str, float]], rounds: int = 200, seed: int = 0, k: float = 16.0):
    """Elo depends on the order of the matches. Shuffle many times: mean and 5th-95th percentile per player."""
    rng = random.Random(seed)
    runs: dict[str, list[float]] = {}
    for _ in range(rounds):
        m = matches[:]
        rng.shuffle(m)
        for name, v in elo(m, k).items():
            runs.setdefault(name, []).append(v)
    return {n: (float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))) for n, v in runs.items()}

"""Chapter 8: dataset engineering for stop reports (synthetic, no real plant data).

Raw reports come with the problems real ones have: exact and near duplicates, junk entries,
typos, wrong labels and uneven coverage. Everything here is deterministic given the seed.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
import zlib
from dataclasses import dataclass, field

import numpy as np

from aiind.evaluation import FAMILIES, MACHINES, PARTS, TEMPLATES

FAMS = list(FAMILIES)
CONFUSABLE = {"vazamento": "calor", "calor": "vazamento", "ruído": "travamento", "travamento": "ruído", "leitura": "travamento"}
JUNK = ["teste", "", "asdf", "ok", "-", "xx", "ver depois", "???"]
PREFIXES = ["obs:", "turno b -", "operador:", "urgente", "ref."]


@dataclass
class Rec:
    id: int
    text: str
    label: str            # what the annotator wrote (may be wrong)
    true: str             # what the fault really was
    group: int            # id of the original report (duplicates share it)
    kind: str = "original"  # original | exact | near | junk


def make_note(rng: random.Random, n_symptoms: int = 4, weights=None) -> tuple[str, str]:
    fam = rng.choices(FAMS, weights=weights)[0] if weights else rng.choice(FAMS)
    p = rng.choice(FAMILIES[fam]["pecas"])
    s = rng.choice(FAMILIES[fam]["sintomas"][:n_symptoms])
    m = rng.choice([x for x in MACHINES if p in PARTS[x]])
    text = rng.choice(TEMPLATES).format(m=m, p=p, s=s)
    if rng.random() < 0.9:  # the free-text detail that makes real reports different from each other
        text += " " + rng.choice(["desde {h}h{mm:02d}", "lote {l}", "pressão {b} bar", "temp {t} graus", "turno {u}", "linha {n}"]).format(
            h=rng.randint(0, 23), mm=rng.randint(0, 59), l=rng.randint(1000, 9999), b=rng.randint(1, 9), t=rng.randint(40, 95),
            u=rng.choice("ABC"), n=rng.randint(1, 4))
    return text, fam


def corrupt(text: str, rng: random.Random, strength: float = 1.0) -> str:
    """Operator noise: case, spaces, a typo, a prefix, trailing punctuation."""
    if rng.random() < 0.3 * strength:
        text = text.upper() if rng.random() < 0.5 else text.capitalize()
    if rng.random() < 0.3 * strength:
        w = text.split()
        i = rng.randrange(len(w))
        if len(w[i]) > 3:
            j = rng.randrange(len(w[i]) - 1)
            w[i] = w[i][:j] + w[i][j + 1] + w[i][j] + w[i][j + 2:]
        text = " ".join(w)
    if rng.random() < 0.2 * strength:
        text = rng.choice(PREFIXES) + " " + text
    if rng.random() < 0.3 * strength:
        text = text + rng.choice([".", "!", " ..", ";"])
    if rng.random() < 0.2 * strength:
        text = text.replace(" ", "  ", 1)
    return text


def raw_reports(n: int = 400, seed: int = 0, dup_rate: float = 0.25, junk_rate: float = 0.03, label_noise: float = 0.10,
                n_symptoms: int = 4, weights=None, noise: float = 1.0) -> list[Rec]:
    """n distinct reports plus duplicates and junk; labels flipped to a confusable family with prob label_noise."""
    rng = random.Random(seed)
    recs: list[Rec] = []
    for i in range(n):
        t, fam = make_note(rng, n_symptoms, weights)
        recs.append(Rec(len(recs), corrupt(t, rng, noise), fam, fam, i, "original"))
    originals = list(recs)
    for _ in range(int(n * dup_rate)):
        o = rng.choice(originals)
        if rng.random() < 0.5:
            recs.append(Rec(len(recs), o.text, o.label, o.true, o.group, "exact"))
        else:
            base = re.sub(r"[.!;]+$", "", o.text.strip().lower())
            recs.append(Rec(len(recs), corrupt(base, rng, 1.2), o.label, o.true, o.group, "near"))
    for _ in range(int(n * junk_rate)):
        recs.append(Rec(len(recs), rng.choice(JUNK), rng.choice(FAMS), "?", -1 - len(recs), "junk"))
    for r in recs:
        if r.kind != "junk" and rng.random() < label_noise:
            r.label = CONFUSABLE[r.true]
    rng.shuffle(recs)
    return recs


# ------------------------------------------------------------------ normalization and duplicates
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def shingles(text: str, k: int = 4) -> set[str]:
    t = " " + normalize(text) + " "
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def minhash(sh: set[str], num_perm: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2**31 - 1, num_perm).astype(np.int64)
    b = rng.randint(0, 2**31 - 1, num_perm).astype(np.int64)
    h = np.array([zlib.crc32(s.encode()) for s in sh], dtype=np.int64)
    return ((a[:, None] * h[None, :] + b[:, None]) % 2147483647).min(axis=1)


def est_jaccard(sa: np.ndarray, sb: np.ndarray) -> float:
    return float((sa == sb).mean())


def dedup_exact(recs: list[Rec]) -> list[Rec]:
    seen, out = set(), []
    for r in recs:
        if r.text not in seen:
            seen.add(r.text)
            out.append(r)
    return out


def dedup_normalized(recs: list[Rec]) -> list[Rec]:
    seen, out = set(), []
    for r in recs:
        k = normalize(r.text)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def near_clusters(recs: list[Rec], threshold: float = 0.7, num_perm: int = 64, bands: int = 16) -> list[int]:
    """Cluster id per record. Candidates come from MinHash bands (LSH); a pair merges when the exact Jaccard >= threshold."""
    shs = [shingles(r.text) for r in recs]
    sig = [minhash(s, num_perm) for s in shs]
    rows = num_perm // bands
    buckets: dict[tuple, list[int]] = {}
    for i, s in enumerate(sig):
        for b in range(bands):
            buckets.setdefault((b, tuple(s[b * rows:(b + 1) * rows])), []).append(i)
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for ids in buckets.values():
        for k in range(1, len(ids)):
            i, j = ids[0], ids[k]
            if find(i) != find(j) and jaccard(shs[i], shs[j]) >= threshold:
                parent[find(j)] = find(i)
    return [find(i) for i in range(len(recs))]


def dedup_near(recs: list[Rec], threshold: float = 0.7) -> list[Rec]:
    cl = near_clusters(recs, threshold)
    seen, out = set(), []
    for r, c in zip(recs, cl):
        if c not in seen:
            seen.add(c)
            out.append(r)
    return out


# ------------------------------------------------------------------ cleaning rules
def clean_rules(recs: list[Rec], min_words: int = 3) -> tuple[list[Rec], dict[str, int]]:
    out, dropped = [], {"vazio ou curto": 0, "sem palavra de falha": 0}
    sintomas = {s for f in FAMILIES.values() for s in f["sintomas"]}
    for r in recs:
        n = normalize(r.text)
        if len(n.split()) < min_words:
            dropped["vazio ou curto"] += 1
        elif not any(normalize(s) in n for s in sintomas):
            dropped["sem palavra de falha"] += 1
        else:
            out.append(r)
    return out, dropped


# ------------------------------------------------------------------ small classifier (naive Bayes) and a memorizer
def tok(text: str) -> list[str]:
    return normalize(text).split()


class NB:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def fit(self, texts, labels):
        self.classes = sorted(set(labels))
        self.vocab = {w: i for i, w in enumerate(sorted({w for t in texts for w in tok(t)}))}
        V, C = len(self.vocab), len(self.classes)
        cnt = np.zeros((C, V)); prior = np.zeros(C)
        ci = {c: i for i, c in enumerate(self.classes)}
        for t, y in zip(texts, labels):
            prior[ci[y]] += 1
            for w in tok(t):
                cnt[ci[y], self.vocab[w]] += 1
        self.logp = np.log((cnt + self.alpha) / (cnt.sum(1, keepdims=True) + self.alpha * (V + 1)))
        self.unk = np.log(self.alpha / (cnt.sum(1) + self.alpha * (V + 1)))
        self.logprior = np.log((prior + 1) / (prior.sum() + C))
        return self

    def proba(self, texts) -> np.ndarray:
        out = []
        for t in texts:
            s = self.logprior.copy()
            for w in tok(t):
                s += self.logp[:, self.vocab[w]] if w in self.vocab else self.unk
            s = np.exp(s - s.max())
            out.append(s / s.sum())
        return np.array(out)

    def predict(self, texts) -> list[str]:
        return [self.classes[i] for i in self.proba(texts).argmax(1)]

    def accuracy(self, texts, labels) -> float:
        return float(np.mean([p == y for p, y in zip(self.predict(texts), labels)]))


class Memorizer:
    """1-nearest-neighbour by character shingles: a model that does little more than remember."""

    def fit(self, texts, labels):
        self.sh = [shingles(t) for t in texts]
        self.y = list(labels)
        return self

    def predict(self, texts):
        return [self.y[int(np.argmax([jaccard(shingles(t), s) for s in self.sh]))] for t in texts]

    def accuracy(self, texts, labels):
        return float(np.mean([p == y for p, y in zip(self.predict(texts), labels)]))


# ------------------------------------------------------------------ splits and leakage
def split_random(recs: list[Rec], frac: float = 0.3, seed: int = 0):
    rng = random.Random(seed)
    idx = list(range(len(recs))); rng.shuffle(idx)
    k = int(len(recs) * frac)
    return [recs[i] for i in idx[k:]], [recs[i] for i in idx[:k]]


def split_by_group(recs: list[Rec], frac: float = 0.3, seed: int = 0, groups: list[int] | None = None):
    groups = groups if groups is not None else [r.group for r in recs]
    rng = random.Random(seed)
    ug = sorted(set(groups)); rng.shuffle(ug)
    test_g = set(ug[:int(len(ug) * frac)])
    tr = [r for r, g in zip(recs, groups) if g not in test_g]
    te = [r for r, g in zip(recs, groups) if g in test_g]
    return tr, te


def leakage(train: list[Rec], test: list[Rec], threshold: float = 0.7) -> float:
    """Share of test reports with a near-duplicate in train."""
    tr = [shingles(r.text) for r in train]
    return float(np.mean([max(jaccard(shingles(r.text), s) for s in tr) >= threshold for r in test]))


# ------------------------------------------------------------------ labels
def flag_suspects(texts, labels, k: int = 5, seed: int = 0, thr: float = 0.25) -> list[int]:
    """Out-of-fold probability of the written label below thr marks the report for review."""
    idx = list(range(len(texts))); random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    out = []
    for f in folds:
        fs = set(f)
        tr = [i for i in idx if i not in fs]
        m = NB().fit([texts[i] for i in tr], [labels[i] for i in tr])
        P = m.proba([texts[i] for i in f])
        for row, i in zip(P, f):
            if labels[i] in m.classes and row[m.classes.index(labels[i])] < thr:
                out.append(i)
    return sorted(out)


def annotate(true: list[str], seed: int, error: float = 0.1) -> list[str]:
    """One annotator: right most of the time, otherwise picks the confusable family."""
    rng = random.Random(seed)
    return [CONFUSABLE.get(t, t) if rng.random() < error else t for t in true]


# ------------------------------------------------------------------ synthetic data
STYLE_B = ["falha: {p} {s} ({m})", "{m}: {p} {s}", "relato {m} / {p} / {s}"]


def synth(n: int, seed: int = 0, symptoms: int = 4, styles: str = "real", pairing: str = "valid"):
    """Template-based synthetic reports. pairing='random' mixes parts and symptoms of different families (an invented fault)."""
    rng = random.Random(seed)
    tpl = TEMPLATES if styles == "real" else STYLE_B
    out = []
    for _ in range(n):
        fam = rng.choice(FAMS)
        fam_p = fam if pairing == "valid" else rng.choice(FAMS)
        p = rng.choice(FAMILIES[fam_p]["pecas"])
        s = rng.choice(FAMILIES[fam]["sintomas"][:symptoms])
        m = rng.choice([x for x in MACHINES if p in PARTS[x]])
        out.append((rng.choice(tpl).format(m=m, p=p, s=s), fam))
    return out


def diversity(texts: list[str]) -> dict[str, float]:
    toks = [w for t in texts for w in tok(t)]
    bi = [(a, b) for t in texts for a, b in zip(tok(t), tok(t)[1:])]
    return {"textos distintos": len({normalize(t) for t in texts}) / len(texts), "bigramas distintos": len(set(bi)) / max(1, len(bi)),
            "vocabulário": len(set(toks))}


def coverage_table(recs: list[Rec]) -> dict[tuple[str, str], int]:
    """(family, symptom word) -> count, using the true family."""
    sintomas = {s: f for f, d in FAMILIES.items() for s in d["sintomas"]}
    cnt: dict[tuple[str, str], int] = {(f, s): 0 for f, d in FAMILIES.items() for s in d["sintomas"]}
    for r in recs:
        n = normalize(r.text)
        for s, f in sintomas.items():
            if normalize(s) in n:
                cnt[(f, s)] += 1
    return cnt


def held_out(n: int = 600, seed: int = 999):
    rng = random.Random(seed)
    out = [make_note(rng) for _ in range(n)]
    return [corrupt(t, rng) for t, _ in out], [f for _, f in out]

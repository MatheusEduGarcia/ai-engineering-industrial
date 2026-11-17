"""
From a free-text stop note to a JSON record the OEE system can store.

    "enchedora parada vedação vazando chamado manutenção"
    -> {"maquina": "enchedora", "peca": "vedação", "problema": "vazando", "categoria": "falha"}

Three ways to get structured output out of a language model, compared in notebook 02:

  1. ask and hope: generate freely, then try to parse;
  2. validate and retry: parse, and sample again when it fails;
  3. constrained decoding: at every step, only tokens the schema allows can be sampled.

The model here is the n-gram of chapter 1, trained on (note => JSON) lines. It is far
too small to *read* the note well, which is the point: structure and correctness are
different things, and constrained decoding only guarantees the first.
"""
from __future__ import annotations

import json
import random

from .corpus import ACTIONS, CALLS, MACHINES, PARTS, PLANNED, PROBLEMS, SHORTAGE
from .tinylm import EOS

CATEGORIES = ["falha", "planejada", "falta", "retorno"]
NONE = "nenhum"
FIELDS = ["maquina", "peca", "problema", "categoria"]
SEP = "=>"


def labeled(n: int = 2000, seed: int = 42) -> list[tuple[str, dict]]:
    """Notes with the record a careful person would extract from them."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        m = rng.choice(MACHINES)
        p = rng.choice(PARTS[m])
        r = rng.random()
        if r < 0.55:
            prob = rng.choice(PROBLEMS)
            s = f"{m} parada {p} {prob}"
            if rng.random() < 0.6:
                s += f" {rng.choice(CALLS)}"
            if rng.random() < 0.5:
                s += f" {p} {rng.choice(ACTIONS)}"
            rec = {"maquina": m, "peca": p, "problema": prob, "categoria": "falha"}
        elif r < 0.8:
            s, rec = f"{m} {rng.choice(PLANNED)}", {"maquina": m, "peca": NONE, "problema": NONE, "categoria": "planejada"}
        elif r < 0.95:
            s, rec = f"{m} parada {rng.choice(SHORTAGE)}", {"maquina": m, "peca": NONE, "problema": NONE, "categoria": "falta"}
        else:
            s, rec = f"{m} voltou a operar", {"maquina": m, "peca": NONE, "problema": NONE, "categoria": "retorno"}
        out.append((s, rec))
    return out


# ------------------------------------------------------------------ JSON as tokens
def json_tokens(rec: dict) -> list[str]:
    """{"maquina": "x", ...} as word-like tokens. Multi-word values are joined with '_'."""
    toks = ["{"]
    for i, f in enumerate(FIELDS):
        toks += [f'"{f}":', '"' + rec[f].replace(" ", "_") + '"']
        toks.append("," if i < len(FIELDS) - 1 else "}")
    return toks


def parse(tokens: list[str]) -> dict | None:
    """Turn generated tokens back into a dict. None when it is not valid JSON."""
    text = " ".join(tokens).replace('":', '":').replace("_", " ")
    try:
        d = json.loads(text)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def valid(rec: dict | None) -> list[str]:
    """Problems with a record against the schema. Empty means valid."""
    if rec is None:
        return ["not JSON"]
    errs = [f"missing {f}" for f in FIELDS if f not in rec]
    if errs:
        return errs
    if rec["maquina"] not in MACHINES:
        errs.append(f"unknown machine {rec['maquina']!r}")
    elif rec["peca"] != NONE and rec["peca"] not in PARTS[rec["maquina"]]:
        errs.append(f"{rec['peca']!r} is not a part of {rec['maquina']}")
    if rec["problema"] != NONE and rec["problema"] not in PROBLEMS:
        errs.append(f"unknown problem {rec['problema']!r}")
    if rec["categoria"] not in CATEGORIES:
        errs.append(f"unknown category {rec['categoria']!r}")
    return errs


def training_sequences(data: list[tuple[str, dict]]) -> list[list[str]]:
    return [s.split() + [SEP] + json_tokens(r) for s, r in data]


def schema_allowed(prompt_len: int):
    """A function for `sampling.generate(allowed=...)`: what the schema permits next.

    It walks the JSON being built and, at each slot, offers only the legal tokens.
    The part slot depends on the machine chosen: a schema-aware constraint.
    """
    q = lambda v: '"' + v.replace(" ", "_") + '"'

    def allowed(tokens: list[str]):
        gen = tokens[prompt_len:]
        i = len(gen)
        if i == 0:
            return {"{"}
        if i == 13:
            return {EOS}
        pos = (i - 1) % 3            # 0: key, 1: value, 2: separator
        field = FIELDS[(i - 1) // 3]
        if pos == 0:
            return {f'"{field}":'}
        if pos == 2:
            return {"," if field != FIELDS[-1] else "}"}
        if field == "maquina":
            return {q(m) for m in MACHINES}
        if field == "peca":
            m = gen[2].strip('"')
            return {q(p) for p in PARTS.get(m, [])} | {q(NONE)}
        if field == "problema":
            return {q(p) for p in PROBLEMS} | {q(NONE)}
        return {q(c) for c in CATEGORIES}
    return allowed


def rule_extract(note: str) -> dict:
    """A plain dictionary-matching extractor: the baseline any model must beat."""
    m = next((x for x in MACHINES if note.startswith(x)), None) or "?"
    p = next((x for x in sorted(PARTS.get(m, []), key=len, reverse=True) if x in note), NONE)
    prob = next((x for x in PROBLEMS if x in note), NONE)
    if any(x in note for x in PLANNED):
        cat = "planejada"
    elif "falta de" in note:
        cat = "falta"
    elif "voltou" in note:
        cat = "retorno"
    else:
        cat = "falha"
    return {"maquina": m, "peca": p, "problema": prob, "categoria": cat}


def coherent_note(note: str) -> bool:
    """Verifier for generated notes: known machine, only its own parts, not rambling.

    Parts are matched as whole words, so "garra" is not found inside "garrafa".
    """
    words = note.split()
    if not words or words[0] not in MACHINES or len(words) > 12:
        return False
    padded = f" {note} "
    foreign = [p for m, ps in PARTS.items() if m != words[0] for p in ps
               if f" {p} " in padded and p not in PARTS[words[0]]]
    return not foreign

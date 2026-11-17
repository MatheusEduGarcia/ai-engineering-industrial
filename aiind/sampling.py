"""
Sampling strategies on any next-token distribution {token: probability}.

The model decides the probabilities; sampling decides what to do with them.
Same model, different sampling, very different behaviour.
"""
from __future__ import annotations

import math
import random


def apply_temperature(probs: dict[str, float], t: float) -> dict[str, float]:
    if t == 0:
        best = max(probs, key=probs.get)
        return {k: float(k == best) for k in probs}
    logits = {k: math.log(max(p, 1e-12)) / t for k, p in probs.items()}
    m = max(logits.values())
    z = sum(math.exp(v - m) for v in logits.values())
    return {k: math.exp(v - m) / z for k, v in logits.items()}


def top_k(probs: dict[str, float], k: int) -> dict[str, float]:
    keep = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:k])
    z = sum(keep.values())
    return {t: p / z for t, p in keep.items()}


def top_p(probs: dict[str, float], p: float) -> dict[str, float]:
    """Keep the smallest set of most likely tokens whose probabilities add up to at least p."""
    keep, acc = {}, 0.0
    for t, q in sorted(probs.items(), key=lambda kv: -kv[1]):
        keep[t] = q
        acc += q
        if acc >= p:
            break
    z = sum(keep.values())
    return {t: q / z for t, q in keep.items()}


def sample(probs: dict[str, float], rng: random.Random) -> str:
    toks = list(probs)
    return rng.choices(toks, [probs[t] for t in toks])[0]


def generate(model, rng: random.Random, temperature: float = 1.0, k: int | None = None, p: float | None = None,
             prompt: list[str] | None = None, max_len: int = 25, allowed=None) -> tuple[list[str], float]:
    """Generate tokens with any combination of strategies. Returns (tokens, sum of log-probabilities).

    `allowed(tokens_so_far) -> set | None` restricts the next token (constrained decoding).
    The log-probability is the model's own, before temperature and filtering.
    """
    from .tinylm import EOS
    out, logp = list(prompt or []), 0.0
    while len(out) < max_len:
        raw = model.next_probs(out)
        probs = raw
        if allowed is not None:
            ok = allowed(out)
            if ok is not None:
                probs = {t: q for t, q in probs.items() if t in ok}
                z = sum(probs.values())
                probs = {t: q / z for t, q in probs.items()}
        probs = apply_temperature(probs, temperature)
        if k:
            probs = top_k(probs, k)
        if p:
            probs = top_p(probs, p)
        nxt = sample(probs, rng)
        logp += math.log(max(raw.get(nxt, 1e-12), 1e-12))
        if nxt == EOS:
            break
        out.append(nxt)
    return out, logp

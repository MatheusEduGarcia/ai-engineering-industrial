"""Chapter 9: inference optimization, with numbers you can run or re-derive.

Real code: a tiny decoder with and without a KV cache (identical outputs, measured time), speculative decoding that
keeps the target distribution exactly, response caches. Simulated: the serving queue (batching policies), with step
times built from the roofline arithmetic. Hardware figures are illustrative orders of magnitude: replace them with yours.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from aiind.datasets import jaccard, normalize, shingles
from aiind.foundation import softmax


# ------------------------------------------------------------------ a tiny decoder, with and without KV cache
class TinyDecoder:
    def __init__(self, vocab: int = 60, d: int = 64, heads: int = 4, layers: int = 3, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.V, self.d, self.H, self.L, self.dh = vocab, d, heads, layers, d // heads
        s = 1 / math.sqrt(d)
        self.emb = rng.randn(vocab, d) * s * 3
        self.layers = [{k: rng.randn(*shape) * s for k, shape in
                        {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d), "w1": (d, 4 * d), "w2": (4 * d, d)}.items()} for _ in range(layers)]

    def pos(self, n: int, start: int = 0) -> np.ndarray:
        i = np.arange(start, start + n)[:, None]; j = np.arange(self.d)[None, :]
        ang = i / (10000 ** (2 * (j // 2) / self.d))
        return np.where(j % 2 == 0, np.sin(ang), np.cos(ang))

    @staticmethod
    def norm(x):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)

    def _split(self, x):  # (n, d) -> (H, n, dh)
        return x.reshape(x.shape[0], self.H, self.dh).transpose(1, 0, 2)

    def _block(self, x, lay, past_k=None, past_v=None):
        h = self.norm(x)
        q, k, v = self._split(h @ lay["q"]), self._split(h @ lay["k"]), self._split(h @ lay["v"])
        if past_k is not None:
            k, v = np.concatenate([past_k, k], 1), np.concatenate([past_v, v], 1)
        att = q @ k.transpose(0, 2, 1) / math.sqrt(self.dh)
        n_q, n_k = q.shape[1], k.shape[1]
        mask = np.triu(np.ones((n_q, n_k)), k=n_k - n_q + 1).astype(bool)   # a query sees itself and the past
        att = softmax(np.where(mask, -1e9, att), axis=-1)
        out = (att @ v).transpose(1, 0, 2).reshape(n_q, self.d) @ lay["o"]
        x = x + out
        x = x + np.maximum(self.norm(x) @ lay["w1"], 0) @ lay["w2"]
        return x, k, v

    def logits_full(self, tokens: list[int]) -> np.ndarray:
        """Recompute everything for the whole sequence; logits of the last position."""
        x = self.emb[tokens] + self.pos(len(tokens))
        for lay in self.layers:
            x, _, _ = self._block(x, lay)
        return self.norm(x[-1]) @ self.emb.T

    def prefill(self, tokens: list[int]):
        x = self.emb[tokens] + self.pos(len(tokens)); cache = []
        for lay in self.layers:
            x, k, v = self._block(x, lay); cache.append((k, v))
        return self.norm(x[-1]) @ self.emb.T, cache

    def step(self, token: int, cache, position: int):
        x = self.emb[[token]] + self.pos(1, position); new = []
        for lay, (pk, pv) in zip(self.layers, cache):
            x, k, v = self._block(x, lay, pk, pv); new.append((k, v))
        return self.norm(x[-1]) @ self.emb.T, new

    def generate(self, prompt: list[int], n: int, cache: bool = True) -> list[int]:
        out = list(prompt)
        if cache:
            lg, kv = self.prefill(out)
            for _ in range(n):
                nxt = int(lg.argmax()); out.append(nxt)
                lg, kv = self.step(nxt, kv, len(out) - 1)
        else:
            for _ in range(n):
                out.append(int(self.logits_full(out).argmax()))
        return out


def flops_decode(params: float, tokens: int, context: int | None = None) -> float:
    """About 2 FLOPs per parameter per token (attention over the context is extra and ignored here)."""
    return 2 * params * tokens


def flops_no_cache(params: float, prompt: int, new: int) -> float:
    """Without a cache, every new token reprocesses the whole sequence."""
    return sum(2 * params * (prompt + i) for i in range(new))


def kv_cache_bytes(layers: int, kv_heads: int, d_head: int, tokens: int, bytes_per: int = 2) -> float:
    return 2 * layers * kv_heads * d_head * tokens * bytes_per   # K and V


# ------------------------------------------------------------------ roofline arithmetic
def decode_tps(params: float, bits: int, bandwidth_gbs: float, batch: int = 1) -> float:
    """Decoding reads all the weights once per step, shared by the batch: tokens/s is bound by memory bandwidth."""
    step_s = (params * bits / 8) / (bandwidth_gbs * 1e9)
    return batch / step_s


def prefill_seconds(params: float, tokens: int, tflops: float, utilization: float = 0.5) -> float:
    return 2 * params * tokens / (tflops * 1e12 * utilization)


# ------------------------------------------------------------------ serving simulation
def make_requests(rate: float, seconds: float, seed: int = 0, out_mean: int = 80):
    rng = random.Random(seed); t, reqs = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t > seconds:
            return reqs
        reqs.append((t, max(8, min(400, int(rng.lognormvariate(math.log(out_mean), 0.8))))))


def simulate_serving(reqs, policy: str, max_batch: int = 16, t_weights: float = 0.007, t_seq: float = 0.0004, t_prefill: float = 0.03,
                     wait_window: float = 0.05):
    """Step time = reading the weights (fixed) + a bit per sequence. Returns per-request latency (s) and total tokens/s.
    policy: 'one' (one at a time), 'static' (fill a batch, run until the longest ends), 'continuous' (join and leave every step)."""
    reqs = list(reqs); n = len(reqs); lat = [0.0] * n
    clock, i, done_tokens = 0.0, 0, 0
    queue: list[int] = []
    active: list[list] = []   # [idx, remaining]
    cap = 1 if policy == "one" else max_batch
    while i < n or queue or active:
        while i < n and reqs[i][0] <= clock:
            queue.append(i); i += 1
        if policy == "static":
            if not active:
                if not queue:
                    clock = max(clock, reqs[i][0]); continue
                deadline = reqs[queue[0]][0] + wait_window
                if len(queue) < cap and i < n and clock < deadline:
                    clock = min(reqs[i][0], deadline); continue
                batch = [queue.pop(0) for _ in range(min(cap, len(queue)))]
                clock += t_prefill * len(batch)
                active = [[j, reqs[j][1]] for j in batch]
                longest = max(r for _, r in active)
                for _s in range(longest):
                    clock += t_weights + t_seq * len(active)
                    for a in active:
                        if a[1] > 0:
                            a[1] -= 1; done_tokens += 1
                            if a[1] == 0:
                                lat[a[0]] = clock - reqs[a[0]][0]
                active = []
            continue
        while queue and len(active) < cap:
            j = queue.pop(0); clock += t_prefill; active.append([j, reqs[j][1]])
        if not active:
            clock = max(clock, reqs[i][0]); continue
        clock += t_weights + t_seq * len(active)
        for a in active:
            a[1] -= 1; done_tokens += 1
        for a in [a for a in active if a[1] == 0]:
            lat[a[0]] = clock - reqs[a[0]][0]
        active = [a for a in active if a[1] > 0]
    return lat, done_tokens / clock


# ------------------------------------------------------------------ speculative decoding
def _sample(p: dict, rng: random.Random) -> str:
    toks = list(p)
    return rng.choices(toks, [p[t] for t in toks])[0]


def speculative(target, draft, rng: random.Random, n: int, k: int = 4, prompt: list[str] | None = None):
    """Draft proposes k tokens, target checks them in one pass; accept with prob min(1, p/q), else resample from (p-q)+.
    The output distribution is exactly the target's. Returns (tokens, target passes)."""
    out, passes = list(prompt or []), 0
    while len(out) < (len(prompt or []) + n):
        prop, qs, ctx = [], [], list(out)
        for _ in range(k):
            q = draft.next_probs(ctx); x = _sample(q, rng); prop.append(x); qs.append(q); ctx.append(x)
        passes += 1                           # one target pass scores all k+1 positions
        ctx = list(out); accepted = 0
        for x, q in zip(prop, qs):
            p = target.next_probs(ctx)
            if rng.random() < min(1.0, p.get(x, 0) / max(q.get(x, 1e-12), 1e-12)):
                out.append(x); ctx.append(x); accepted += 1
            else:
                resid = {t: max(0.0, p[t] - q.get(t, 0.0)) for t in p}
                z = sum(resid.values())
                out.append(_sample({t: v / z for t, v in resid.items()} if z > 0 else p, rng)); break
        else:
            out.append(_sample(target.next_probs(ctx), rng))   # all accepted: a free token from the target
    return out[:len(prompt or []) + n], passes


def expected_tokens_per_pass(alpha: float, k: int) -> float:
    return (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1


def speculative_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """Tokens per unit of time against plain decoding: (tokens per pass) / (one target pass + k draft steps)."""
    return expected_tokens_per_pass(alpha, k) / (1 + k * cost_ratio)


# ------------------------------------------------------------------ caches
def query_stream(n: int, seed: int = 0, machines=None, days: int = 14):
    """Stream of OEE questions (text, key) where paraphrases of the same (kind, machine, day) share the key."""
    from aiind.oeeqa import TEXTS
    from aiind.evaluation import MACHINES
    rng = random.Random(seed); machines = machines or list(MACHINES)[:5]
    kinds = ["pecas", "falha", "oee", "perda"]
    pool = [(k, m, d) for k in kinds for m in machines for d in range(1, days + 1)]
    w = [1 / (i + 1) ** 0.8 for i in range(len(pool))]
    out = []
    for _ in range(n):
        k, m, d = rng.choices(pool, w)[0]
        out.append((rng.choice(TEXTS[k]).format(m=m, d=d), (k, m, d)))
    return out


def entities(text: str) -> tuple:
    from aiind.evaluation import MACHINES
    n = normalize(text)
    return (tuple(m for m in MACHINES if normalize(m) in n), tuple(sorted(set(__import__("re").findall(r"\b\d+\b", n)))))


def run_cache(stream, mode: str, threshold: float = 0.8):
    """mode: 'none', 'exact' (normalized text), 'similar' (shingle Jaccard >= threshold), 'guarded' (similar and same machine and numbers),
    'parsed' (key = what the rule parser of chapter 4 understood, falling back to the normalized text).
    Returns (hit rate, wrong-answer rate among hits)."""
    store: list[tuple[str, set, tuple, tuple]] = []; hits = wrong = 0
    canon: dict = {}
    for text, key in stream:
        found = None
        if mode == "parsed":
            from aiind.oeeqa import parse
            ck = parse(text) or normalize(text)      # a structured key when the rules understand the question
            found = canon.get(ck)
            canon.setdefault(ck, key)
        elif mode != "none":
            nt, sh = normalize(text), shingles(text)
            for st, ssh, ent, k0 in store:
                if mode == "exact" and nt == st:
                    found = k0; break
                if mode in ("similar", "guarded") and jaccard(sh, ssh) >= threshold and (mode == "similar" or entities(text) == ent):
                    found = k0; break
        if found is not None:
            hits += 1; wrong += found != key
        else:
            store.append((normalize(text), shingles(text), entities(text), key))
    n = len(stream)
    return hits / n, (wrong / hits if hits else 0.0)


# ------------------------------------------------------------------ cost
def cost_per_million(gpu_per_hour: float, tokens_per_s: float, utilization: float) -> float:
    return gpu_per_hour / (tokens_per_s * utilization * 3600) * 1e6

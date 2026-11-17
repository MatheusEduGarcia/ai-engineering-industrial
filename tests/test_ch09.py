import random

import numpy as np

from aiind.corpus import notes
from aiind.inference import (TinyDecoder, decode_tps, expected_tokens_per_pass, flops_no_cache, flops_decode, kv_cache_bytes, make_requests,
                             query_stream, run_cache, simulate_serving, speculative, speculative_speedup)
from aiind.tinylm import NGram, word_tokens


def test_cache_gives_identical_tokens():
    m = TinyDecoder(seed=1)
    prompt = [3, 14, 15, 9, 2]
    assert m.generate(prompt, 12, cache=True) == m.generate(prompt, 12, cache=False)


def test_cache_logits_match_full():
    m = TinyDecoder(seed=2)
    toks = [5, 6, 7, 8]
    lg, kv = m.prefill(toks)
    lg2, _ = m.step(11, kv, len(toks))
    assert np.allclose(lg2, m.logits_full(toks + [11]), atol=1e-9)


def test_flops_and_kv_arithmetic():
    assert flops_no_cache(1e9, 100, 50) > 10 * flops_decode(1e9, 50)
    assert kv_cache_bytes(32, 32, 128, 1000) == 2 * 32 * 32 * 128 * 1000 * 2
    assert abs(decode_tps(7e9, 16, 2000) - 2000e9 / 14e9) < 1e-6


def test_continuous_beats_static_and_one():
    reqs = make_requests(8, 40, seed=1)
    p95 = {}
    for pol in ("one", "static", "continuous"):
        lat, tps = simulate_serving(reqs, pol)
        p95[pol] = (np.percentile(lat, 95), tps)
    assert p95["continuous"][0] < p95["static"][0] < p95["one"][0]
    assert p95["continuous"][1] >= p95["one"][1]


def test_speculative_matches_target_distribution():
    seqs = [word_tokens(n) for n in notes(600, 1)]
    target, draft = NGram(3).fit(seqs), NGram(1).fit(seqs)
    rng = random.Random(0)
    from collections import Counter
    a = Counter(tuple(speculative(target, draft, rng, 2, 3)[0]) for _ in range(1500))
    b = Counter(tuple(target.generate(rng, max_len=2)) for _ in range(1500))
    keys = set(a) | set(b)
    tv = 0.5 * sum(abs(a[k] - b[k]) / 1500 for k in keys)
    assert tv < 0.12
    assert expected_tokens_per_pass(0.8, 4) > 3 and speculative_speedup(0.8, 4, 0.05) > 2


def test_parsed_cache_has_no_wrong_hits():
    s = query_stream(500, 1)
    h, w = run_cache(s, "similar", 0.7); hp, wp = run_cache(s, "parsed")
    assert w > 0.3 and wp == 0 and hp > run_cache(s, "exact")[0]

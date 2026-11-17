import random

import numpy as np

from aiind.foundation import attention, transformer_params, weights_gb
from aiind.sampling import apply_temperature, generate, top_k, top_p
from aiind.structured import FIELDS, SEP, labeled, parse, rule_extract, schema_allowed, training_sequences, valid
from aiind.tinylm import NGram

P = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}


def test_sampling_filters():
    assert apply_temperature(P, 0) == {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    assert set(top_k(P, 2)) == {"a", "b"}
    assert set(top_p(P, 0.8)) == {"a", "b"} and abs(sum(top_p(P, 0.8).values()) - 1) < 1e-12
    hot = apply_temperature(P, 3)
    assert hot["d"] > P["d"] and hot["a"] < P["a"]


def test_causal_attention_never_looks_ahead():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    _, w = attention(q, k, v)
    assert np.allclose(np.triu(w, 1), 0) and np.allclose(w.sum(1), 1)


def test_size_math():
    assert weights_gb(7e9, 4) == 3.5
    assert 6e9 < transformer_params(32, 4096, 128_000) < 8e9


def test_constrained_decoding_is_always_valid():
    data = labeled(1500)
    m = NGram(3).fit(training_sequences(data))
    rng = random.Random(0)
    for note, _ in labeled(60, seed=7):
        prompt = note.split() + [SEP]
        toks, _ = generate(m, rng, prompt=prompt, max_len=len(prompt) + 16, allowed=schema_allowed(len(prompt)))
        assert valid(parse(toks[len(prompt):])) == []


def test_rule_baseline_and_its_known_bug():
    data = labeled(200, seed=7)
    assert sum(rule_extract(n) == r for n, r in data) / len(data) > 0.98
    assert rule_extract("tampadora parada falta de garrafa")["peca"] == "garra"      # substring bug, kept on purpose

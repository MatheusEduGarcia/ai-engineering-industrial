import copy

import numpy as np

from aiind.finetune import Net, bow, fit, lora_params, quantize, task_data, train_memory_gb, vocab_of
from aiind.evaluation import fault_notes


def small():
    texts = [t for t, _, _ in fault_notes(200, 1)]
    voc = vocab_of(texts)
    X = bow(texts[:30], voc)
    y = np.arange(30) % 3
    net = Net(len(voc), d=8, h=6, seed=1)
    net.add_head("T", 3)
    return net, X, y


def test_gradients_match_numerical():
    net, X, y = small()
    net.add_lora(2)
    net.lora["Vt"] += 0.1                       # so the gradient of U is not trivially zero
    g, _ = net.grads(X, y, "T")
    for name in ("W", "E", "Wo", "U", "Vt"):
        t = net.tensors("T")[name]
        idx = np.unravel_index(np.argmax(np.abs(g[name])), t.shape)
        old = t[idx]
        t[idx] = old + 1e-5; _, lp = net.grads(X, y, "T")
        t[idx] = old - 1e-5; _, lm = net.grads(X, y, "T")
        t[idx] = old
        assert abs((lp - lm) / 2e-5 - g[name][idx]) < 1e-6 * max(1, abs(g[name][idx])) + 1e-7, name


def test_lora_starts_as_the_base_and_trains_only_its_parts():
    net, X, y = small()
    before = net.forward(X, "T")[2].copy()
    W0 = net.p["W"].copy()
    net.add_lora(2)
    assert np.allclose(net.forward(X, "T")[2], before)
    hist = fit(net, X, y, "T", {"U", "Vt", "Wo", "bo"}, epochs=100, lr=0.05)
    assert hist[-1] < hist[0] and np.array_equal(net.p["W"], W0)


def test_merge_is_exact():
    net, X, y = small()
    net.add_lora(2)
    fit(net, X, y, "T", {"U", "Vt", "Wo", "bo"}, epochs=50, lr=0.05)
    with_adapter = net.forward(X, "T")[2]
    merged = copy.deepcopy(net)
    merged.p["W"] = merged.W_eff()
    merged.lora = None
    assert np.allclose(merged.forward(X, "T")[2], with_adapter)


def test_quantization():
    r = np.random.default_rng(0)
    W = r.normal(size=(32, 32))
    e = lambda bits, g=None: np.abs(quantize(W, bits, g) - W).mean()
    assert e(8) < e(4) < e(2)
    W[0, 0] = 50                                  # one outlier ruins a per-tensor scale
    assert np.abs(quantize(W, 4, 16) - W).mean() < np.abs(quantize(W, 4) - W).mean()


def test_memory_arithmetic():
    p = 7e9
    full, lora, qlora = train_memory_gb(p, "full"), train_memory_gb(p, "lora", lora_params(32, 4096, 16)), train_memory_gb(p, "qlora", lora_params(32, 4096, 16))
    assert full > lora > qlora and 100 < full < 120

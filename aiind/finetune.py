"""
Finetuning from scratch, in numpy: a tiny network, full finetuning, LoRA, quantization and the memory arithmetic. Chapter 7.

The network reads a stop note as a bag of words:   x = mean of word vectors  ->  h = tanh(x W + b)  ->  class scores.
Word vectors `E` come from "pretraining" (co-occurrence, notebook 03). `W` is pretrained on a first task
(which machine?). The new task (which fault family?) is learned from few examples that show only some of the
synonyms of each family, and tested on all of them. Methods compared: train from scratch, head only, full
finetuning, LoRA on W. Gradients are written by hand so every step is visible.
"""
from __future__ import annotations

import math

import numpy as np

from .corpus import MACHINES
from .evaluation import FAMILIES, Embedder, fault_notes, words

FAMS = list(FAMILIES)


# ------------------------------------------------------------------ data
def vocab_of(texts: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(sorted({w for t in texts for w in words(t)}))}


def bow(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    X = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        ws = [w for w in words(t) if w in vocab]
        for w in ws:
            X[i, vocab[w]] += 1 / max(len(ws), 1)
    return X


def task_data(n: int, seed: int, symptoms_seen: int | None = None):
    """Notes with two labels: machine (task A) and fault family (task B).
    `symptoms_seen` keeps only the first k synonyms of each family, to test generalization to unseen words."""
    rows = []
    for text, fam, sym in fault_notes(n * 3, seed):
        if symptoms_seen is not None and FAMILIES[fam]["sintomas"].index(sym) >= symptoms_seen:
            continue
        machine = next(m for m in MACHINES if m in text.split())
        rows.append((text, MACHINES.index(machine), FAMS.index(fam)))
        if len(rows) == n:
            break
    return [r[0] for r in rows], np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


# ------------------------------------------------------------------ the network
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class Net:
    def __init__(self, V: int, d: int = 32, h: int = 64, seed: int = 0):
        r = np.random.default_rng(seed)
        self.p = {"E": r.normal(0, 0.3, (V, d)), "W": r.normal(0, 1 / math.sqrt(d), (d, h)), "b": np.zeros(h)}
        self.heads: dict[str, dict] = {}
        self.lora: dict | None = None
        self.d, self.h, self.rng = d, h, r

    def add_head(self, name: str, classes: int):
        self.heads[name] = {"Wo": self.rng.normal(0, 1 / math.sqrt(self.h), (self.h, classes)), "bo": np.zeros(classes)}

    def add_lora(self, rank: int, alpha: float = 8.0):
        """W_effective = W + (alpha / rank) * U @ Vt. Vt starts at zero, so the model starts unchanged."""
        self.lora = {"U": self.rng.normal(0, 1 / math.sqrt(self.d), (self.d, rank)), "Vt": np.zeros((rank, self.h)), "scale": alpha / rank}

    def W_eff(self):
        return self.p["W"] + (self.lora["scale"] * self.lora["U"] @ self.lora["Vt"] if self.lora else 0)

    def forward(self, X, head):
        x = X @ self.p["E"]
        a = x @ self.W_eff() + self.p["b"]
        hid = np.tanh(a)
        return x, hid, hid @ self.heads[head]["Wo"] + self.heads[head]["bo"]

    def predict(self, X, head):
        return self.forward(X, head)[2].argmax(1)

    def accuracy(self, X, y, head):
        return float((self.predict(X, head) == y).mean())

    # parameters that can be trained, by name
    def tensors(self, head: str) -> dict[str, np.ndarray]:
        t = {f"E": self.p["E"], "W": self.p["W"], "b": self.p["b"], "Wo": self.heads[head]["Wo"], "bo": self.heads[head]["bo"]}
        if self.lora:
            t.update({"U": self.lora["U"], "Vt": self.lora["Vt"]})
        return t

    def grads(self, X, y, head):
        n = len(y)
        x, hid, z = self.forward(X, head)
        p = softmax(z); p[np.arange(n), y] -= 1; p /= n
        g = {"Wo": hid.T @ p, "bo": p.sum(0)}
        da = (p @ self.heads[head]["Wo"].T) * (1 - hid ** 2)
        dWeff = x.T @ da
        g["b"] = da.sum(0)
        g["W"] = dWeff
        if self.lora:
            s = self.lora["scale"]
            g["U"] = s * dWeff @ self.lora["Vt"].T
            g["Vt"] = s * self.lora["U"].T @ dWeff
        g["E"] = X.T @ (da @ self.W_eff().T)
        loss = -np.log(softmax(z)[np.arange(n), y] + 1e-12).mean()
        return g, float(loss)


def fit(net: Net, X, y, head: str, train: set[str], epochs: int = 300, lr: float = 0.02, wd: float = 0.0):
    """Adam on the chosen parameters only; everything else is frozen. Returns the loss history."""
    ts = net.tensors(head)
    m = {k: np.zeros_like(ts[k]) for k in train}
    v = {k: np.zeros_like(ts[k]) for k in train}
    hist = []
    for t in range(1, epochs + 1):
        g, loss = net.grads(X, y, head)
        hist.append(loss)
        for k in train:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            ts[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8) + lr * wd * ts[k]
    return hist


def n_params(net: Net, names) -> int:
    ts = net.tensors(next(iter(net.heads)))
    return sum(ts[k].size for k in names)


# ------------------------------------------------------------------ quantization
def quantize(W: np.ndarray, bits: int, group: int | None = None) -> np.ndarray:
    """Symmetric absmax quantization and back (simulated): returns the weights as the low-precision model sees them.
    group=None: one scale for the whole tensor. group=g: one scale per g consecutive values."""
    q = 2 ** (bits - 1) - 1
    flat = W.reshape(-1)
    if group is None:
        scale = np.abs(flat).max() / q or 1.0
        return (np.clip(np.round(flat / scale), -q, q) * scale).reshape(W.shape)
    pad = (-len(flat)) % group
    f = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group)
    scale = np.abs(f).max(1, keepdims=True) / q
    scale[scale == 0] = 1.0
    out = np.clip(np.round(f / scale), -q, q) * scale
    return out.reshape(-1)[:len(flat)].reshape(W.shape)


# ------------------------------------------------------------------ memory arithmetic
def lora_params(layers: int, d: int, rank: int, matrices: int = 4) -> int:
    """LoRA on the q, k, v, o projections (d x d each): rank * (d + d) per matrix."""
    return layers * matrices * rank * 2 * d


def train_memory_gb(params: float, method: str, trainable: float = 0.0) -> float:
    """Rule-of-thumb training memory for weights + gradients + optimizer states, in GB. Activations come on top.
    full: fp16 weights (2) + fp16 grads (2) + Adam in fp32 (master copy 4 + two states 8) = 16 bytes per parameter.
    lora: frozen fp16 weights (2) + the 16 bytes per *trainable* parameter.
    qlora: frozen weights in 4 bits (0.5) + 16 bytes per trainable parameter."""
    if method == "full":
        return params * 16 / 1e9
    if method == "lora":
        return (params * 2 + trainable * 16) / 1e9
    if method == "qlora":
        return (params * 0.5 + trainable * 16) / 1e9
    raise ValueError(method)

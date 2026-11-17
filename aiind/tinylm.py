"""
The smallest language model that still teaches the big ideas.

An n-gram model counts which token follows which context, and turns the counts
into probabilities. It has no neural network, but it shares with a large language
model the three ideas that matter for chapter 1:

  * it predicts the next token from the previous ones;
  * it learns from raw text with no labels (self-supervision);
  * generating text is just sampling the next token, again and again.

Also here: three tokenizers (characters, words, and a tiny byte-pair encoding),
because "token" is the unit everything in AI engineering is priced and limited in.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


# ----------------------------------------------------------------- tokenizers
def char_tokens(text: str) -> list[str]:
    return list(text)


def word_tokens(text: str) -> list[str]:
    return text.split()


class BPE:
    """Byte-pair encoding in 30 lines: start from characters, merge the most frequent pair, repeat."""

    def __init__(self, merges: int = 100):
        self.n_merges = merges
        self.merges: list[tuple[str, str]] = []

    def fit(self, texts: list[str]) -> "BPE":
        words = Counter(tuple(w) + ("_",) for t in texts for w in t.split())
        for _ in range(self.n_merges):
            pairs = Counter()
            for w, c in words.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            words = Counter({self._merge(w, best): c for w, c in words.items()})
        return self

    @staticmethod
    def _merge(w: tuple, pair: tuple) -> tuple:
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                out.append(w[i] + w[i + 1]); i += 2
            else:
                out.append(w[i]); i += 1
        return tuple(out)

    def tokens(self, text: str) -> list[str]:
        out = []
        for word in text.split():
            w = tuple(word) + ("_",)
            for pair in self.merges:
                w = self._merge(w, pair)
            out.extend(w)
        return out

    @property
    def vocab(self) -> set[str]:
        return {a + b for a, b in self.merges}


# ----------------------------------------------------------------- model
class NGram:
    """Next-token model with add-k smoothing. order=2 is a bigram model."""

    def __init__(self, order: int = 2, k: float = 0.01):
        self.order, self.k = order, k
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, sequences: list[list[str]]) -> "NGram":
        for seq in sequences:
            seq = [BOS] * (self.order - 1) + seq + [EOS]
            self.vocab.update(seq)
            for i in range(self.order - 1, len(seq)):
                self.counts[tuple(seq[i - self.order + 1:i])][seq[i]] += 1
        return self

    def training_pairs(self, seq: list[str]) -> list[tuple[tuple, str]]:
        """The (context, next token) examples hidden in one sequence: the labels come from the text itself."""
        seq = [BOS] * (self.order - 1) + seq + [EOS]
        return [(tuple(seq[i - self.order + 1:i]), seq[i]) for i in range(self.order - 1, len(seq))]

    def next_probs(self, context: list[str]) -> dict[str, float]:
        ctx = tuple(([BOS] * (self.order - 1) + list(context))[-(self.order - 1):]) if self.order > 1 else ()
        c = self.counts.get(ctx, Counter())
        total = sum(c.values()) + self.k * len(self.vocab - {BOS})
        return {t: (c[t] + self.k) / total for t in self.vocab if t != BOS}

    def generate(self, rng: random.Random, temperature: float = 1.0, max_len: int = 20, prompt: list[str] | None = None) -> list[str]:
        out = list(prompt or [])
        while len(out) < max_len:
            probs = self.next_probs(out)
            toks = list(probs)
            if temperature == 0:
                nxt = max(toks, key=probs.get)
            else:
                w = [math.exp(math.log(probs[t]) / temperature) for t in toks]
                nxt = rng.choices(toks, w)[0]
            if nxt == EOS:
                break
            out.append(nxt)
        return out

    def perplexity(self, sequences: list[list[str]]) -> float:
        """How surprised the model is by new text: exp of the average negative log-probability per token."""
        nll, n = 0.0, 0
        for seq in sequences:
            for ctx, tok in self.training_pairs(seq):
                p = self.next_probs(list(ctx)[-(self.order - 1):] if self.order > 1 else []).get(tok, 1e-12)
                nll -= math.log(p); n += 1
        return math.exp(nll / n)

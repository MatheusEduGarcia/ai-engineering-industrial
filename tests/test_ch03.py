import math
import random

from aiind.corpus import notes
from aiind.evaluation import (Embedder, SimJudge, auc, cohen_kappa, cross_entropy, debiased_prefer, elo, entropy,
                              fault_notes, kl, summaries, strict_note, token_f1)


def test_entropy_and_kl():
    uni = {c: 0.25 for c in "abcd"}
    assert abs(entropy(uni) - 2) < 1e-12 and 2 ** entropy(uni) == 4
    skew = {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1}
    assert cross_entropy(uni, skew) >= entropy(uni) and kl(uni, uni) == 0 and kl(uni, skew) > 0


def test_similarity_and_auc():
    assert token_f1("a b", "a b") == 1 and token_f1("a", "b") == 0
    assert auc([2, 3], [0, 1]) == 1 and auc([1], [1]) == 0.5


def test_embedder_groups_synonyms():
    e = Embedder(12).fit([t for t, _, _ in fault_notes(2000)])
    assert e.similarity("vazando", "pingando") > e.similarity("vazando", "travou") + 0.3


def test_strict_note():
    assert all(strict_note(n) for n in notes(300))
    assert not strict_note("enchedora parada garra vazando")


def test_judge_protocol():
    s = summaries(50)
    a, b = next(x for x in s if x.kind == "curta certa"), next(x for x in s if x.kind == "curta errada")
    fair = SimJudge(noise=0, length_bias=0, position_bias=0)
    assert fair.prefer(a, b) == "A" and fair.prefer(b, a) == "B" and debiased_prefer(fair, a, b) == "A"
    biased = SimJudge(noise=0, length_bias=0, position_bias=5)
    assert debiased_prefer(biased, a, b) == "tie"
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1


def test_elo_conserves_points_and_ranks():
    r = elo([("a", "b", 1.0)] * 5 + [("b", "a", 1.0)])
    assert abs(sum(r.values()) - 2000) < 1e-9 and r["a"] > r["b"]

import numpy as np

from aiind.datasets import (NB, Memorizer, dedup_exact, dedup_near, dedup_normalized, est_jaccard, flag_suspects, jaccard, leakage,
                            minhash, normalize, raw_reports, shingles, split_by_group, split_random, synth, held_out)


def test_normalize_and_exact():
    assert normalize("  Prensa  PARADA, vazamento!! ") == "prensa parada vazamento"
    R = raw_reports(100, 1)
    assert len(dedup_exact(R)) >= len(dedup_normalized(R)) >= len(dedup_near(R, 0.8))


def test_minhash_estimates_jaccard():
    a, b = shingles("prensa parada vedacao vazando lote 1234"), shingles("prensa parada vedacao vazando lote 1299")
    errs = [abs(est_jaccard(minhash(a, 256, s), minhash(b, 256, s)) - jaccard(a, b)) for s in range(5)]
    assert np.mean(errs) < 0.08


def test_near_dedup_finds_duplicates():
    R = raw_reports(200, 2, label_noise=0)
    kept = dedup_near(R, 0.8)
    groups = {r.group for r in R if r.kind != "junk"}
    assert len(groups) * 0.9 <= len([r for r in kept if r.kind != "junk"]) <= len(groups) * 1.1 + 12


def test_group_split_has_less_leakage():
    R = [r for r in raw_reports(200, 3, dup_rate=1.0) if r.kind != "junk"]
    a, b = split_random(R, 0.3, 0)
    c, d = split_by_group(R, 0.3, 0)
    assert leakage(a, b) > leakage(c, d) + 0.2
    assert not ({r.group for r in c} & {r.group for r in d})


def test_label_flagging_beats_chance():
    R = raw_reports(300, 4, dup_rate=0, junk_rate=0, label_noise=0.2)
    fl = set(flag_suspects([r.text for r in R], [r.label for r in R], thr=0.3))
    wrong = {i for i, r in enumerate(R) if r.label != r.true}
    assert len(fl & wrong) / max(1, len(fl)) > 0.6


def test_synthetic_coverage_helps():
    tt, ty = held_out(300)
    R = raw_reports(40, 5, dup_rate=0, junk_rate=0, label_noise=0, n_symptoms=2)
    base = NB().fit([r.text for r in R], [r.label for r in R]).accuracy(tt, ty)
    S = synth(200, 5)
    aug = NB().fit([r.text for r in R] + [t for t, _ in S], [r.label for r in R] + [f for _, f in S]).accuracy(tt, ty)
    assert aug > base + 0.1

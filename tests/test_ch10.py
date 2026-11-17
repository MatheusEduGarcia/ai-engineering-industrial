import numpy as np

from aiind.oeeqa import build_questions, make_data
from aiind.system import System, audit_sample, novel_questions, operator_feedback, p_chart, parse_v2, run, summarize


def setup():
    data = make_data(5)
    return data, build_questions(data, 300, 21)


def test_components_each_help():
    data, qs = setup()
    full = summarize(run(System(data), qs))
    no_rules = summarize(run(System(data, use_rules=False), qs))
    no_guard = summarize(run(System(data, use_guard=False), qs))
    assert full["erro mostrado"] < no_rules["erro mostrado"] and full["custo por mil"] < no_rules["custo por mil"]
    assert full["erro mostrado"] <= no_guard["erro mostrado"]


def test_cache_is_cheaper_and_not_wrong():
    data, qs = setup()
    rep = qs + qs
    a = summarize(run(System(data), rep)); b = summarize(run(System(data, use_cache=False), rep))
    assert a["custo por mil"] < b["custo por mil"] and a["certo"] >= b["certo"] - 0.02


def test_v2_parses_more_and_improves():
    data, qs = setup()
    assert parse_v2("Peças boas da prensa no 3º dia?") is not None
    v1 = summarize(run(System(data, 1), qs)); v2 = summarize(run(System(data, 2), qs))
    assert v2["certo"] > v1["certo"] and v2["erro mostrado"] < v1["erro mostrado"]


def test_feedback_is_biased_and_audit_is_not():
    data, qs = setup()
    ev = run(System(data), qs * 3)
    sig = operator_feedback(ev, 1)
    ups = sum(s.thumb == "up" for s in sig); downs = sum(s.thumb == "down" for s in sig)
    truth = np.mean([e.ok for e in ev])
    assert abs(ups / (ups + downs) - truth) > 0.15
    aud = audit_sample(ev, 400, 0)
    assert abs(np.mean([e.ok for e in aud]) - truth) < 0.05


def test_p_chart_flags_shift():
    p0, lo, hi = p_chart([20, 22, 18, 21, 60], [250] * 5)
    assert 60 / 250 > hi[4] and 20 / 250 < hi[0]

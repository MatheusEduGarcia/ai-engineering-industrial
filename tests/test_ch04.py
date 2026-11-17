from aiind.oeeqa import (DAYS, build_questions, correct, factors, format_ok, ingenuo, make_data, oee_naive,
                         paired_bootstrap, parse, regras, wilson)

DATA = make_data()
QS = build_questions(DATA, 200)


def test_oee_is_product_of_factors():
    f = factors(DATA, "prensa", 3)
    assert abs(f["oee"] - f["disponibilidade"] * f["desempenho"] * f["qualidade"]) < 1e-12 and 0 < f["oee"] < 1


def test_gold_is_deterministic_and_has_traps():
    assert [q.gold for q in QS] == [q.gold for q in build_questions(DATA, 200)]
    assert any(q.gold == "sem dados" for q in QS) and all(q.day > DAYS for q in QS if q.gold == "sem dados")


def test_rules_are_exact_where_they_parse():
    for q in QS:
        if parse(q.text) and q.asked != "perda":
            assert correct(q, regras(DATA, q)), q.text


def test_naive_shortcut_is_wrong_often():
    wrong = sum(not correct(q, ingenuo(DATA, q)) for q in QS if q.kind == "oee")
    assert wrong > 0 and any(abs(oee_naive(DATA, "prensa", d) - factors(DATA, "prensa", d)["oee"]) > 0.005 for d in range(1, 15))


def test_scoring():
    q = next(q for q in QS if q.kind == "oee")
    assert correct(q, f"O valor é {q.gold}.") and not correct(q, "não sei")
    assert format_ok("72,3 %") and format_ok("esteira") and not format_ok("O valor é 72")


def test_stats():
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1
    mean, lo, hi = paired_bootstrap([True] * 90 + [False] * 10, [True] * 50 + [False] * 50)
    assert lo > 0 and abs(mean - 0.4) < 1e-9

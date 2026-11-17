from aiind.oeeqa import make_data
from aiind.rag import (BM25, Agent, Reader, SimPlanner, chunk_paragraph, chunk_section, chunk_window, expand, glossary, hit,
                       make_manuals, make_questions, make_tasks, make_tools, ranking, score_task)

PARAS, FACTS = make_manuals()
QS = make_questions(FACTS, 120)
CH = chunk_paragraph(PARAS, glossary_too=False)
BM = BM25(CH)
DATA = make_data()


def test_every_fact_has_its_evidence_in_the_manual():
    text = " ".join(p[3] for p in PARAS)
    assert all(f.evidence in text and f.answer in f.evidence or f.kind == "alarme" for f in FACTS)


def test_paragraph_chunks_keep_evidence_windows_may_cut_it():
    assert all(any(q.evidence in c for c in CH) for q in QS)
    assert not all(any(q.evidence in c for c in chunk_window(PARAS, 20, 4)) for q in QS)


def test_bm25_finds_numeric_and_code_questions():
    for q in QS:
        if q.kind in ("numerica", "codigo"):
            assert hit(CH, ranking(BM.scores(q.text)), q.evidence, 1), q.text


def test_glossary_expansion_helps_symptom_questions():
    gl = glossary(PARAS)
    sym = [q for q in QS if q.kind == "sintoma"]
    plain = sum(hit(CH, ranking(BM.scores(q.text)), q.evidence, 1) for q in sym)
    expd = sum(hit(CH, ranking(BM.scores(expand(q.text, gl))), q.evidence, 1) for q in sym)
    assert expd > plain


def test_reader_abstains_or_invents_without_evidence():
    q = QS[0]
    assert Reader(True).answer(q, ["nada"], 0) != q.answer and Reader(True).answer(q, [q.evidence], 0) == q.answer


def test_agent_validation_permission_and_approval():
    gl = glossary(PARAS)
    tools = make_tools(DATA, lambda c: CH[ranking(BM.scores(expand(c, gl)))[0]])
    tasks = make_tasks(DATA, FACTS, 30)
    ok = [score_task(t, Agent(tools, SimPlanner(0.0), 6).run(t)) for t in tasks]
    assert sum(ok) / len(ok) > 0.8
    t = tasks[0]
    assert Agent(tools, SimPlanner(0.0), 6, allowed=set()).run(t).status == "bloqueado"
    os_task = next(t for t in tasks if t.kind == "os")
    assert Agent(tools, SimPlanner(0.0), 6).run(os_task).effects == []
    assert len(Agent(tools, SimPlanner(0.0), 6, approve=lambda n, a: True).run(os_task).effects) == 1

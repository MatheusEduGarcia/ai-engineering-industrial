import random
from aiind.corpus import notes
from aiind.tinylm import BPE, NGram, char_tokens, word_tokens

TR, TE = notes(2000), notes(300, seed=7)


def test_more_context_less_perplexity():
    p = [NGram(o).fit([word_tokens(t) for t in TR]).perplexity([word_tokens(t) for t in TE]) for o in (1, 2, 3)]
    assert p[0] > p[1] > p[2]


def test_probabilities_sum_to_one():
    m = NGram(2).fit([word_tokens(t) for t in TR])
    assert abs(sum(m.next_probs(["prensa"]).values()) - 1) < 1e-9


def test_temperature_zero_is_deterministic():
    m = NGram(3).fit([word_tokens(t) for t in TR])
    assert len({" ".join(m.generate(random.Random(i), temperature=0)) for i in range(5)}) == 1


def test_bpe_sits_between_chars_and_words_and_handles_unseen_words():
    b = BPE(80).fit(TR)
    text = " ".join(TE)
    assert len(word_tokens(text)) < len(b.tokens(text)) < len(char_tokens(text))
    assert "".join(b.tokens("robô colaborativo")).replace("_", "") == "robôcolaborativo"

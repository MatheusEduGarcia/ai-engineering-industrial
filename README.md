# AI Engineering, applied to industry · study notebooks

[![CI](https://github.com/MatheusEduGarcia/ai-engineering-industrial/actions/workflows/ci.yml/badge.svg)](https://github.com/MatheusEduGarcia/ai-engineering-industrial/actions/workflows/ci.yml)

A study path through *AI Engineering: Building Applications with Foundation Models* (Chip Huyen, O'Reilly, 2025), one executed notebook per chapter, with every idea turned into code and applied to the shop floor: operator stop notes, maintenance manuals, OEE.

**This repository does not reproduce the book.** The text, figures and examples here are original; the book is the reading path, cited by chapter and section. Read the chapter, then run the notebook. If you do not have the book, the notebooks still stand on their own, but the book goes much deeper. [Get it from O'Reilly](https://www.oreilly.com/library/view/ai-engineering/9781098166298/).

**Status: complete.** Ten chapters, ten executed notebooks, 54 tests, CI rebuilding every notebook on each push.

## At a glance

What the path shows, with numbers produced by the code in this repository:

- **Structure is not correctness.** Constrained decoding gives 100 % valid JSON and 0 % correct records; plain rules give 99.5 % (ch. 2).
- **Code first, model for the rest.** On an OEE question set with computed answers, rules plus a simulated assistant reach 90 % against 80 % for the assistant alone, at half the cost (ch. 4), and halve the wrong answers shown to operators (ch. 10).
- **Defences belong in code.** Prompt injection succeeds 54 to 63 % of the time undefended and 0 % when the model only picks a tool call and code computes the value (ch. 5).
- **Retrieval details decide RAG.** Fixed windows find 2 to 62 % of the evidence, paragraph chunks 100 % (ch. 6).
- **LoRA protects what pretraining knew.** With 40 examples, LoRA reaches 97 % against 88 % for full finetuning, and switching the adapter off restores the old task exactly (ch. 7).
- **Leakage lies.** With repeated reports, a random split reports 98 % for a true 91 % (ch. 8).
- **Decoding is memory-bound.** Quantizing to 4 bits raises the decoding ceiling about 4x; the KV cache gives identical outputs about 20x faster at 300 tokens (ch. 9).
- **Feedback is biased.** Thumbs up say 37 % approval for a true 81 % accuracy; a small random audit says 79 % (ch. 10).

**What this repository is not:** no real language model is called or trained. Every model-like component (assistant, judge, reader, planner, operators) is a simulated stand-in with declared flaws, so each measurement can be checked against the truth. The method transfers; the numbers are hypotheses to re-measure with a real model and real, anonymized plant data.

## The path

| # | Book chapter | Notebook |
|---|---|---|
| 01 | Introduction to Building AI Applications with Foundation Models | [O que é engenharia de IA, construindo um modelo de linguagem do zero](notebooks/01_o_que_e_engenharia_de_ia.ipynb) |
| 02 | Understanding Foundation Models | [Por dentro dos modelos: dados, tamanho, amostragem e saída estruturada](notebooks/02_por_dentro_dos_modelos.ipynb) |
| 03 | Evaluation Methodology | [Como saber se está bom: perplexidade, avaliação exata, embeddings, IA como juiz, Elo](notebooks/03_como_saber_se_esta_bom.ipynb) |
| 04 | Evaluate AI Systems | [Avaliando o sistema inteiro: um conjunto de avaliação sobre OEE](notebooks/04_avaliando_o_sistema_inteiro.ipynb) |
| 05 | Prompt Engineering | [Engenharia de prompts: versionar, testar e atacar](notebooks/05_engenharia_de_prompts.ipynb) |
| 06 | RAG and Agents | [RAG e agentes: manuais, ferramentas e o coletor de OEE](notebooks/06_rag_e_agentes.ipynb) |
| 07 | Finetuning | [Finetuning: quando ajustar, quanto custa, LoRA e quantização](notebooks/07_finetuning.ipynb) |
| 08 | Dataset Engineering | [Engenharia de dados: duplicatas, vazamento, rótulos ruins e dados sintéticos](notebooks/08_engenharia_de_dados.ipynb) |
| 09 | Inference Optimization | [Inferência: latência, vazão, custo e onde está o gargalo](notebooks/09_inferencia.ipynb) |
| 10 | AI Engineering Architecture and User Feedback | [O sistema inteiro: arquitetura, observabilidade e o ciclo de feedback](notebooks/10_o_sistema_inteiro.ipynb) |

## How each notebook teaches

Written in Portuguese, committed executed, in the same format as my other study repositories:

- **Antes de começar:** what you need, what you do not, how long it takes.
- **Pare e preveja** before every experiment: write your answer, then run the cell.
- **Erros comuns**, a quiz with hidden answers, exercises and a glossary.

## Notebook 01 in one picture

A language model built from nothing but counting, trained on 2,000 synthetic operator notes. With no context it hesitates among about 40 words at each step; with one word of context, about 4; with two, fewer still. Then tokens (characters, words, a 30-line BPE), temperature, and a scoring matrix for deciding which factory problems deserve generative AI.

## Notebook 02 in one finding

Turning a free-text stop note into JSON for the OEE system, three ways, with the same small model:

| Strategy | Valid JSON | Correct record |
|---|---|---|
| ask and hope | 29 % | 0 % |
| validate and retry (3x) | 67 % | 0 % |
| constrained decoding | **100 %** | 0 % |
| plain dictionary rules, no AI | 100 % | 99.5 % |

Constrained decoding guarantees the format, never the content: **structure is not correctness**. Also in the notebook: why Portuguese costs about 2.5 times more tokens under an English-heavy tokenizer, attention in 15 lines of numpy, how much memory a model needs at each precision, top-k and top-p measured, and best-of-N with a domain verifier.

## Notebook 03 in one finding

How to measure a model, and when each measure lies. Perplexity on text the model has not seen: it falls with context on training data but rises again from order 4 on new text (overfitting), and it is only comparable across tokenizers as bits per character. A character model spends 0.42 bits per character on new notes, 2.9 on the same words shuffled and 6 on the English translation, which makes perplexity a cheap drift alarm.

| Measure | What it caught |
|---|---|
| exact record match | 0 % for the small model, 100 % for rules; category accuracy is the metric the OEE actually consumes |
| word overlap vs embeddings (AUC, same fault told differently) | 0.69 vs 0.94, but embeddings rate "seal without leak" almost equal to "seal leaking" |
| simulated AI judge | picks the right summary 88 % of the time when it comes first and 56 % when second; asking in both orders trades coverage for precision |
| best-of-N, same verifier for choosing and scoring | 100 % by the verifier, 91 % by an independent check (Goodhart) |
| Elo from pairwise comparisons | needs hundreds of comparisons; human notes and best-of-3 cannot be separated with this judge |

The AI judge in the notebook is simulated, with known biases, so the protocol for measuring a judge can be practised offline. Chapter 6 swaps in a real model.

## Notebook 04 in one finding

An evaluation set for an assistant that answers OEE questions, with gold answers computed by code from a synthetic plant (6 machines, 14 days, 3 shifts). Three answering systems with known flaws: exact rules, a plausible naive shortcut, and a simulated model-like assistant.

| System | Overall | Traps (days with no data) | Cost per 1000 questions |
|---|---|---|---|
| rules | 55 % | 100 % | ~0 |
| naive shortcut | 45 % | 32 % | ~0 |
| simulated assistant | 80 % | 36 % | US$ 0.88 (illustrative) |
| rules first, assistant only when rules refuse | **90 %** | 75 % | US$ 0.42 |

The average hides what matters: the best overall system invents a number for two of every three questions about days that do not exist. Also in the notebook: why the naive average of shift OEEs is wrong by more than a point, how many questions separate two systems (50 to 100 for a 10 to 20 point gap, thousands for a point), duplicate questions inflating a memorizer from 12 % to 22 %, floors from trivial systems, and a quality gate that blocks a version that improves the average but keeps inventing data.

## Notebook 05 in one finding

Prompts as versioned, tested code, for the OEE assistant of notebook 04. Six versions, one change at a time, each measured on the evaluation set and fingerprinted by hash: format fixed by one instruction, traps by an explicit rule plus an example, multi-step questions by step-by-step reasoning. A single run is not enough: seeds move the score by 1 to 2 points, enough to fake an improvement.

Then the attack surface. An operator-notes field is untrusted text inside the prompt, and six injected orders (force a number, fake authority, leak the instructions, plant a link, deny service, change role) work on the undefended prompt in 54 to 63 % of questions.

| Defence | Attack success | What it does not cover |
|---|---|---|
| none | 54 to 63 % | |
| mark the data as data | 9 to 13 % | still not zero |
| + output guardrails (code) | 0 to 9 % | plausible wrong values, "I don't know" |
| model only picks the tool call; code computes; notes never enter the prompt | 0 % by construction | permissions still need checking |

**The model here is a simulator** with readable rules of how a model reacts to a prompt. The attack rates are assumptions; the method, the quality gate with an attack suite, and the architectural point (do not put untrusted text in the prompt, let code do the calculation) are what transfer. Chapter 6 swaps in a real model behind the same interface.

## Notebook 06 in one finding

A RAG built from scratch over synthetic maintenance manuals (every question carries the exact evidence sentence, so retrieval is scored without judging), then a tool-using agent over the OEE data.

| What was measured | Result |
|---|---|
| fixed 20-word windows vs paragraph chunks | 2 to 62 % vs 100 % of evidence found: windows cut the sentence in half |
| BM25 vs embeddings vs hybrid | BM25 wins; the hybrid does not beat it because the dense side is weaker |
| symptom questions ("dripping" when the manual says "leaking") | 44 % at rank 1; expanding the query with the manual's glossary gives 94 %, but only after removing the glossary from the index (indexed, it hijacks the search and the score falls to 0 %) |
| answer quality at k=1 | same 89 % with or without a "say not found" instruction, but invented answers drop from 7 % to 1 % |
| agent with a planner that errs 30 % of the time | 73 % of tasks right without argument validation, 95 % with it, at the cost of one extra step |

The agent's protections are code, not prompt: permission checked in the tool, human approval for any tool that changes the world, a step limit, and a trace. The reader and the planner are **simulated**; retrieval, tools, protections and evaluation are real and model-agnostic. The notebook ends with a sketch for plugging in a real model, marked as untested.

## Notebook 07 in one finding

LoRA, quantization and the training-memory arithmetic, with a tiny numpy network and hand-written gradients (checked against numerical derivatives in the tests). The memory table uses real model sizes; the experiments use a small classifier, so the mechanism transfers and the numbers do not.

| What was measured | Result |
|---|---|
| training memory, 7 B model | about 112 GB full finetuning, 14 GB LoRA, 4 GB QLoRA (rule of thumb of 16 bytes per parameter; activations come on top) |
| 40 examples, new task whose training set shows only half of the synonyms | from scratch 82 %, full finetuning 88 %, head only 97 %, LoRA 97 % |
| why full finetuning loses here | it moves the vectors of the seen words and breaks the structure that tied the unseen synonyms to them |
| forgetting of the old task | full finetuning falls to 67 % down to 36 %; LoRA stays at 97 to 100 % with small learning rates, and switching the adapter off restores exactly 100 % |
| LoRA delta | rank is capped at r, merging it into W gives identical outputs |
| quantization | with one outlier weight, 4 bits per tensor falls to 8 % while 4 bits per group stays at 100 % |

Coverage of the data mattered more than volume: with 320 examples, full finetuning stayed at 93 %. No real LLM was finetuned.

## Notebook 08 in one finding

Raw synthetic stop reports (duplicates, near duplicates, junk, typos, wrong labels, uneven coverage) turned into a trustworthy set, measuring each decision with a small naive Bayes classifier. No real plant data.

| What was measured | Result |
|---|---|
| exact, normalized and near-duplicate removal | 512 records -> 453 -> 428 -> 398 (400 distinct reports); similarity threshold 0.8 balances merges (13 impure clusters) against misses (90 % of repeats caught) |
| leakage with 100 % repeated reports | random split reports 98 % while the external truth is 91 %; a split by group reports 92 % |
| wrong labels | 10 % noise costs about 4 points, 40 % costs over 20; cross-validated flagging points at the wrong labels with 74 to 85 % precision |
| quality vs quantity | 30 clean examples match about 120 noisy ones (30 % wrong labels) |
| synthetic data for coverage gaps | 75 % -> 97 % when it covers the missing synonyms, 92 % when faults are invented, and 80 % on words nobody listed |

The cleaning rule that drops reports without a failure word also dropped 24 good ones with a typo in that very word, and the notebook says so. The classifier is simple on purpose: the subject is the data.

## Notebook 09 in one finding

Why decoding is memory-bound, a real numpy decoder with and without KV cache, simulated batching, speculative decoding checked for exactness, response caches and an edge vs cloud cost model. Hardware and price figures are illustrative; the serving queue is a simulation with assumed step times.

| What was measured | Result |
|---|---|
| KV cache | identical outputs; 3x faster at 50 generated tokens and about 20x at 300 (time grows with the square of the length without it) |
| KV cache memory, 7 B model | 0.5 MB per token; 32 users at 4000 tokens need 67 GB, more than the 14 GB of weights (GQA cuts it 4x) |
| batching policies, simulated | one at a time saturates at about 130 tokens/s; continuous batching reaches about 880 tokens/s with p95 near 4 s up to 6 requests per second |
| speculative decoding | distance to the target distribution is at the noise level; 3 to 4x with a good draft (acceptance 0.86), a loss with a bad or costly one |
| response cache over 800 OEE questions | text similarity: 72 to 97 % hits but 63 to 92 % of them wrong; a structured key (parsed question) gives 62 % hits and 0 % wrong |
| cost | same cost per token at equal utilization, but about US$ 1,440 a month fixed for a rented GPU against about US$ 48 for an edge box |

## Notebook 10 in one finding

The pieces of the earlier chapters assembled into one OEE question-answering system (cache, rules, assistant, guardrails, escalation to a person), with traces, simulated operator feedback of known bias, a regression gate and drift monitoring. The assistant, the costs and the operators are simulated.

| What was measured | Result |
|---|---|
| what each component contributes | rules in front halve the wrong answers shown (7.7 % against 15.3 % without) and the cost; guardrails trade wrong answers for escalations; the cache cuts cost by about 40 % but replays wrong answers too |
| thumbs up as accuracy | 37 % approval from thumbs against 81 % true accuracy (selection bias); a random audited sample of 100 gave 79 % (70 to 86 %) |
| where the feedback points | the question type and phrasing that really fail; the cause was a parser that did not understand "3º dia", a code fix, not a model fix |
| regression gate | v2 beats v1 in every slice with intervals above zero; a candidate with a cache key that forgot the machine scores 100 % when the cache is off and 67 % in the whole system |
| drift | the share of questions outside the rules' vocabulary alarms in week 5, the true error in week 6, a 60-question audit in week 8, and operator feedback never |

The notebook ends with a table of what each chapter left as a tool, and with what this path does not do: no real language model was called or trained, so the method transfers and the numbers are hypotheses.

## Run it

```bash
pip install -r requirements.txt
python -m pytest                      # the small models behind the notebooks
python tools/make_notebooks.py        # rebuild and execute every notebook
```

No GPU, no model download and no API key are needed: everything runs on CPU with numpy. **No real language model is called or trained anywhere in this repository.** Where a chapter needs one (the assistant of chapter 4, the judge of chapter 3, the reader and planner of chapter 6, the operators of chapter 10), a simulated stand-in with known, declared flaws is used, so the measurements can be checked against the truth. The method transfers; the numbers are hypotheses to be measured again with a real model.

## Layout

```
aiind/corpus.py        synthetic shop-floor text (no real plant data)
aiind/tinylm.py        n-gram language model, perplexity, sampling, tokenizers (chars, words, BPE)
aiind/sampling.py      temperature, top-k, top-p, constrained decoding
aiind/structured.py    stop note to JSON: schema, constrained decoding, rule baseline, verifier
aiind/foundation.py    language mix, attention, parameter and memory math
aiind/evaluation.py    entropy, perplexity, exact metrics, embeddings, simulated judge, Elo
aiind/oeeqa.py         synthetic OEE data, questions with computed answers, rule and simulated systems
aiind/prompting.py     versioned prompts, injection attacks, guardrails
aiind/rag.py           chunking, BM25, dense and hybrid search, guarded agent
aiind/finetune.py      small network with hand-written gradients, LoRA, quantization, memory math
aiind/datasets.py      deduplication, leakage, label noise, synthetic data
aiind/inference.py     KV cache, serving simulation, speculative decoding, caches, cost
aiind/system.py        the assembled system, feedback, drift monitoring
tools/make_notebooks.py  the notebooks as code, executed on build
notebooks/             executed study notebooks
```

## Related

[automation-portfolio](https://github.com/MatheusEduGarcia/automation-portfolio) · [oee-monitor-lab](https://github.com/MatheusEduGarcia/oee-monitor-lab) · [industrial-ontology-lab](https://github.com/MatheusEduGarcia/industrial-ontology-lab) · [iiot-protocols-examples](https://github.com/MatheusEduGarcia/iiot-protocols-examples)

MIT license for the code and notebooks in this repository. The book is © 2025 Developer Experience Advisory LLC, published by O'Reilly Media.

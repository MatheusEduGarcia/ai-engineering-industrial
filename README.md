# AI Engineering, applied to industry · study notebooks

[![CI](https://github.com/MatheusEduGarcia/ai-engineering-industrial/actions/workflows/ci.yml/badge.svg)](https://github.com/MatheusEduGarcia/ai-engineering-industrial/actions/workflows/ci.yml)

A study path through *AI Engineering: Building Applications with Foundation Models* (Chip Huyen, O'Reilly, 2025), one executed notebook per chapter, with every idea turned into code and applied to the shop floor: operator stop notes, maintenance manuals, OEE.

**This repository does not reproduce the book.** The text, figures and examples here are original; the book is the reading path, cited by chapter and section. Read the chapter, then run the notebook. If you do not have the book, the notebooks still stand on their own, but the book goes much deeper. [Get it from O'Reilly](https://www.oreilly.com/library/view/ai-engineering/9781098166298/).

## The path

| # | Book chapter | Notebook | Status |
|---|---|---|---|
| 01 | Introduction to Building AI Applications with Foundation Models | [O que é engenharia de IA, construindo um modelo de linguagem do zero](notebooks/01_o_que_e_engenharia_de_ia.ipynb) | done |
| 02 | Understanding Foundation Models | [Por dentro dos modelos: dados, tamanho, amostragem e saída estruturada](notebooks/02_por_dentro_dos_modelos.ipynb) | done |
| 03 | Evaluation Methodology | [Como saber se está bom: perplexidade, avaliação exata, embeddings, IA como juiz, Elo](notebooks/03_como_saber_se_esta_bom.ipynb) | done |
| 04 | Evaluate AI Systems | [Avaliando o sistema inteiro: um conjunto de avaliação sobre OEE](notebooks/04_avaliando_o_sistema_inteiro.ipynb) | done |
| 05 | Prompt Engineering | [Engenharia de prompts: versionar, testar e atacar](notebooks/05_engenharia_de_prompts.ipynb) | done |
| 06 | RAG and Agents | RAG over manuals; an agent whose tools query an OEE collector | next |
| 07 | Finetuning | memory math, quantization and LoRA on a small model, on CPU | planned |
| 08 | Dataset Engineering | deduplicating, cleaning and synthesizing stop reports | planned |
| 09 | Inference Optimization | latency, throughput and caching, measured | planned |
| 10 | AI Engineering Architecture and User Feedback | guardrails, model router, operator feedback | planned |

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

## Run it

```bash
pip install -r requirements.txt
python -m pytest                      # the small models behind the notebooks
python tools/make_notebooks.py        # rebuild and execute every notebook
```

No GPU and no API key are needed for the early chapters. From chapter 2 on, notebooks that call a real model will default to a small local model and accept an API key as an option, with recorded responses so they still run in CI.

## Layout

```
aiind/corpus.py        synthetic shop-floor text (no real plant data)
aiind/tinylm.py        n-gram language model, perplexity, sampling, tokenizers (chars, words, BPE)
aiind/sampling.py      temperature, top-k, top-p, constrained decoding
aiind/structured.py    stop note to JSON: schema, constrained decoding, rule baseline, verifier
aiind/foundation.py    language mix, attention, parameter and memory math
tools/make_notebooks.py  the notebooks as code, executed on build
notebooks/             executed study notebooks
```

## Related

[oee-monitor-lab](https://github.com/MatheusEduGarcia/oee-monitor-lab) · [industrial-ontology-lab](https://github.com/MatheusEduGarcia/industrial-ontology-lab) · [iiot-protocols-examples](https://github.com/MatheusEduGarcia/iiot-protocols-examples)

MIT license for the code and notebooks in this repository. The book is © 2025 Developer Experience Advisory LLC, published by O'Reilly Media.

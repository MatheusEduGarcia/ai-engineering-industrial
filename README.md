# AI Engineering, applied to industry · study notebooks

[![CI](https://github.com/MatheusEduGarcia/ai-engineering-industrial/actions/workflows/ci.yml/badge.svg)](https://github.com/MatheusEduGarcia/ai-engineering-industrial/actions/workflows/ci.yml)

A study path through *AI Engineering: Building Applications with Foundation Models* (Chip Huyen, O'Reilly, 2025), one executed notebook per chapter, with every idea turned into code and applied to the shop floor: operator stop notes, maintenance manuals, OEE.

**This repository does not reproduce the book.** The text, figures and examples here are original; the book is the reading path, cited by chapter and section. Read the chapter, then run the notebook. If you do not have the book, the notebooks still stand on their own, but the book goes much deeper. [Get it from O'Reilly](https://www.oreilly.com/library/view/ai-engineering/9781098166298/).

## The path

| # | Book chapter | Notebook | Status |
|---|---|---|---|
| 01 | Introduction to Building AI Applications with Foundation Models | [O que é engenharia de IA, construindo um modelo de linguagem do zero](notebooks/01_o_que_e_engenharia_de_ia.ipynb) | done |
| 02 | Understanding Foundation Models | sampling, temperature and structured output: free-text stop note to JSON | next |
| 03 | Evaluation Methodology | perplexity, embeddings and AI as a judge on maintenance text | planned |
| 04 | Evaluate AI Systems | an evaluation set of OEE questions with known answers | planned |
| 05 | Prompt Engineering | versioned, tested prompts and prompt injection on a shop-floor assistant | planned |
| 06 | RAG and Agents | RAG over manuals; an agent whose tools query an OEE collector | planned |
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
tools/make_notebooks.py  the notebooks as code, executed on build
notebooks/             executed study notebooks
```

## Related

[oee-monitor-lab](https://github.com/MatheusEduGarcia/oee-monitor-lab) · [industrial-ontology-lab](https://github.com/MatheusEduGarcia/industrial-ontology-lab) · [iiot-protocols-examples](https://github.com/MatheusEduGarcia/iiot-protocols-examples)

MIT license for the code and notebooks in this repository. The book is © 2025 Developer Experience Advisory LLC, published by O'Reilly Media.

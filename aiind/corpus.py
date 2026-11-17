"""
A small, synthetic corpus of shop-floor text, used across the notebooks.

Operators write a short note when a machine stops. Real notes are noisy, short and
repetitive, which makes them a good playground: a language model trained on them
learns the "grammar" of a stop report in seconds.

Everything here is generated from templates with a fixed seed. No real plant data.
"""
from __future__ import annotations

import random

MACHINES = ["enchedora", "tampadora", "rotuladora", "prensa", "esteira", "paletizadora"]
PARTS = {
    "enchedora": ["bico de enchimento", "válvula", "sensor de nível", "vedação"],
    "tampadora": ["cabeçote", "garra", "sensor de tampa", "motor"],
    "rotuladora": ["rolo de cola", "sensor de etiqueta", "bobina", "faca"],
    "prensa": ["matriz", "cilindro hidráulico", "sensor de segurança", "mangueira"],
    "esteira": ["correia", "rolamento", "motor", "sensor de presença"],
    "paletizadora": ["garra", "filme", "sensor de altura", "cortina de luz"],
}
PROBLEMS = ["travou", "vazando", "com ruído", "desalinhado", "sem leitura", "quebrou", "aquecendo"]
ACTIONS = ["ajustado", "trocado", "limpo", "reapertado", "recalibrado", "lubrificado"]
CALLS = ["chamado manutenção", "chamado elétrica", "chamado líder", "aguardando peça"]
PLANNED = ["parada para almoço", "troca de formato", "limpeza programada", "preventiva programada", "troca de turno"]
SHORTAGE = ["falta de tampa", "falta de garrafa", "falta de etiqueta", "falta de operador", "falta de palete"]


def note(rng: random.Random) -> str:
    """One operator note, in one of a few styles people actually use."""
    m = rng.choice(MACHINES)
    p = rng.choice(PARTS[m])
    kind = rng.random()
    if kind < 0.55:
        s = f"{m} parada {p} {rng.choice(PROBLEMS)}"
        if rng.random() < 0.6:
            s += f" {rng.choice(CALLS)}"
        if rng.random() < 0.5:
            s += f" {p} {rng.choice(ACTIONS)}"
        return s
    if kind < 0.8:
        return f"{m} {rng.choice(PLANNED)}"
    if kind < 0.95:
        return f"{m} parada {rng.choice(SHORTAGE)}"
    return f"{m} voltou a operar"


def notes(n: int = 2000, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [note(rng) for _ in range(n)]

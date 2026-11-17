"""
Helpers for chapter 2: training data mix, attention, model size.
"""
from __future__ import annotations

import numpy as np

# word-for-word English for the shop-floor vocabulary, to build a parallel corpus
EN = {
    "enchedora": "filler", "tampadora": "capper", "rotuladora": "labeler", "prensa": "press", "esteira": "conveyor",
    "paletizadora": "palletizer", "parada": "stopped", "bico": "nozzle", "de": "", "enchimento": "filling",
    "válvula": "valve", "sensor": "sensor", "nível": "level", "vedação": "seal", "cabeçote": "head", "garra": "gripper",
    "tampa": "cap", "motor": "motor", "rolo": "roller", "cola": "glue", "etiqueta": "label", "bobina": "reel",
    "faca": "knife", "matriz": "die", "cilindro": "cylinder", "hidráulico": "hydraulic", "segurança": "safety",
    "mangueira": "hose", "correia": "belt", "rolamento": "bearing", "presença": "presence", "filme": "film",
    "altura": "height", "cortina": "curtain", "luz": "light", "travou": "jammed", "vazando": "leaking", "com": "with",
    "ruído": "noise", "desalinhado": "misaligned", "sem": "no", "leitura": "reading", "quebrou": "broke",
    "aquecendo": "overheating", "ajustado": "adjusted", "trocado": "replaced", "limpo": "cleaned",
    "reapertado": "retightened", "recalibrado": "recalibrated", "lubrificado": "lubricated", "chamado": "called",
    "manutenção": "maintenance", "elétrica": "electrical", "líder": "lead", "aguardando": "waiting", "peça": "part",
    "para": "for", "almoço": "lunch", "troca": "change", "formato": "format", "limpeza": "cleaning",
    "programada": "scheduled", "preventiva": "preventive", "turno": "shift", "falta": "shortage", "garrafa": "bottle",
    "operador": "operator", "palete": "pallet", "voltou": "back", "a": "", "operar": "running",
}


def to_english(note: str) -> str:
    return " ".join(w for w in (EN.get(t, t) for t in note.split()) if w)


# ------------------------------------------------------------------ attention
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention. Returns (output, weights)."""
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        scores = scores + np.triu(np.full(scores.shape, -1e9), 1)
    w = softmax(scores)
    return w @ v, w


# ------------------------------------------------------------------ size
def transformer_params(layers: int, d_model: int, vocab: int, ffn_mult: int = 4) -> int:
    """Approximate parameter count of a decoder-only transformer.

    Per layer: attention projections 4·d² plus feed-forward 2·ffn_mult·d²; plus the
    token embedding table vocab·d. Biases and norms are ignored (they are tiny).
    """
    return layers * (4 + 2 * ffn_mult) * d_model ** 2 + vocab * d_model


def weights_gb(params: float, bits: int) -> float:
    return params * bits / 8 / 1e9

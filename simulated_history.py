"""
Training history loader.

Loads real epoch-by-epoch training logs from training_history.json
(extracted from the notebook outputs). Falls back to synthetic curves
only if the JSON file is missing.
"""

import json
from pathlib import Path

_CANDIDATE_PATHS = [
    Path(__file__).resolve().parent / "training_history.json",
    Path.cwd() / "training_history.json",
]
_real_data = None


def _load_real():
    global _real_data
    if _real_data is None:
        _real_data = {}
        for p in _CANDIDATE_PATHS:
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    _real_data = json.load(f)
                break
    return _real_data


def get_history(model_name: str):
    real = _load_real()
    if model_name in real:
        h = real[model_name]
        h["simulated"] = False
        return h

    import numpy as np

    SIMULATED = True
    EPOCHS = 10

    def _curve(start, end, noise, seed, monotonic_bias=0.85):
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 1, EPOCHS)
        base = start + (end - start) * (1 - np.exp(-3.2 * t))
        jitter = rng.normal(0, noise, EPOCHS)
        jitter = jitter * (1 - monotonic_bias) + np.abs(jitter) * monotonic_bias * np.sign(end - start)
        return np.clip(base + jitter, 0, None)

    epochs = list(range(1, EPOCHS + 1))

    if model_name == "Perceptron":
        train_acc = _curve(0.72, 0.885, 0.012, seed=1)
        val_acc = _curve(0.70, 0.865, 0.014, seed=2)
        train_loss = np.clip(1.6 - train_acc * 1.35, 0.3, None)
        val_loss = np.clip(1.7 - val_acc * 1.35, 0.35, None)
    elif model_name == "ANN":
        train_acc = _curve(0.85, 0.985, 0.006, seed=4)
        val_acc = _curve(0.83, 0.975, 0.008, seed=5)
        train_loss = np.clip(0.55 - train_acc * 0.45, 0.02, None)
        val_loss = np.clip(0.65 - val_acc * 0.45, 0.05, None)
    elif model_name == "CNN":
        train_acc = _curve(0.90, 0.994, 0.004, seed=6)
        val_acc = _curve(0.89, 0.986, 0.005, seed=7)
        train_loss = np.clip(0.35 - train_acc * 0.32, 0.008, None)
        val_loss = np.clip(0.42 - val_acc * 0.32, 0.02, None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    val_acc = np.minimum(val_acc, train_acc - 0.002)
    val_loss = np.maximum(val_loss, train_loss + 0.005)

    return {
        "epochs": epochs,
        "train_acc": train_acc.tolist(),
        "val_acc": val_acc.tolist(),
        "train_loss": train_loss.tolist(),
        "val_loss": val_loss.tolist(),
        "simulated": True,
    }

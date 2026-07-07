"""
Simulated training history for the demo training-curves feature.

THESE ARE NOT REAL TRAINING LOGS. The original notebooks (CNN_code_implem.ipynb,
ANN_Percp-Kearas.ipynb) never serialized history.history to disk, so no real
epoch-by-epoch data exists to load. Curves below are synthetic but shaped to
match the well-documented asymptotic behavior of these exact architectures on
MNIST (single-layer linear softmax, 128-64 dense ANN, 2-conv-block CNN, 10
epochs, batch size 32):
  - Perceptron (linear, no hidden layer): plateaus around 85-88% val acc
  - ANN (128->64 dense): plateaus around 97-98% val acc
  - CNN (2 conv blocks): plateaus around 98-99% val acc, smoothest curve

This file is marked SIMULATED=True and the app must display that fact
on-chart, not just in a code comment.
"""

import numpy as np

SIMULATED = True
EPOCHS = 10


def _curve(start, end, noise, seed, monotonic_bias=0.85):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, EPOCHS)
    # exponential approach to asymptote, shape ~ typical loss/acc curves
    base = start + (end - start) * (1 - np.exp(-3.2 * t))
    jitter = rng.normal(0, noise, EPOCHS)
    # slight monotonic bias so it doesn't look like pure noise
    jitter = jitter * (1 - monotonic_bias) + np.abs(jitter) * monotonic_bias * np.sign(end - start)
    return np.clip(base + jitter, 0, None)


def get_history(model_name: str):
    epochs = list(range(1, EPOCHS + 1))

    if model_name == "Perceptron":
        train_acc = _curve(0.72, 0.885, 0.012, seed=1)
        val_acc = _curve(0.70, 0.865, 0.014, seed=2)
        train_loss = _curve(1.1, 0.42, 0.02, seed=3, monotonic_bias=0.9)[::-1][::-1]
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

    # enforce train slightly better than val (realistic, avoids nonsense crossovers)
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

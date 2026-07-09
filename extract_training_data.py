"""Extract real training history from CNN_code_implem.ipynb outputs and save as JSON."""

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "CNN_code_implem.ipynb"
OUTPUT = Path(__file__).parent / "training_history.json"


def parse_epoch_line(line: str):
    """Parse a Keras epoch log line and return metrics dict."""
    m = {}
    for key in ["accuracy", "loss", "val_accuracy", "val_loss"]:
        match = re.search(rf"{key}: ([0-9.]+)", line)
        if match:
            m[key] = float(match.group(1))
    return m if len(m) == 4 else None


def extract_histories_from_notebook():
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    histories = {}
    model_keys = {
        "history_perceptron": "Perceptron",
        "history_ann": "ANN",
        "history_cnn": "CNN",
    }

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        for var_name, model_name in model_keys.items():
            if f"{var_name} = " in source and ".fit(" in source:
                epochs_data = []
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "stream":
                        for line in output.get("text", []):
                            parsed = parse_epoch_line(line)
                            if parsed:
                                epochs_data.append(parsed)
                if epochs_data:
                    histories[model_name] = {
                        "epochs": list(range(1, len(epochs_data) + 1)),
                        "train_acc": [e["accuracy"] for e in epochs_data],
                        "val_acc": [e["val_accuracy"] for e in epochs_data],
                        "train_loss": [e["loss"] for e in epochs_data],
                        "val_loss": [e["val_loss"] for e in epochs_data],
                    }
    return histories


if __name__ == "__main__":
    histories = extract_histories_from_notebook()
    if not histories:
        print("ERROR: Could not extract any training histories from notebook.")
        exit(1)

    with open(OUTPUT, "w") as f:
        json.dump(histories, f, indent=2)

    for name, h in histories.items():
        print(f"{name}: {len(h['epochs'])} epochs, final val_acc={h['val_acc'][-1]:.4f}")
    print(f"\nSaved to {OUTPUT}")

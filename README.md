# Digit Recognition Studio

A multi-model handwritten digit classifier built with Streamlit. Draw a digit or upload an image, then compare predictions across three different machine learning architectures side by side.

## Models

| Model | Architecture | MNIST Accuracy | Params |
|-------|-------------|----------------|--------|
| **CNN** | 2 Conv/Pool blocks + Dense head | 99.04% | 225,034 |
| **ANN** | 256→128→64 Dense + BatchNorm | 98.30% | 244,554 |
| **Perceptron** | Single-layer linear classifier | 88.32% | 7,850 |

## Features

- **Draw or Upload** — use the canvas to draw a digit, or upload a PNG/JPG image
- **Model Selector** — switch between CNN, ANN, and Perceptron models
- **Side-by-Side Comparison** — see all three models predict the same input
- **Confidence Bars** — probability distribution across all 10 digits
- **Training Insights** — animated accuracy/loss curves and 3D comparison surface
- **Clear Canvas** — erase and redraw with one click

## Tech Stack

- **Streamlit** — interactive web UI
- **TensorFlow/Keras** — CNN and ANN models
- **scikit-learn** — Perceptron model
- **Plotly** — interactive charts
- **Pillow** — image preprocessing

## Run Locally

```bash
# Clone
git clone https://github.com/loisekk/mnist-digit-studio.git
cd mnist-digit-studio

# Create venv (Python 3.12 required for TensorFlow)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── simulated_history.py   # Synthetic training curves data
├── cnn_model.keras        # Trained CNN model
├── ann_model.keras        # Trained ANN model
├── perceptron_model.pkl   # Trained Perceptron model
├── requirements.txt       # Python dependencies
└── run.bat                # Windows launcher (uses venv)
```

## How It Works

1. **Input** — Canvas captures your drawing as RGBA, or you upload an image
2. **Preprocessing** — Bounding box detection, centering, resize to 28×28 matching MNIST convention
3. **Prediction** — Each model processes the preprocessed input independently
4. **Display** — Predicted digit, confidence score, and probability distribution shown

## License

MIT

"""
MNIST Digit Recognition Studio
Multi-model handwritten digit classifier with drawing canvas.

Models: CNN (Keras), ANN (Keras), Perceptron (sklearn, linear).
Note on Perceptron: sklearn's Perceptron has no predict_proba. The "confidence"
bars for this model are a softmax over decision_function() margins — a linear
score turned into a probability-shaped display, NOT a calibrated probability.
Labeled as such in the UI so it isn't misread as equivalent to the Keras models.
"""

import io
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import joblib
from simulated_history import get_history

# TensorFlow import is slow; defer and cache
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(
    page_title="Digit Recognition Studio",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Theming — dark, futuristic, matches the reference screenshot's teal/dark
# aesthetic. All custom CSS lives here; no external stylesheet dependency.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void: #06110f;
    --bg-panel: #0d1f1b;
    --bg-panel-2: #12281f;
    --border-glow: #1f4a3d;
    --accent-teal: #2dd4a8;
    --accent-teal-dim: #1a8f6f;
    --accent-purple: #a78bfa;
    --accent-amber: #f5b942;
    --text-primary: #eafff6;
    --text-dim: #7fa596;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(45,212,168,0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(167,139,250,0.05) 0%, transparent 40%),
        linear-gradient(180deg, #06110f 0%, #081813 100%);
    background-attachment: fixed;
}

/* subtle grid overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(45,212,168,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(45,212,168,0.035) 1px, transparent 1px);
    background-size: 42px 42px;
    pointer-events: none;
    z-index: 0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1a15 0%, #081512 100%);
    border-right: 1px solid var(--border-glow);
}

h1, h2, h3 {
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #eafff6 0%, #2dd4a8 60%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: var(--text-dim);
    font-size: 1.02rem;
    max-width: 640px;
}

.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-glow);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 0 0 1px rgba(45,212,168,0.04), 0 12px 32px rgba(0,0,0,0.35);
}

.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(45,212,168,0.09);
    border: 1px solid rgba(45,212,168,0.25);
    color: var(--accent-teal);
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

.pred-digit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 5.2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--accent-teal);
    text-shadow: 0 0 24px rgba(45,212,168,0.45);
}

.pred-label {
    color: var(--text-dim);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-ok { background: var(--accent-teal); box-shadow: 0 0 8px var(--accent-teal); }
.status-warn { background: var(--accent-amber); box-shadow: 0 0 8px var(--accent-amber); }

div[data-testid="stMetricValue"] {
    color: var(--accent-teal);
    font-family: 'JetBrains Mono', monospace;
}

.stButton>button {
    background: linear-gradient(135deg, var(--accent-teal-dim), var(--accent-teal));
    color: #041a14;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(45,212,168,0.35);
}

hr { border-color: var(--border-glow) !important; }

.disclaimer-box {
    background: rgba(245,185,66,0.08);
    border: 1px solid rgba(245,185,66,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #e8c97a;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading — cached so switching models in the UI doesn't reload weights
# from disk every rerun. Each model type has a distinct loader/predict path.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cnn():
    from tensorflow.keras.models import load_model
    return load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))


@st.cache_resource(show_spinner=False)
def load_ann():
    from tensorflow.keras.models import load_model
    return load_model(os.path.join(MODEL_DIR, "ann_model.keras"))


@st.cache_resource(show_spinner=False)
def load_perceptron():
    data = joblib.load(os.path.join(MODEL_DIR, "perceptron_model.pkl"))
    if isinstance(data, dict):
        return data["model"], data["scaler"]
    return data, None


MODEL_REGISTRY = {
    "CNN": {
        "loader": load_cnn,
        "kind": "keras",
        "input_shape": (1, 28, 28, 1),
        "desc": "Convolutional network — 2 conv/pool blocks + dense head. Strongest spatial feature extraction.",
        "params": "675,104",
    },
    "ANN": {
        "loader": load_ann,
        "kind": "keras",
        "input_shape": (1, 784),
        "desc": "Fully-connected network — 128→64→10 dense layers on flattened pixels.",
        "params": "328,160",
    },
    "Perceptron": {
        "loader": load_perceptron,
        "kind": "sklearn_linear",
        "input_shape": (1, 784),
        "desc": "Single-layer linear classifier. No hidden layers — a hard baseline.",
        "params": "7,850",
    },
}


# ---------------------------------------------------------------------------
# Live training — trains all three models from scratch on MNIST, updating
# charts epoch-by-epoch. Only runs when the user clicks "Train Models".
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_mnist_data():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train_img = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test_img = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_train_flat = X_train_img.reshape(-1, 784)
    X_test_flat = X_test_img.reshape(-1, 784)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return X_train_img, X_test_img, X_train_flat, X_test_flat, y_train_cat, y_test_cat


def build_perceptron():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_ann():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class LiveChartCallback:
    """Keras training callback that updates a Streamlit chart after each epoch."""

    def __init__(self, chart_placeholder, metric_key, model_name, color, all_histories):
        self.chart_placeholder = chart_placeholder
        self.metric_key = metric_key
        self.model_name = model_name
        self.color = color
        self.all_histories = all_histories
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logs.append({
            "train_acc": logs.get("accuracy", 0),
            "val_acc": logs.get("val_accuracy", 0),
            "train_loss": logs.get("loss", 0),
            "val_loss": logs.get("val_loss", 0),
        })
        self.all_histories[self.model_name] = {
            "epochs": list(range(1, len(self.logs) + 1)),
            "train_acc": [l["train_acc"] for l in self.logs],
            "val_acc": [l["val_acc"] for l in self.logs],
            "train_loss": [l["train_loss"] for l in self.logs],
            "val_loss": [l["val_loss"] for l in self.logs],
        }
        self._update_chart()

    def _update_chart(self):
        key = self.metric_key
        fig = go.Figure()
        for mname, color in MODEL_COLORS.items():
            h = self.all_histories.get(mname)
            if not h or not h.get("epochs"):
                continue
            fig.add_trace(go.Scatter(
                x=h["epochs"], y=h[key],
                mode="lines+markers", name=mname,
                line=dict(color=color, width=3),
                marker=dict(size=7),
            ))
        fig.update_layout(
            height=420,
            plot_bgcolor="#0d1f1b", paper_bgcolor="#0d1f1b",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#eafff6"),
            xaxis=dict(title="Epoch", dtick=1, color="#7fa596",
                       gridcolor="rgba(45,212,168,0.15)"),
            yaxis=dict(title="Validation accuracy" if "acc" in key else "Validation loss",
                       color="#7fa596", gridcolor="rgba(45,212,168,0.15)"),
            legend=dict(bgcolor="rgba(13,31,27,0.8)", font=dict(color="#eafff6")),
            margin=dict(l=50, r=20, t=20, b=50),
        )
        self.chart_placeholder.plotly_chart(fig, use_container_width=True,
                                           config={"displayModeBar": False})


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(model_name, img_array_28x28):
    """
    img_array_28x28: float32, shape (28,28), values in [0,1], MNIST-style
    (digit as light/white strokes on dark background).
    Returns (predicted_digit:int, probs:np.ndarray shape (10,), is_calibrated:bool)
    """
    entry = MODEL_REGISTRY[model_name]
    model = entry["loader"]()

    if entry["kind"] == "keras" and model_name == "CNN":
        x = img_array_28x28.reshape(1, 28, 28, 1).astype("float32")
        probs = model.predict(x, verbose=0)[0]
        return int(np.argmax(probs)), probs, True

    if entry["kind"] == "keras" and model_name == "ANN":
        x = img_array_28x28.reshape(1, 784).astype("float32")
        probs = model.predict(x, verbose=0)[0]
        return int(np.argmax(probs)), probs, True

    if entry["kind"] == "sklearn_linear":
        x = img_array_28x28.reshape(1, 784).astype("float32")
        model, scaler = entry["loader"]()
        if scaler is not None:
            x = scaler.transform(x)
        margins = model.decision_function(x)[0]  # shape (10,)
        probs = softmax(margins)  # NOT a calibrated probability — display-only
        return int(np.argmax(margins)), probs, False

    raise ValueError(f"Unknown model kind for {model_name}")


def preprocess_canvas_image(rgba_array):
    """
    Convert a canvas RGBA array (white strokes on black bg, as configured)
    into a normalized 28x28 float array matching MNIST convention.
    MNIST digits are centered ~20x20 inside the 28x28 frame.
    """
    img = Image.fromarray(rgba_array.astype("uint8"), mode="RGBA").convert("L")
    arr = np.array(img)

    # Find bounding box of the drawn digit (non-zero pixels)
    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        return np.zeros((28, 28), dtype="float32")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop to digit bounding box
    cropped = arr[y0:y1, x0:x1]

    # Make square by padding the shorter side
    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    square = np.zeros((size, size), dtype=np.uint8)
    square[pad_h:pad_h + h, pad_w:pad_w + w] = cropped

    # Resize to 20x20 (MNIST digit occupies ~20x20 inside 28x28)
    digit_img = Image.fromarray(square)
    digit_img = digit_img.resize((20, 20), Image.Resampling.LANCZOS)
    digit_arr = np.array(digit_img).astype("float32") / 255.0

    # Center in 28x28 with 4px padding on each side
    result = np.zeros((28, 28), dtype="float32")
    result[4:24, 4:24] = digit_arr
    return result


def preprocess_uploaded_image(pil_img):
    """
    Handle arbitrary uploaded images: convert to grayscale, resize to 28x28,
    and auto-detect polarity (MNIST = light digit on dark background).
    """
    img = pil_img.convert("L")
    arr = np.array(img)

    # Auto-detect polarity: if bright background with dark strokes, invert
    if arr.mean() > 127:
        arr = 255 - arr

    # Find bounding box of the digit
    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        return np.zeros((28, 28), dtype="float32")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop to digit
    cropped = arr[y0:y1, x0:x1]

    # Make square
    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    square = np.zeros((size, size), dtype=np.uint8)
    square[pad_h:pad_h + h, pad_w:pad_w + w] = cropped

    # Resize to 20x20, center in 28x28
    digit_img = Image.fromarray(square)
    digit_img = digit_img.resize((20, 20), Image.Resampling.LANCZOS)
    digit_arr = np.array(digit_img).astype("float32") / 255.0

    result = np.zeros((28, 28), dtype="float32")
    result[4:24, 4:24] = digit_arr
    return result


# ---------------------------------------------------------------------------
# Sidebar — model selector + info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✳ Model Selector")
    model_name = st.radio(
        "Choose inference model",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
        label_visibility="collapsed",
    )
    entry = MODEL_REGISTRY[model_name]
    st.markdown(f"""
<div class="panel" style="margin-top:0.8rem;">
<span class="metric-chip"><span class="status-dot status-ok"></span>{model_name}</span>
<p style="color:var(--text-dim);font-size:0.85rem;margin-top:0.6rem;">{entry['desc']}</p>
<p style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--accent-teal);">
Params: {entry['params']}
</p>
</div>
""", unsafe_allow_html=True)

    if model_name == "Perceptron":
        st.markdown("""
<div class="disclaimer-box" style="margin-top:0.8rem;">
⚠ This model outputs a linear decision margin, not a calibrated probability.
The confidence bars shown are a softmax approximation for display purposes only.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙ Input Source")
    input_mode = st.radio(
        "How to provide the digit",
        options=["Draw", "Upload image"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🎨 Canvas Settings" if input_mode == "Draw" else "### 📁 Upload Settings")
    if input_mode == "Draw":
        stroke_width = st.slider("Stroke width", 8, 30, 18)
    else:
        stroke_width = 18

    st.markdown("---")
    st.caption("Digit Recognition Studio · built on CNN / ANN / Perceptron trained on MNIST")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Digit Recognition Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Draw a digit or upload an image. Switch models to compare '
    'a linear baseline, a dense network, and a convolutional network on the same input.</div>',
    unsafe_allow_html=True,
)
st.write("")

col_input, col_result = st.columns([1, 1.15], gap="large")

# ---------------------------------------------------------------------------
# Input column
# ---------------------------------------------------------------------------
processed_array = None
predict_clicked = False

with col_input:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### ✏️ Input")

    if input_mode == "Draw":
        try:
            from streamlit_drawable_canvas import st_canvas

            if "canvas_key" not in st.session_state:
                st.session_state.canvas_key = 0

            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,1)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
                display_toolbar=True,
            )

            if canvas_result.image_data is not None and canvas_result.image_data[:, :, 3].sum() > 0:
                processed_array = preprocess_canvas_image(canvas_result.image_data)

            btn_clear, btn_predict = st.columns(2)
            with btn_clear:
                if st.button("🗑 Clear Canvas", use_container_width=True):
                    st.session_state.canvas_key += 1
                    st.rerun()
            with btn_predict:
                predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")
        except Exception as e:
            st.error(
                "Drawing canvas failed to load in this browser session "
                "(streamlit-drawable-canvas is a known-fragile component on "
                "newer Streamlit versions). Use 'Upload image' from the "
                "sidebar instead."
            )
            st.caption(f"Technical detail: {e}")
            predict_clicked = False
    else:
        uploaded = st.file_uploader(
            "Upload a digit image (PNG/JPG, ideally clear single digit on plain background)",
            type=["png", "jpg", "jpeg"],
        )
        predict_clicked = uploaded is not None
        if uploaded is not None:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            st.image(pil_img, caption="Uploaded image", width=200)
            processed_array = preprocess_uploaded_image(pil_img)

    st.markdown("</div>", unsafe_allow_html=True)

if processed_array is not None:
        with st.expander("Preprocessed 28×28 input (what the model actually sees)"):
            st.image(
                (processed_array * 255).astype("uint8"),
                width=140,
                clamp=True,
            )

# ---------------------------------------------------------------------------
# Result column
# ---------------------------------------------------------------------------
with col_result:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🔮 Prediction")

    if processed_array is None or not predict_clicked:
        st.info("Draw or upload a digit, then click **Predict** to see results.")
    else:
        with st.spinner(f"Running {model_name}..."):
            pred_digit = None
            probs = None
            is_calibrated = False
            try:
                pred_digit, probs, is_calibrated = predict(model_name, processed_array)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        if pred_digit is not None:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f'<div class="pred-digit">{pred_digit}</div>', unsafe_allow_html=True)
                st.markdown('<div class="pred-label">Predicted digit</div>', unsafe_allow_html=True)
                conf_label = "Confidence" if is_calibrated else "Margin score (uncalibrated)"
                st.metric(conf_label, f"{probs[pred_digit]*100:.1f}%")

            with c2:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=list(range(10)),
                            y=probs,
                            marker=dict(
                                color=probs,
                                colorscale=[[0, "#12281f"], [1, "#2dd4a8"]],
                                line=dict(color="#1f4a3d", width=1),
                            ),
                            text=[f"{p*100:.1f}%" for p in probs],
                            textposition="outside",
                            textfont=dict(color="#7fa596", size=10),
                        )
                    ]
                )
                fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="#0d1f1b",
                    paper_bgcolor="#0d1f1b",
                    xaxis=dict(
                        tickmode="linear", tick0=0, dtick=1,
                        color="#7fa596", gridcolor="rgba(45,212,168,0.15)",
                        title="Digit",
                    ),
                    yaxis=dict(
                        color="#7fa596", gridcolor="rgba(45,212,168,0.15)",
                        title=("Probability" if is_calibrated else "Softmax(margin)"),
                        range=[0, max(probs) * 1.25],
                    ),
                    font=dict(family="JetBrains Mono, monospace", size=11, color="#eafff6"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model comparison section — run all three at once if input exists
# ---------------------------------------------------------------------------
if processed_array is not None and predict_clicked:
    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### ⚖ Compare all models on this input")
    cols = st.columns(3)
    for i, mname in enumerate(MODEL_REGISTRY.keys()):
        with cols[i]:
            try:
                d, p, calibrated = predict(mname, processed_array)
                st.markdown(f"**{mname}**")
                st.markdown(f'<div style="font-size:2.2rem;font-family:JetBrains Mono,monospace;color:#2dd4a8;">{d}</div>', unsafe_allow_html=True)
                label = "confidence" if calibrated else "margin (uncalibrated)"
                st.caption(f"{p[d]*100:.1f}% {label}")
            except Exception as e:
                st.markdown(f"**{mname}**")
                st.caption(f"Failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Training Insights — live training or saved curves + 3D comparison surface.
# ---------------------------------------------------------------------------
st.write("")
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("#### 📈 Training Insights")

MODEL_COLORS = {"Perceptron": "#f5b942", "ANN": "#a78bfa", "CNN": "#2dd4a8"}

if "live histories" not in st.session_state:
    st.session_state["live histories"] = None

live_col, saved_col = st.columns([1, 1])
with live_col:
    train_clicked = st.button("▶ Train Models Live (MNIST, ~30s)", key="train_btn")
with saved_col:
    if st.session_state["live histories"] is not None:
        if st.button("↩ Load Saved Results", key="load_saved_btn"):
            st.session_state["live histories"] = None
            st.rerun()

if train_clicked:
    st.session_state["live histories"] = {}
    live_hist = st.session_state["live histories"]

    st.info("Loading MNIST dataset...")
    X_train_img, X_test_img, X_train_flat, X_test_flat, y_train_cat, y_test_cat = load_mnist_data()

    SUBSET = 10000
    EPOCHS_LIVE = 5
    X_train_sub = X_train_img[:SUBSET]
    X_test_sub = X_test_img[:SUBSET]
    y_train_sub = y_train_cat[:SUBSET]
    y_test_sub = y_test_cat[:SUBSET]
    X_train_3d = X_train_sub.reshape(-1, 28, 28)
    X_test_3d = X_test_sub.reshape(-1, 28, 28)

    chart_ph = st.empty()
    status_ph = st.empty()
    progress = st.progress(0, text="Starting training...")

    models_to_train = [
        ("Perceptron", build_perceptron, X_train_3d, X_test_3d),
        ("ANN", build_ann, X_train_3d, X_test_3d),
        ("CNN", build_cnn, X_train_sub, X_test_sub),
    ]

    total_models = len(models_to_train)
    for idx, (mname, builder, X_tr, X_te) in enumerate(models_to_train):
        status_ph.info(f"Training **{mname}** ({EPOCHS_LIVE} epochs on {SUBSET:,} samples)...")
        model = builder()

        cb = LiveChartCallback(
            chart_placeholder=chart_ph,
            metric_key="val_acc",
            model_name=mname,
            color=MODEL_COLORS[mname],
            all_histories=live_hist,
        )

        from tensorflow.keras.callbacks import Callback as KerasCB
        class _LiveCB(KerasCB):
            def on_epoch_end(self_inner, epoch, logs=None):
                cb.on_epoch_end(epoch, logs)
                done = (idx * EPOCHS_LIVE + epoch + 1)
                total = total_models * EPOCHS_LIVE
                progress.progress(done / total, text=f"{mname} — epoch {epoch+1}/{EPOCHS_LIVE}")

        model.fit(
            X_tr, y_train_sub,
            epochs=EPOCHS_LIVE, batch_size=64,
            validation_data=(X_te, y_test_sub),
            callbacks=[_LiveCB()],
            verbose=0,
        )
        status_ph.success(f"✅ {mname} done — val acc: {cb.logs[-1]['val_acc']*100:.1f}%")

    progress.progress(1.0, text="All models trained!")
    status_ph.success("All models trained!")
    st.rerun()

ALL_HISTORIES = st.session_state["live histories"] or {m: get_history(m) for m in MODEL_REGISTRY.keys()}
using_live = st.session_state["live histories"] is not None

if not using_live:
    st.markdown(
        '<div class="disclaimer-box">'
        '⚠ Showing saved results. Click <b>▶ Train Models Live</b> to watch real training.'
        '</div>',
        unsafe_allow_html=True,
    )
st.write("")

hist_tabs = st.tabs(["Animated accuracy/loss", "3D comparison surface", "Final metrics"])

with hist_tabs[0]:
    metric_choice = st.radio(
        "Metric", ["Validation accuracy", "Validation loss"],
        horizontal=True, key="metric_choice",
    )
    key = "val_acc" if metric_choice == "Validation accuracy" else "val_loss"

    fig_anim = go.Figure()
    for mname, color in MODEL_COLORS.items():
        h = ALL_HISTORIES.get(mname)
        if not h:
            continue
        fig_anim.add_trace(go.Scatter(
            x=h["epochs"], y=h[key],
            mode="lines+markers", name=mname,
            line=dict(color=color, width=3),
            marker=dict(size=7),
        ))
    fig_anim.update_layout(
        height=420,
        plot_bgcolor="#0d1f1b", paper_bgcolor="#0d1f1b",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#eafff6"),
        xaxis=dict(title="Epoch", dtick=1, color="#7fa596",
                   gridcolor="rgba(45,212,168,0.15)"),
        yaxis=dict(title=metric_choice, color="#7fa596",
                   gridcolor="rgba(45,212,168,0.15)"),
        legend=dict(bgcolor="rgba(13,31,27,0.8)", font=dict(color="#eafff6")),
        margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(fig_anim, use_container_width=True, config={"displayModeBar": False})

with hist_tabs[1]:
    fig3d = go.Figure()
    for mname, color in MODEL_COLORS.items():
        h = ALL_HISTORIES.get(mname)
        if not h:
            continue
        fig3d.add_trace(go.Scatter3d(
            x=h["epochs"], y=h["val_acc"], z=h["val_loss"],
            mode="lines+markers", name=mname,
            line=dict(color=color, width=5),
            marker=dict(size=4, color=color),
        ))
    fig3d.update_layout(
        height=520, paper_bgcolor="#0d1f1b",
        scene=dict(
            xaxis=dict(title="Epoch", backgroundcolor="#0d1f1b",
                       gridcolor="rgba(45,212,168,0.2)", color="#7fa596"),
            yaxis=dict(title="Val Accuracy", backgroundcolor="#0d1f1b",
                       gridcolor="rgba(45,212,168,0.2)", color="#7fa596"),
            zaxis=dict(title="Val Loss", backgroundcolor="#0d1f1b",
                       gridcolor="rgba(45,212,168,0.2)", color="#7fa596"),
        ),
        legend=dict(bgcolor="rgba(13,31,27,0.8)", font=dict(color="#eafff6")),
        font=dict(family="JetBrains Mono, monospace", color="#eafff6"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})
    st.caption("Rotate/zoom to inspect epoch trajectories across models.")

with hist_tabs[2]:
    fcols = st.columns(3)
    for i, mname in enumerate(MODEL_COLORS.keys()):
        h = ALL_HISTORIES.get(mname)
        with fcols[i]:
            st.markdown(f"**{mname}**")
            if h and h.get("val_acc"):
                st.metric("Final val. accuracy", f"{h['val_acc'][-1]*100:.1f}%")
                st.metric("Final val. loss", f"{h['val_loss'][-1]:.3f}")

st.markdown("</div>", unsafe_allow_html=True)

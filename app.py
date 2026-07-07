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
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        tickmode="linear", tick0=0, dtick=1,
                        color="#7fa596", gridcolor="rgba(45,212,168,0.08)",
                        title="Digit",
                    ),
                    yaxis=dict(
                        color="#7fa596", gridcolor="rgba(45,212,168,0.08)",
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
# Training Insights — animated epoch-by-epoch curves + 3D comparison surface.
#
# IMPORTANT: none of this is real training telemetry. The source notebooks
# never serialized history.history, so there is nothing real to load. These
# curves are synthetic, shaped to match documented MNIST benchmark behavior
# for each architecture (see simulated_history.py for exact assumptions).
# This is disclosed on-chart, not just in code, because a chart that looks
# like a real training log but isn't is a credibility problem if this app
# gets screenshotted or demoed without the surrounding conversation.
# ---------------------------------------------------------------------------
st.write("")
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("#### 📈 Training Insights")
st.markdown(
    '<div class="disclaimer-box">'
    '⚠ SIMULATED DATA — the original notebooks did not save real epoch-by-epoch '
    'training logs. Curves below are synthetic, shaped to match documented MNIST '
    'benchmark ranges for each architecture (Perceptron ~86-88%, ANN ~97-98%, '
    'CNN ~98-99% val. accuracy). Not measured from an actual training run.'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

hist_tabs = st.tabs(["Animated accuracy/loss", "3D comparison surface", "Final metrics"])

ALL_HISTORIES = {m: get_history(m) for m in MODEL_REGISTRY.keys()}
MODEL_COLORS = {"Perceptron": "#f5b942", "ANN": "#a78bfa", "CNN": "#2dd4a8"}

with hist_tabs[0]:
    metric_choice = st.radio(
        "Metric", ["Validation accuracy", "Validation loss"],
        horizontal=True, key="metric_choice",
    )
    key = "val_acc" if metric_choice == "Validation accuracy" else "val_loss"

    fig_anim = go.Figure()
    for mname, color in MODEL_COLORS.items():
        h = ALL_HISTORIES[mname]
        fig_anim.add_trace(go.Scatter(
            x=h["epochs"],
            y=h[key],
            mode="lines+markers",
            name=mname,
            line=dict(color=color, width=3),
            marker=dict(size=7),
        ))
    fig_anim.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#eafff6"),
        xaxis=dict(
            title="Epoch", dtick=1,
            color="#7fa596", gridcolor="rgba(45,212,168,0.08)",
        ),
        yaxis=dict(
            title=metric_choice,
            color="#7fa596", gridcolor="rgba(45,212,168,0.08)",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#eafff6")),
        margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(fig_anim, use_container_width=True, config={"displayModeBar": False})

with hist_tabs[1]:
    fig3d = go.Figure()
    for mname, color in MODEL_COLORS.items():
        h = ALL_HISTORIES[mname]
        fig3d.add_trace(go.Scatter3d(
            x=h["epochs"],
            y=h["val_acc"],
            z=h["val_loss"],
            mode="lines+markers",
            name=mname,
            line=dict(color=color, width=5),
            marker=dict(size=4, color=color),
        ))
    fig3d.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="Epoch", backgroundcolor="rgba(13,31,27,0.6)",
                       gridcolor="rgba(45,212,168,0.15)", color="#7fa596"),
            yaxis=dict(title="Val Accuracy", backgroundcolor="rgba(13,31,27,0.6)",
                       gridcolor="rgba(45,212,168,0.15)", color="#7fa596"),
            zaxis=dict(title="Val Loss", backgroundcolor="rgba(13,31,27,0.6)",
                       gridcolor="rgba(45,212,168,0.15)", color="#7fa596"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#eafff6")),
        font=dict(family="JetBrains Mono, monospace", color="#eafff6"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Each trajectory traces (epoch, validation accuracy, validation loss) "
        "for one model across simulated training. Rotate/zoom to inspect."
    )

with hist_tabs[2]:
    fcols = st.columns(3)
    for i, mname in enumerate(MODEL_COLORS.keys()):
        h = ALL_HISTORIES[mname]
        with fcols[i]:
            st.markdown(f"**{mname}**")
            st.metric("Final val. accuracy", f"{h['val_acc'][-1]*100:.1f}%")
            st.metric("Final val. loss", f"{h['val_loss'][-1]:.3f}")

st.markdown("</div>", unsafe_allow_html=True)

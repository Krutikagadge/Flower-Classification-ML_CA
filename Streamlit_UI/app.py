import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import base64
import mimetypes

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="FloraVision | AI Flower Identifier",
    page_icon="🌿",
    layout="wide"
)

# -------------------- CUSTOM STYLES --------------------
st.markdown("""
<style>
body {
    background-color: #f8faf5;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    color: #2e4a1f;
}

/* --- TABS STYLE --- */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: #f3f8f1;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: 600;
    color: #3b5323;
    transition: all 0.3s ease;
    font-size: 1.05rem;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #cdeac0;
    color: #1b5e20;
    transform: scale(1.05);
}
.stTabs [aria-selected="true"] {
    background-color: #2e7d32 !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(46,125,50,0.3);
}

/* --- HEADER BANNER --- */
.header-banner {
    background: linear-gradient(90deg, #2E7D32, #388E3C);
    color: white;
    text-align: center;
    padding: 2rem 0;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    margin-bottom: 2rem;
}
.header-banner h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.header-banner p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* --- COMPONENT STYLING --- */
.upload-box {
    border: 2px dashed #a5c882;
    background-color: #f6fbf4;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    transition: all 0.3s ease;
}
.upload-box:hover {
    border-color: #7aa874;
    background-color: #f0f8ec;
}
.pred-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fdf6 100%);
    border-left: 5px solid #7aa874;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}
.pred-card:hover {
    transform: translateY(-2px);
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f6fbf4 100%);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border: 1px solid #e8f5e8;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2e4a1f;
    margin: 0.5rem 0;
}
.metric-label {
    font-size: 0.9rem;
    color: #6b705c;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* --- FLOWER DATASET CARDS (global baseline) --- */
.flower-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    border: 1px solid #f0f0f0;
    /* allow height to adjust responsively */
    height: auto;
}
.flower-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}
.flower-card h4 {
    font-weight: 800;
    color: #2e4a1f;
    border-bottom: 2px solid #cdeac0;
    display: inline-block;
    margin: 0; /* tightened spacing */
    padding-bottom: 0.35rem;
    font-size: 1.02rem;
}

/* --- FOOTER --- */
.footer {
    text-align: center;
    color: #6b705c;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid #e8e8e8;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="header-banner">
    <h1>🌿 FloraVision</h1>
    <p>AI-Powered Flower Identification System</p>
</div>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    try:
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
    except Exception:
        cnn_model = None
    try:
        vgg_model = tf.keras.models.load_model("vgg_model.h5")
    except Exception:
        vgg_model = None
    return cnn_model, vgg_model

cnn_model, vgg_model = load_models()

# -------------------- CLASS NAMES --------------------
folder_to_flower = {
    0: "daffodil", 1: "snowdrop", 2: "lily_valley", 3: "bluebell",
    4: "crocus", 5: "iris", 6: "tigerlily", 7: "tulip",
    8: "fritillary", 9: "sunflower", 10: "daisy", 11: "coltsfoot",
    12: "dandelion", 13: "cowslip", 14: "buttercup", 15: "windflower", 16: "pansy"
}
class_names = [folder_to_flower[i] for i in range(17)]

# -------------------- PREDICTION FUNCTION --------------------
def predict_image(model, img, class_names, target_size=(128, 128)):
    if model is None:
        return "Unknown", 0.0, []
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)
    top3 = np.argsort(pred[0])[::-1][:3]
    return class_names[class_index], confidence, [(class_names[i], float(pred[0][i])) for i in top3]

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["🌿 Main Project", "📊 Dashboard", "🌼 Dataset Info"])

# -------------------- TAB 1 --------------------
with tab1:
    st.markdown("<h2 style='text-align:center;'>🌸 Flower Classification</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#6b705c;'>Upload an image and let AI identify the flower using CNN and VGG16 models.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload a flower image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(img, caption="🌼 Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("<div class='upload-box'>🧠 Analyzing image... please wait</div>", unsafe_allow_html=True)
            cnn_label, cnn_conf, cnn_top3 = predict_image(cnn_model, img, class_names)
            vgg_label, vgg_conf, vgg_top3 = predict_image(vgg_model, img, class_names)

            st.markdown(f"<div class='pred-card'><h3>🧩 CNN Prediction:</h3><b>{cnn_label.title()}</b> ({cnn_conf*100:.2f}%)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='pred-card'><h3>🌺 VGG16 Prediction:</h3><b>{vgg_label.title()}</b> ({vgg_conf*100:.2f}%)</div>", unsafe_allow_html=True)

            if cnn_label == vgg_label:
                st.success(f"✅ Both models agree: **{cnn_label.title()}**")
            else:
                st.warning(f"🤔 Models disagree:\n- CNN → `{cnn_label.title()}`\n- VGG16 → `{vgg_label.title()}`")
    else:
        st.markdown("<div class='upload-box'>📸 Drag & Drop or Browse an image to get started!</div>", unsafe_allow_html=True)

# -------------------- TAB 2 --------------------
with tab2:
    st.markdown("<h2 style='text-align:center;'>📊 Model Performance Dashboard</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("CNN Accuracy", "75%", "Test Dataset"),
        ("VGG16 Accuracy", "89%", "Test Dataset"),
        ("Total Species", "17", "Flower Types"),
        ("Training Samples", "1,360", "Total Images")
    ]
    for (label, value, desc), col in zip(metrics, [col1, col2, col3, col4]):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div style='color: #6b705c; font-size: 0.8rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📈 Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(5, 3))
        models = ["CNN", "VGG16"]
        acc = [75, 89]
        bars = ax.bar(models, acc, color=["#a7c957", "#2E7D32"], alpha=0.9)
        for bar, val in zip(bars, acc):
            ax.text(bar.get_x() + bar.get_width()/2., val + 1, f"{val}%", ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        st.pyplot(fig)

    with col4:
        st.subheader("📈 CNN vs VGG16 Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(5, 3))

        # Example data
        epochs = list(range(1, 21))
        cnn_train_acc = [0.15, 0.35, 0.52, 0.63, 0.70, 0.76, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.89, 0.90, 0.90, 0.91, 0.91, 0.92, 0.92]
        cnn_val_acc = [0.18, 0.40, 0.58, 0.63, 0.68, 0.70, 0.73, 0.74, 0.75, 0.75, 0.76, 0.76, 0.77, 0.77, 0.77, 0.78, 0.78, 0.78, 0.78, 0.78]
        vgg_train_acc = [0.25, 0.45, 0.60, 0.70, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.91, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93, 0.93, 0.93]
        vgg_val_acc = [0.30, 0.50, 0.65, 0.72, 0.78, 0.80, 0.83, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]

        # Plot lines
        ax.plot(epochs, cnn_train_acc, 'b--', label='CNN Train Accuracy')
        ax.plot(epochs, cnn_val_acc, 'orange', label='CNN Val Accuracy')
        ax.plot(epochs, vgg_train_acc, 'g--', label='VGG16 Train Accuracy')
        ax.plot(epochs, vgg_val_acc, 'r', label='VGG16 Val Accuracy')

        # Labels and title
        ax.set_title("CNN vs VGG16 Accuracy Comparison", fontsize=11, weight='bold')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

# -------------------- TAB 3 --------------------
with tab3:
    st.markdown("""
    <style>
    /* Ensure uniform image size only for Tab 3 */
    .dataset-tab .flower-card img {
        width: 100%;
        height: 230px;          /* fixed unified height */
        object-fit: cover;      /* keeps aspect ratio */
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .dataset-tab .flower-card img:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }

    /* Keep cards same height visually */
    .dataset-tab .flower-card {
        min-height: 310px; /* adjust card container height */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
    }
    </style>

    <div class="dataset-tab">
        <div style="background: linear-gradient(180deg, #f7fdf5 0%, #eef8ea 100%);
                    padding: 2rem; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
            <h2 style="text-align:center; font-size:2rem; color:#2e4a1f; text-shadow:1px 1px 2px rgba(46,125,50,0.2);">
                🌼 Oxford 17 Flower Dataset Overview
            </h2>
            <p style="text-align:center;color:#6b705c;font-size:1rem;margin-bottom:2rem;">
                17 flower categories, each with 80 images under various lighting, scale, and pose conditions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    dataset_path = "Result"
    cols = st.columns(4)

    def image_to_data_uri(path):
        if not os.path.exists(path):
            return None
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "image/jpeg"
        with open(path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f"data:{mime_type};base64,{b64}"

    for idx, flower in enumerate(class_names):
        col = cols[idx % 4]
        with col:
            image_path = os.path.join(dataset_path, f"{flower}.jpg")
            data_uri = image_to_data_uri(image_path)
            if data_uri:
                card_html = f"""
                <div class="flower-card-wrapper">
                  <div class="flower-card">
                    <h4 style="text-transform:capitalize;">{flower.replace('_',' ').title()}</h4>
                    <img src="{data_uri}" alt="{flower}" />
                  </div>
                </div>
                """
            else:
                card_html = f"""
                <div class="flower-card-wrapper">
                  <div class="flower-card">
                    <h4 style="text-transform:capitalize;">{flower.replace('_',' ').title()}</h4>
                    <div style="width:100%; height:220px; display:flex; align-items:center; justify-content:center; color:#6b705c;">
                      🌸 Image not available
                    </div>
                  </div>
                </div>
                """
            st.markdown(card_html, unsafe_allow_html=True)

        if (idx + 1) % 4 == 0 and idx < len(class_names) - 1:
            cols = st.columns(4)

    st.markdown("""
    <hr>
    <p style="text-align:center;color:#6b705c;padding:1rem;">
        📘 Dataset Source: <b>Oxford 17 Category Flower Dataset</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

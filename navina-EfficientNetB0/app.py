import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #fdf6f0 0%, #fef9f5 50%, #f5f0eb 100%);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #2d1b0e;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #9c7c6a;
    text-align: center;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Upload area */
.upload-hint {
    text-align: center;
    color: #b08a78;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    font-style: italic;
}

/* Result card */
.result-card {
    background: white;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    box-shadow: 0 8px 40px rgba(160, 100, 60, 0.10);
    border: 1px solid #f0e6de;
    margin-top: 1.5rem;
}
.flower-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #2d1b0e;
    text-transform: capitalize;
    margin-bottom: 0.2rem;
}
.flower-label {
    font-size: 0.8rem;
    color: #b08a78;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
    margin-bottom: 1rem;
}
.meaning-box {
    background: linear-gradient(135deg, #fdf0e8, #fdf6f2);
    border-left: 3px solid #e07b4f;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-style: italic;
    color: #5c3a28;
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
}
.confidence-label {
    font-size: 0.78rem;
    color: #9c7c6a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.divider {
    border: none;
    border-top: 1px solid #f0e6de;
    margin: 1.2rem 0;
}

/* Top 5 table */
.top5-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #2d1b0e;
    margin-bottom: 0.8rem;
}
.top5-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.55rem;
}
.rank-badge {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #f0e6de;
    color: #7a4f3a;
    font-size: 0.72rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.rank-badge.first {
    background: #e07b4f;
    color: white;
}
.bar-bg {
    flex: 1;
    background: #f5ede7;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #e07b4f, #f0a07a);
}
.bar-pct {
    font-size: 0.78rem;
    color: #9c7c6a;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}
.bar-name {
    font-size: 0.82rem;
    color: #5c3a28;
    width: 160px;
    flex-shrink: 0;
    text-transform: capitalize;
}

/* Incorrect badge */
.badge-correct {
    display: inline-block;
    background: #d4edda;
    color: #276937;
    border-radius: 99px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.5rem;
}
.badge-wrong {
    display: inline-block;
    background: #fde8e8;
    color: #a33;
    border-radius: 99px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Mappings ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dataset/model.keras")

@st.cache_data
def load_mappings():
    with open("cat_to_name.json") as f:
        cat_to_name = json.load(f)
    with open("dataset/class_names.json") as f:
        class_names = json.load(f)
    return cat_to_name, class_names

flower_meaning = {
    "pink primrose": "Young love", "hard-leaved pocket orchid": "Rare beauty",
    "canterbury bells": "Gratitude", "sweet pea": "Delicate pleasure",
    "english marigold": "Warmth", "tiger lily": "Confidence",
    "moon orchid": "Elegance", "bird of paradise": "Freedom",
    "monkshood": "Caution", "globe thistle": "Strength",
    "snapdragon": "Grace", "colt's foot": "Healing",
    "king protea": "Transformation", "spear thistle": "Protection",
    "yellow iris": "Passion", "globe-flower": "Joy",
    "purple coneflower": "Health", "peruvian lily": "Friendship",
    "balloon flower": "Honesty", "giant white arum lily": "Purity",
    "fire lily": "Passion", "pincushion flower": "Love",
    "fritillary": "Pride", "red ginger": "Prosperity",
    "grape hyacinth": "Trust", "corn poppy": "Remembrance",
    "prince of wales feathers": "Elegance", "stemless gentian": "Determination",
    "artichoke": "Hope", "sweet william": "Admiration",
    "carnation": "Love", "garden phlox": "Harmony",
    "love in the mist": "Mystery", "mexican aster": "Elegance",
    "alpine sea holly": "Independence", "ruby-lipped cattleya": "Luxury",
    "cape flower": "Beauty", "great masterwort": "Courage",
    "siam tulip": "Exotic beauty", "lenten rose": "Serenity",
    "barbeton daisy": "Cheerfulness", "daffodil": "New beginnings",
    "sword lily": "Strength", "poinsettia": "Celebration",
    "bolero deep blue": "Mystery", "wallflower": "Faithfulness",
    "marigold": "Creativity", "buttercup": "Joy",
    "oxeye daisy": "Innocence", "common dandelion": "Hope",
    "petunia": "Resentment", "wild pansy": "Thoughtfulness",
    "primula": "Youth", "sunflower": "Happiness",
    "pelargonium": "Comfort", "bishop of llandaff": "Elegance",
    "gaura": "Grace", "geranium": "Friendship",
    "pink-yellow dahlia": "Kindness", "orange dahlia": "Energy",
    "cautleya spicata": "Exotic", "japanese anemone": "Anticipation",
    "black-eyed susan": "Encouragement", "silverbush": "Protection",
    "californian poppy": "Rest", "osteospermum": "Optimism",
    "spring crocus": "Cheerfulness", "bearded iris": "Wisdom",
    "windflower": "Fragility", "tree poppy": "Peace",
    "gazania": "Wealth", "azalea": "Abundance",
    "water lily": "Enlightenment", "rose": "Love",
    "thorn apple": "Mystery", "morning glory": "Affection",
    "passion flower": "Faith", "lotus lotus": "Purity",
    "toad lily": "Uniqueness", "anthurium": "Hospitality",
    "frangipani": "Charm", "clematis": "Ingenuity",
    "hibiscus": "Delicate beauty", "columbine": "Foolishness",
    "tree mallow": "Gentleness", "magnolia": "Dignity",
    "cyclamen": "Sincerity", "watercress": "Vitality",
    "canna lily": "Confidence", "hippeastrum": "Pride",
    "bee balm": "Compassion", "ball moss": "Growth",
    "foxglove": "Healing", "bougainvillea": "Passion",
    "camellia": "Admiration", "mallow": "Softness",
    "mexican petunia": "Resilience", "bromelia": "Protection",
    "blanket flower": "Warmth", "trumpet creeper": "Energy",
    "blackberry lily": "Mystery"
}

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌸 Flower Identifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle"> Flower Classifier · MobileNetV2</div>', unsafe_allow_html=True)

# ─── Load resources ────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model = load_model()
    cat_to_name, class_names = load_mappings()

# ─── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Supports JPG, JPEG, PNG</p>', unsafe_allow_html=True)

# ─── Predict ───────────────────────────────────────────────────────────────────
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Uploaded image")

    # Preprocess — raw pixels, model handles rescaling internally
    img_resized = img.resize((224, 224))
    img_array  = np.array(img_resized, dtype=np.float32)
    img_array  = np.expand_dims(img_array, axis=0)

    with st.spinner("Identifying flower..."):
        pred = model.predict(img_array, verbose=0)

    pred_index   = int(np.argmax(pred))
    confidence   = float(np.max(pred)) * 100
    pred_label   = str(class_names[pred_index])
    pred_name    = cat_to_name.get(pred_label, "Unknown")
    pred_meaning = flower_meaning.get(pred_name.lower(), "Meaning not available")

    # Top 5
    top5_idx   = np.argsort(pred[0])[::-1][:5]
    top5_names = [cat_to_name.get(str(class_names[i]), "Unknown") for i in top5_idx]
    top5_confs = [float(pred[0][i]) * 100 for i in top5_idx]

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="flower-label">Identified as</div>
            <div class="flower-name">{pred_name}</div>
            <div class="meaning-box">✦ {pred_meaning}</div>
            <hr class="divider">
            <div class="confidence-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)
        st.markdown(f"**{confidence:.1f}%** confidence")

    # ── Top 5 breakdown ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="top5-title">🌿 Top 5 Predictions</div>', unsafe_allow_html=True)

    for rank, (name, conf) in enumerate(zip(top5_names, top5_confs), 1):
        badge_class = "first" if rank == 1 else ""
        bar_width   = conf / top5_confs[0] * 100  # relative to top prediction
        st.markdown(f"""
        <div class="top5-row">
            <div class="rank-badge {badge_class}">{rank}</div>
            <div class="bar-name">{name}</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{bar_width:.1f}%"></div></div>
            <div class="bar-pct">{conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a flower photo above to get started.")

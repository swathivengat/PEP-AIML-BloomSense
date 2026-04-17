import json
import os
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("dataset/model.keras")

# Load label → name mapping
with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)

# Load class order (index → label)
with open("dataset/class_names.json", "r") as f:
    class_names = json.load(f)

flower_meaning = {
    "pink primrose": "Young love",
    "hard-leaved pocket orchid": "Rare beauty",
    "canterbury bells": "Gratitude",
    "sweet pea": "Delicate pleasure",
    "english marigold": "Warmth",
    "tiger lily": "Confidence",
    "moon orchid": "Elegance",
    "bird of paradise": "Freedom",
    "monkshood": "Caution",
    "globe thistle": "Strength",
    "snapdragon": "Grace",
    "colt's foot": "Healing",
    "king protea": "Transformation",
    "spear thistle": "Protection",
    "yellow iris": "Passion",
    "globe-flower": "Joy",
    "purple coneflower": "Health",
    "peruvian lily": "Friendship",
    "balloon flower": "Honesty",
    "giant white arum lily": "Purity",
    "fire lily": "Passion",
    "pincushion flower": "Love",
    "fritillary": "Pride",
    "red ginger": "Prosperity",
    "grape hyacinth": "Trust",
    "corn poppy": "Remembrance",
    "prince of wales feathers": "Elegance",
    "stemless gentian": "Determination",
    "artichoke": "Hope",
    "sweet william": "Admiration",
    "carnation": "Love",
    "garden phlox": "Harmony",
    "love in the mist": "Mystery",
    "mexican aster": "Elegance",
    "alpine sea holly": "Independence",
    "ruby-lipped cattleya": "Luxury",
    "cape flower": "Beauty",
    "great masterwort": "Courage",
    "siam tulip": "Exotic beauty",
    "lenten rose": "Serenity",
    "barbeton daisy": "Cheerfulness",
    "daffodil": "New beginnings",
    "sword lily": "Strength",
    "poinsettia": "Celebration",
    "bolero deep blue": "Mystery",
    "wallflower": "Faithfulness",
    "marigold": "Creativity",
    "buttercup": "Joy",
    "oxeye daisy": "Innocence",
    "common dandelion": "Hope",
    "petunia": "Resentment",
    "wild pansy": "Thoughtfulness",
    "primula": "Youth",
    "sunflower": "Happiness",
    "pelargonium": "Comfort",
    "bishop of llandaff": "Elegance",
    "gaura": "Grace",
    "geranium": "Friendship",
    "pink-yellow dahlia": "Kindness",
    "orange dahlia": "Energy",
    "cautleya spicata": "Exotic",
    "japanese anemone": "Anticipation",
    "black-eyed susan": "Encouragement",
    "silverbush": "Protection",
    "californian poppy": "Rest",
    "osteospermum": "Optimism",
    "spring crocus": "Cheerfulness",
    "bearded iris": "Wisdom",
    "windflower": "Fragility",
    "tree poppy": "Peace",
    "gazania": "Wealth",
    "azalea": "Abundance",
    "water lily": "Enlightenment",
    "rose": "Love",
    "thorn apple": "Mystery",
    "morning glory": "Affection",
    "passion flower": "Faith",
    "lotus lotus": "Purity",
    "toad lily": "Uniqueness",
    "anthurium": "Hospitality",
    "frangipani": "Charm",
    "clematis": "Ingenuity",
    "hibiscus": "Delicate beauty",
    "columbine": "Foolishness",
    "tree mallow": "Gentleness",
    "magnolia": "Dignity",
    "cyclamen": "Sincerity",
    "watercress": "Vitality",
    "canna lily": "Confidence",
    "hippeastrum": "Pride",
    "bee balm": "Compassion",
    "ball moss": "Growth",
    "foxglove": "Healing",
    "bougainvillea": "Passion",
    "camellia": "Admiration",
    "mallow": "Softness",
    "mexican petunia": "Resilience",
    "bromelia": "Protection",
    "blanket flower": "Warmth",
    "trumpet creeper": "Energy",
    "blackberry lily": "Mystery"
}

# ─── Image path ────────────────────────────────────────────────────────────────
img_path = "dataset/test/11/image_03114.jpg"

# Load and preprocess image
# Do NOT divide by 255 manually — ResNet50's preprocess_input is baked into the model
# It will subtract ImageNet channel means automatically
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)   # raw [0, 255] pixels
img_array = np.expand_dims(img_array, axis=0)                # shape: (1, 224, 224, 3)

# Predict
pred = model.predict(img_array, verbose=0)
pred_index = np.argmax(pred)
confidence = float(np.max(pred)) * 100

# class_names[pred_index] gives the folder name (e.g., "11")
# look it up in cat_to_name to get the flower name
pred_label = class_names[pred_index]
pred_name  = cat_to_name.get(str(pred_label), "Unknown")

# Get meaning
pred_meaning = flower_meaning.get(pred_name.lower(), "Meaning not available")

# Actual label from folder structure
true_label  = os.path.basename(os.path.dirname(img_path))
true_name   = cat_to_name.get(str(true_label), "Unknown")
true_meaning = flower_meaning.get(true_name.lower(), "Meaning not available")

# Debug info
print("Model         :", "ResNet50")
print("Output shape  :", model.output_shape)
print("Total classes :", len(class_names))
print(f"Pred index    : {pred_index} → label: '{pred_label}' → name: '{pred_name}'")

# Output
print("\n--- Prediction Result ---")
print(f"Image: {img_path}")

print(f"\nPredicted (confidence: {confidence:.1f}%):")
print(f"  Label  : {pred_label}")
print(f"  Flower : {pred_name}")
print(f"  Meaning: {pred_meaning}")

print(f"\nActual:")
print(f"  Label  : {true_label}")
print(f"  Flower : {true_name}")
print(f"  Meaning: {true_meaning}")

print(f"\n{'✅ CORRECT' if pred_name == true_name else '❌ INCORRECT'}")

# Top 5 predictions
top5_idx = np.argsort(pred[0])[::-1][:5]
print("\n--- Top 5 Predictions ---")
for rank, idx in enumerate(top5_idx, 1):
    lbl  = class_names[idx]
    name = cat_to_name.get(str(lbl), "Unknown")
    conf = pred[0][idx] * 100
    print(f"  {rank}. {name:30s}  ({conf:.1f}%)")

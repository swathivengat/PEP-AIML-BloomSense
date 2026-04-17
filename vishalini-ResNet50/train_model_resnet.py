import tensorflow as tf
import json

# Paths
train_dir = "archive/102 flower/flowers/train"
val_dir = "archive/102 flower/flowers/valid"
test_dir = "archive/102 flower/flowers/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Save class names so predict.py can use the same order
with open("archive/class_names.json", "w") as f:
    json.dump(class_names, f)

# Performance tuning
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation (raw pixel values — preprocessing happens inside model)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# ✅ CHANGED: ResNet50 preprocessor
# ResNet expects zero-centered channels (subtracts ImageNet mean per channel)
# NOT [-1, 1] like MobileNetV2 — each model has its own unique formula
preprocess_input = tf.keras.applications.resnet.preprocess_input

# ✅ CHANGED: ResNet50 base model (replaces MobileNetV2)
# ResNet50 has 175 layers and 25.6M parameters vs MobileNetV2's 53 layers and 3.4M
# More powerful → expects ~93-94% accuracy vs 91%
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,      # remove ResNet's original 1000-class top layer
    weights='imagenet'      # use pretrained ImageNet weights
)
base_model.trainable = False  # freeze during Phase 1

# Build Model — same structure as before, only base_model changed
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Phase 1: Train top layers only (base frozen) ---
print("\n=== Phase 1: Training top layers ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --- Phase 2: Fine-tune last 30 layers of ResNet50 ---
print("\n=== Phase 2: Fine-tuning ResNet50 ===")
base_model.trainable = True

# ✅ CHANGED: ResNet50 has 175 layers total
# We freeze all except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Total ResNet50 layers: {len(base_model.layers)}")
print(f"Trainable layers: {sum(1 for l in base_model.layers if l.trainable)}")

# Recompile with very low learning rate for fine-tuning
# Must recompile after changing trainable layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    initial_epoch=history.epoch[-1]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save model
model.save("archive/model.keras")
print("Model saved as dataset/model.keras")

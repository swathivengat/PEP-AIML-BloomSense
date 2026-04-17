import tensorflow as tf
import json

# Paths
train_dir = "dataset/train"
val_dir   = "dataset/val"
test_dir  = "dataset/test"

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

# FIX 1: Save class names so predict.py can use the same order
with open("dataset/class_names.json", "w") as f:
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

# FIX 2: Use MobileNetV2's own preprocess_input instead of plain Rescaling(1/255)
# MobileNetV2 expects input in [-1, 1], not [0, 1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Base Model (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze during initial training

# Build Model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)            # FIX 2: correct preprocessing
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

# --- Phase 1: Train with frozen base ---
print("\n=== Phase 1: Training top layers ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --- Phase 2: Fine-tune — unfreeze top layers of base model ---
# FIX 3: Fine-tuning dramatically improves accuracy
print("\n=== Phase 2: Fine-tuning top layers of MobileNetV2 ===")
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
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

# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

model.save("dataset/model.keras")
print("Model saved as dataset/model.keras")

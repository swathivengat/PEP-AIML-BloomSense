import os
import shutil
import random

# Original dataset path (current)
source_dir = "dataset/train"

# New split paths
base_dir = "dataset"

train_dir = os.path.join(base_dir, "train_split")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Create folders
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for class_name in os.listdir(source_dir):

    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {class_name}")

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)

    train_split = int(train_ratio * total)
    val_split = int(val_ratio * total)

    train_images = images[:train_split]
    val_images = images[train_split:train_split + val_split]
    test_images = images[train_split + val_split:]

    # Create class folders
    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(val_dir, class_name, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(test_dir, class_name, img))

    print(f"{class_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

print("\nDataset split complete")
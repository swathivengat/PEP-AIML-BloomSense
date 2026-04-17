import os
import random
from PIL import Image

base_dir = "dataset/train"   # adjust if needed

IMG_SIZE = (224, 224)
TARGET_COUNT = 50

for class_name in os.listdir(base_dir):

    class_path = os.path.join(base_dir, class_name)

    # Skip non-folders
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {class_name}")

    images = os.listdir(class_path)
    valid_images = []

    # Resize images
    for img_name in images:
        img_path = os.path.join(class_path, img_name)

        # ✅ Skip if it's not a file
        if not os.path.isfile(img_path):
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(img_path, "JPEG")

                valid_images.append(img_name)

        except Exception as e:
            print(f"Skipping: {img_name}")

    print(f"Valid images: {len(valid_images)}")

    # Limit to TARGET_COUNT
    if len(valid_images) > TARGET_COUNT:

        extra = len(valid_images) - TARGET_COUNT
        print(f"Removing {extra} extra images...")

        remove_images = random.sample(valid_images, extra)

        for img_name in remove_images:
            img_path = os.path.join(class_path, img_name)

            try:
                os.remove(img_path)
            except:
                print(f"Could not delete: {img_name}")

        print(f"{class_name} → reduced to {TARGET_COUNT}")

    else:
        print(f"{class_name} → {len(valid_images)} images (no deletion)")

print("\nDataset preparation complete")
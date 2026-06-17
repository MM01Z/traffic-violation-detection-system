import os
import cv2
from ultralytics import YOLO

# ====== CONFIG ======
MODEL_PATH = "models/custom2.pt"          # your trained model
INPUT_DIR = "data/images/"      # folder with images
OUTPUT_DIR = "violations/accident(test_images)"    # where results will be saved
CONF_THRESHOLD = 0.25           # adjust if needed

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Process images
for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)

    # Skip non-images
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", "avif", "webp")):
        continue

    # Run inference
    results = model(img_path, conf=CONF_THRESHOLD)

    # Get annotated image
    annotated_img = results[0].plot()

    # Save output
    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, annotated_img)

    print(f"Saved: {save_path}")

print("Done.")
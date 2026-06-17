import os
import cv2
import numpy as np
import shutil
from datetime import datetime
import mysql.connector
from paddleocr import PaddleOCR
from db_config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

# --------------------------
# Folder structure
# --------------------------

#No-helmet
RAW_BASE = "violations"
PROCESSED_BASE = "processed"

RIDER_RAW = os.path.join(RAW_BASE, "nohelmet")
PLATE_RAW = os.path.join(RAW_BASE, "plates")

RIDER_PROCESSED = os.path.join(PROCESSED_BASE, "nohelmet", "rider")
PLATE_PROCESSED = os.path.join(PROCESSED_BASE, "nohelmet", "plates")

os.makedirs(RIDER_PROCESSED, exist_ok=True)
os.makedirs(PLATE_PROCESSED, exist_ok=True)

#Accident detection
ACCIDENT_RAW = os.path.join(RAW_BASE, "accident")
ACCIDENT_PROCESSED = os.path.join(PROCESSED_BASE, "accident")

os.makedirs(ACCIDENT_PROCESSED, exist_ok=True)

#Triple-riding
TRIPLE_RAW = os.path.join(RAW_BASE, "triple_riding")

TRIPLE_RIDER_PROCESSED = os.path.join(PROCESSED_BASE, "triple_riding", "rider")
TRIPLE_PLATE_PROCESSED = os.path.join(PROCESSED_BASE, "triple_riding", "plates")

os.makedirs(TRIPLE_RIDER_PROCESSED, exist_ok=True)
os.makedirs(TRIPLE_PLATE_PROCESSED, exist_ok=True)


# Missing number plate
CAR_RAW = os.path.join(RAW_BASE, "cars")

MISSING_RIDER_PROCESSED = os.path.join(PROCESSED_BASE, "missing_numberplate", "rider")
MISSING_CAR_PROCESSED = os.path.join(PROCESSED_BASE, "missing_numberplate", "car")

os.makedirs(MISSING_RIDER_PROCESSED, exist_ok=True)
os.makedirs(MISSING_CAR_PROCESSED, exist_ok=True)

# --------------------------
# Folder structure
# --------------------------

MISSING_HELMET_CHALLAN_PRICE = 1000
TRIPLE_RIDING_CHALLAN_PRICE = 1500

ocr = PaddleOCR(use_angle_cls=True, lang="en")

conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)
cursor = conn.cursor()

# --------------------------
# OCR func
# --------------------------
def extract_ocr_text(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        return "not scanned"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([
        [0, -0.25, 0],
        [-0.25, 2, -0.25],
        [0, -0.25, 0]
    ])
    sharpened = cv2.filter2D(gray, -1, kernel)

    if debug:
        cv2.imshow("Enhanced Plate", sharpened)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    try:
        result = ocr.ocr(sharpened)
        if not result:
            return "not scanned"

        texts = [line[1][0] for group in result for line in group]
        return " ".join(texts) if texts else "not scanned"

    except Exception as e:
        print(f"OCR error for {img_path}: {e}")
        return "not scanned"
# --------------------------
# OCR func
# --------------------------

# --------------------------
# Logic
# --------------------------
#No-Helmet
def process_nohelmet():
    if not os.path.exists(RIDER_RAW):
        return

    for filename in os.listdir(RIDER_RAW):
        if not filename.endswith(".jpg"):
            continue

        rider_src = os.path.join(RIDER_RAW, filename)
        plate_src = os.path.join(PLATE_RAW, filename)

        rider_dst = os.path.join(RIDER_PROCESSED, filename)
        plate_dst = os.path.join(PLATE_PROCESSED, filename)

        # OCR only if plate exists
        if os.path.exists(plate_src):
            ocr_text = extract_ocr_text(plate_src)
            shutil.copy2(plate_src, plate_dst)  # Copy plate
        else:
            ocr_text = "not scanned"
            plate_dst = None

        # Move rider image
        shutil.move(rider_src, rider_dst)

        # Insert DB row
        cursor.execute("""
            INSERT INTO violations
            (violation_type, violation_datetime, violation_image, plate_image, ocr_text, challan_sent, challan_price)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            "nohelmet",
            datetime.now(),
            rider_dst,   # now goes into violation_image
            plate_dst,
            ocr_text,
            False,
            MISSING_HELMET_CHALLAN_PRICE
        ))

        conn.commit()

    # --------------------------
    # Cleanup raw folders after processing
    # --------------------------
    for folder in [RIDER_RAW, PLATE_RAW]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    os.remove(path)
    print("Raw rider and plate folders cleaned up.")
    print("No-helmet post-processing complete.")
#No-Helmet

#triple riding
def process_triple_riding():
    if not os.path.exists(TRIPLE_RAW):
        return

    for filename in os.listdir(TRIPLE_RAW):
        if not filename.endswith(".jpg"):
            continue

        rider_src = os.path.join(TRIPLE_RAW, filename)
        plate_src = os.path.join(PLATE_RAW, filename)

        # Keep same filename (e.g., rider_5.jpg)
        rider_dst = os.path.join(TRIPLE_RIDER_PROCESSED, filename)
        plate_dst = os.path.join(TRIPLE_PLATE_PROCESSED, filename)

        # OCR if plate exists
        if os.path.exists(plate_src):
            ocr_text = extract_ocr_text(plate_src)
            shutil.copy2(plate_src, plate_dst)  # overwrite allowed
        else:
            ocr_text = "not scanned"
            plate_dst = None

        # Move rider image (overwrite allowed)
        shutil.move(rider_src, rider_dst)

        # Insert into DB
        cursor.execute("""
            INSERT INTO violations
            (violation_type, violation_datetime, violation_image, plate_image, ocr_text, challan_sent, challan_price)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            "triple_riding",
            datetime.now(),
            rider_dst,
            plate_dst,
            ocr_text,
            False,
            TRIPLE_RIDING_CHALLAN_PRICE
        ))

        conn.commit()

    print("Triple riding post-processing complete.")
#triple riding

# Accident
import uuid

def process_accidents():
    if not os.path.exists(ACCIDENT_RAW):
        return

    for filename in os.listdir(ACCIDENT_RAW):
        if not filename.endswith(".jpg"):
            continue

        src_path = os.path.join(ACCIDENT_RAW, filename)

        # Generate unique filename
        unique_name = f"accident_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        dst_path = os.path.join(ACCIDENT_PROCESSED, unique_name)

        # Move file (rename during move)
        shutil.move(src_path, dst_path)

        cursor.execute("""
            INSERT INTO violations
            (violation_type, violation_datetime, violation_image, plate_image, ocr_text, challan_sent, challan_price)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            "accident",
            datetime.now(),
            dst_path,   # accident frame
            None,
            "accident",
            False,
            0
        ))

        conn.commit()

    print("Accident post-processing complete.")
# Accident








#missing_plate
def process_missing_numberplate():

    # --------------------------
    # Riders (shared source → COPY)
    # --------------------------
    if os.path.exists(RIDER_RAW):

        for filename in os.listdir(RIDER_RAW):
            if not filename.endswith(".jpg"):
                continue

            plate_path = os.path.join(PLATE_RAW, filename)

            # If plate exists, skip (not a missing plate case)
            if os.path.exists(plate_path):
                continue

            rider_src = os.path.join(RIDER_RAW, filename)
            rider_dst = os.path.join(MISSING_RIDER_PROCESSED, filename)

            # COPY because rider image is reused by nohelmet pipeline
            shutil.copy2(rider_src, rider_dst)

            cursor.execute("""
                INSERT INTO violations
                (violation_type, violation_datetime, violation_image,
                 plate_image, ocr_text, challan_sent, challan_price)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                "missing_numberplate",
                datetime.now(),
                rider_dst,
                None,
                "plate missing",
                False,
                0
            ))

            conn.commit()

    # --------------------------
    # Cars (single-use batch → MOVE)
    # --------------------------
    if os.path.exists(CAR_RAW):

        for filename in os.listdir(CAR_RAW):
            if not filename.endswith(".jpg"):
                continue

            plate_path = os.path.join(PLATE_RAW, filename)

            # If plate exists, skip
            if os.path.exists(plate_path):
                continue

            car_src = os.path.join(CAR_RAW, filename)
            car_dst = os.path.join(MISSING_CAR_PROCESSED, filename)

            # MOVE because cars folder is batch-consumed
            shutil.move(car_src, car_dst)

            cursor.execute("""
                INSERT INTO violations
                (violation_type, violation_datetime, violation_image,
                 plate_image, ocr_text, challan_sent, challan_price)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                "missing_numberplate",
                datetime.now(),
                car_dst,
                None,
                "plate missing",
                False,
                0
            ))

            conn.commit()

    print("Missing number plate post-processing complete.")
#missing_plate






# --------------------------
# Logic
# --------------------------

if __name__ == "__main__":

    process_accidents()

    process_triple_riding()

    process_missing_numberplate()

    process_nohelmet()
    
    cursor.close()
    conn.close()

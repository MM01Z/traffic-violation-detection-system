# violations.py
import os
from utils import is_contained, crop_and_save, save_best_crop
import cv2


# --------------------------
# Folder structure
# --------------------------
#Helmet violation
BASE_DIR = "violations"
NOHELMET_DIR = os.path.join(BASE_DIR, "nohelmet")
PLATE_DIR = os.path.join(BASE_DIR, "plates")

# Create folders if they don't exist
os.makedirs(NOHELMET_DIR, exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)

#Accident detection
ACCIDENT_DIR = os.path.join(BASE_DIR, "accident")
os.makedirs(ACCIDENT_DIR, exist_ok=True)

#Triple riding detection
TRIPLE_DIR = os.path.join(BASE_DIR, "triple_riding")
os.makedirs(TRIPLE_DIR, exist_ok=True)

#missing plate for cars
CAR_DIR = os.path.join(BASE_DIR, "cars")
os.makedirs(CAR_DIR, exist_ok=True)
# --------------------------
# Folder structure
# --------------------------


# --------------------------
# Logic
# --------------------------
#Accident_detection logic
def check_accident(frame, detections):
    """
    Save full frame whenever 'accident' is detected.
    Overwrites the same file each time.
    """
    accident_detected = any(d["cls"] == "accident" for d in detections)

    if accident_detected:
        save_path = os.path.join(ACCIDENT_DIR, "accident.jpg")
        cv2.imwrite(save_path, frame)

#Helmet violation logic
def check_no_helmet(frame, detections):
    """
    Save rider/no-helmet crops and associated plate crops.
    Plate is saved only if it is inside the rider BB.
    """
    riders = [d for d in detections if d["cls"] == "rider"]
    nohelmets = [d for d in detections if d["cls"] == "nohelmet"]
    plates = [d for d in detections if d["cls"] == "numberplate"]

    for rider in riders:
        rider_bbox = rider["bbox"]
        rider_id = rider["track_id"]
        if rider_id is None:
            continue

        # Check if any nohelmet bbox is inside rider bbox
        violation = any(is_contained(nh["bbox"], rider_bbox) for nh in nohelmets)
        if not violation:
            continue

        # Save rider crop (overwrite per track ID)
        rider_path = os.path.join(NOHELMET_DIR, f"{rider_id}.jpg")
        crop_and_save(frame, rider_bbox, rider_path)

#Triple riding logic
def check_triple_riding(frame, detections):
    riders = [d for d in detections if d["cls"] == "rider"]
    persons = [d for d in detections if d["cls"] == "person"]

    for rider in riders:
        rider_bbox = rider["bbox"]
        rider_id = rider["track_id"]

        if rider_id is None:
            continue

        rx1, ry1, rx2, ry2 = rider_bbox
        rider_area = (rx2 - rx1) * (ry2 - ry1)

        count = 0

        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            person_area = (px2 - px1) * (py2 - py1)

            # Use default containment (threshold = 0.7)
            if not is_contained(p["bbox"], rider_bbox, threshold=0.2):
                continue

            # Size filter (remove distant small persons)
            if person_area < 0.1 * rider_area:
                continue

            count += 1

        if count >= 3:
            save_path = os.path.join(TRIPLE_DIR, f"{rider_id}.jpg")

            save_best_crop(
                frame,
                rider_bbox,
                rider_id,
                save_path
            )

#car image save for missing plate
def save_cars(frame, detections):
    """
    Save reasonably sized vehicles for plate detection
    Uses frame-relative area threshold instead of fixed pixel value
    """

    vehicles = [d for d in detections if d["entity"] == "vehicle"]

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # Minimum 2% of frame area
    min_car_area = 0.1 * frame_area

    for v in vehicles:
        bbox = v["bbox"]
        track_id = v["track_id"]

        if track_id is None:
            continue

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        # Skip small/distant vehicles
        if area < min_car_area:
            continue

        save_path = os.path.join(CAR_DIR, f"{track_id}.jpg")

        save_best_crop(
            frame,
            bbox,
            track_id,
            save_path
        )

#Plate save - for rider
# def save_plate(frame, detections, entity_cls, save_dir):
#     """
#     Save plates inside the bounding boxes of a given entity (rider or vehicle)

#     frame       : current frame
#     detections  : list of detection dicts
#     entity_cls  : class of the entity to check containment for ("rider" or "vehicle")
#     save_dir    : folder to save plates
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     entities = [d for d in detections if d["entity"] == entity_cls]
#     plates   = [d for d in detections if d["cls"] == "numberplate"]

#     for entity in entities:
#         entity_bbox = entity["bbox"]
#         track_id = entity["track_id"]
#         if track_id is None:
#             continue

#         for plate in plates:
#             if is_contained(plate["bbox"], entity_bbox):
#                 plate_path = os.path.join(save_dir, f"{track_id}.jpg")
#                 save_best_crop(
#                     frame,
#                     plate["bbox"],
#                     track_id,
#                     plate_path
#                 )

#save plate - vehicle
def save_vehicle_plates(frame, detections):
    vehicles = [d for d in detections if d["entity"] == "vehicle"]
    riders = [d for d in detections if d["entity"] == "rider"]
    plates = [d for d in detections if d["cls"] == "numberplate"]

    for plate in plates:
        plate_bbox = plate["bbox"]

        # Step 1: Skip if plate belongs to a rider
        if is_plate_in_rider(plate_bbox, riders):
            continue

        # Step 2: Assign to vehicle
        for v in vehicles:
            if is_contained(plate_bbox, v["bbox"]):
                track_id = v["track_id"]

                if track_id is None:
                    continue

                save_path = os.path.join(PLATE_DIR, f"{track_id}.jpg")

                save_best_crop(
                    frame,
                    plate_bbox,
                    track_id,
                    save_path
                )
                break
#utility
def is_plate_in_rider(plate_bbox, riders):
    for r in riders:
        if is_contained(plate_bbox, r["bbox"]):
            return True
    return False

#test
def save_plate(frame, detections, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    riders   = [d for d in detections if d["entity"] == "rider"]
    vehicles = [d for d in detections if d["entity"] == "vehicle"]
    plates   = [d for d in detections if d["cls"] == "numberplate"]

    for plate in plates:
        plate_bbox = plate["bbox"]

        assigned = False

        # 1. Check rider first (priority)
        for r in riders:
            if is_contained(plate_bbox, r["bbox"]):
                track_id = r["track_id"]
                if track_id:
                    save_best_crop(frame, plate_bbox, track_id,
                                   os.path.join(save_dir, f"{track_id}.jpg"))
                assigned = True
                break

        if assigned:
            continue

        # 2. Fallback to vehicle
        for v in vehicles:
            if is_contained(plate_bbox, v["bbox"]):
                track_id = v["track_id"]
                if track_id:
                    save_best_crop(frame, plate_bbox, track_id,
                                   os.path.join(save_dir, f"{track_id}.jpg"))
                break
    
# --------------------------
# Logic
# --------------------------
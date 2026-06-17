# utils.py
import cv2

def is_contained(inner, outer, threshold=0.7):
    """
    Check if inner bbox is mostly inside outer bbox by at least a threshold.
    inner, outer: [x1, y1, x2, y2]
    threshold: fraction of inner bbox area that must overlap outer bbox
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    # Compute intersection
    x_left = max(ix1, ox1)
    y_top = max(iy1, oy1)
    x_right = min(ix2, ox2)
    y_bottom = min(iy2, oy2)

    if x_right <= x_left or y_bottom <= y_top:
        return False  # no overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (ix2 - ix1) * (iy2 - iy1)

    return (intersection_area / inner_area) >= threshold


#=====================================================
#For Rider/Vehicle

def crop_and_save(frame, bbox, save_path):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        cv2.imwrite(save_path, crop)


#=====================================================
#Experiment

BEST_CROP_SCORE = {}

def save_best_crop(frame, bbox, track_id, save_path):
    x1, y1, x2, y2 = map(int, bbox)
    area = (x2 - x1) * (y2 - y1)

    prev_best = BEST_CROP_SCORE.get(track_id, 0)

    if area <= prev_best:
        return  # not better

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return

    cv2.imwrite(save_path, crop)
    BEST_CROP_SCORE[track_id] = area

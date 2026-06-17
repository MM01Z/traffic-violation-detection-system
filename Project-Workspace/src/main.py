# main.py
import cv2
from ultralytics import YOLO
from violations import check_no_helmet, save_plate, check_accident, check_triple_riding, save_cars, save_vehicle_plates

# --------------------------
# Load models
# --------------------------
model_custom = YOLO("models/custom2.pt")        # rider, helmet, nohelmet, numberplate
model_default = YOLO("models/yolov8n.pt")    # car, bus, truck, person, traffic light

# --------------------------
# Video capture
# --------------------------
cap = cv2.VideoCapture("data/videos/video2.mp4")
orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps         = cap.get(cv2.CAP_PROP_FPS)

# Output video writer (full resolution)
out = cv2.VideoWriter('data/output/output_with_bb.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (orig_width, orig_height))

# Max display size
MAX_WIDTH = 640
MAX_HEIGHT = 640

# --------------------------
# Frame skipping configuration
# --------------------------
frame_counter = 0
VIOLATION_CHECK_FREQ = 5  # run helmet check every n frames

# --------------------------
# Main loop
# --------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    unified_detections = []

    # ---- Custom model inference with tracking ----
    results_custom = model_custom.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
    if results_custom and results_custom[0].boxes is not None:
        boxes = results_custom[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model_custom.names[cls_id]
            track_id = f"rider_{int(box.id[0])}" if box.id is not None and cls_name == "rider" else None
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            unified_detections.append({
                "bbox": [x1, y1, x2, y2],
                "cls": cls_name,
                "entity": "rider" if cls_name == "rider" else cls_name,
                "track_id": track_id,
                "source": "best"
            })

    # ---- Default model inference with tracking ----
    results_default = model_default.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
    if results_default and results_default[0].boxes is not None:
        boxes = results_default[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model_default.names[cls_id]
            track_id = f"vehicle_{int(box.id[0])}" if box.id is not None and cls_name in ["car","bus","truck"] else None
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            unified_detections.append({
                "bbox": [x1, y1, x2, y2],
                "cls": cls_name,
                "entity": "vehicle" if cls_name in ["car","bus","truck"] else cls_name,
                "track_id": track_id,
                "source": "yolov8n"
            })

    # ---- Make a clean copy for saving ----
    clean_frame = frame.copy()

    # ---- Call violation functions every n frames ----
    if frame_counter % VIOLATION_CHECK_FREQ == 0:
        # Use clean_frame to save crops without BBs

        check_no_helmet(clean_frame, unified_detections)

        #save_plate(clean_frame, unified_detections, "rider", "violations/plates/")

        #save_plate(clean_frame, unified_detections, "vehicle", "violations/plates/")

        save_plate(clean_frame, unified_detections, "violations/plates/")

        check_accident(clean_frame, unified_detections)

        check_triple_riding(clean_frame, unified_detections) 

        save_cars(clean_frame, unified_detections)

        # In future: Vehicle plates
        # save_plate(clean_frame, unified_detections, "vehicle", "violations/plates/")
        # save_plate(clean_frame, unified_detections, "violations/plates/")
        # Future: check_triple_riding(frame, unified_detections)
        # check_missing_plate(frame, unified_detections)
        # check_accident(frame, unified_detections)

        #save_vehicle_plates(clean_frame, unified_detections)

    # ---- Draw bounding boxes and labels on original frame ----
    for det in unified_detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det["cls"]
        color = (0, 255, 0)  # default green
        if det["entity"] == "rider":
            color = (0, 0, 255)  # red
        elif det["entity"] == "vehicle":
            color = (255, 0, 0)  # blue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}:{det['track_id'] if det['track_id'] else ''}"
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ---- Resize frame for display ----
    display_frame = frame.copy()
    scale_w = MAX_WIDTH / orig_width
    scale_h = MAX_HEIGHT / orig_height
    scale = min(scale_w, scale_h, 1.0)  # never upscale
    display_frame = cv2.resize(display_frame, (int(orig_width*scale), int(orig_height*scale)))

    # ---- Show frame live ----
    cv2.imshow("BB Video", display_frame)
    out.write(frame)  # write full resolution frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------
# Cleanup
# --------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

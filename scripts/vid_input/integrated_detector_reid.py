# integrated_bytetrack_dual.py
"""
Run ByteTrack on the suspicious model for stable IDs, and run the weapon/object model
on each frame to draw all boxes. Save annotated video + JSON.
"""

from ultralytics import YOLO
import cv2, json
import numpy as np

# -------------------- CONFIG --------------------
YOLO_SUSPICIOUS = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\models\suspicious_model4\weights\best.pt"
YOLO_WEAPON     = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\models\new_project\weights\best.pt"

INPUT_VIDEO     = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\test_videos\bodycam.mp4"
OUTPUT_VIDEO    = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\results\videos\output_bytetrack_combined.mp4"
OUTPUT_JSON     = r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\results\vid_json\output_bytetrack_combined.json"

CONF_SUSPICIOUS = 0.35   # detection threshold for suspicious model (tracker)
CONF_WEAPON     = 0.35   # detection threshold for weapon model
TARGET_LABELS   = {"suspect", "victim", "terrorist"}  # substring match (lowercase)
TRACKER_CFG     = "bytetrack.yaml"  # ByteTrack config (ultralytics provides this)

# -------------------- Helper --------------------
def is_target_label(label):
    s = str(label).lower()
    return any(t in s for t in TARGET_LABELS)

# -------------------- Main --------------------
def main():
    print("Loading models...")
    suspicious = YOLO(YOLO_SUSPICIOUS)   # model used with .track()
    weapon     = YOLO(YOLO_WEAPON)       # model used with .predict() per-frame

    # Quick label-print (optional)
    print("Suspicious model labels:", suspicious.names)
    print("Weapon model labels    :", weapon.names)

    print("Starting ByteTrack on suspicious model...")
    # stream=True returns a generator that yields Result objects for each frame
    tracked_results = suspicious.track(
        source=INPUT_VIDEO,
        tracker=TRACKER_CFG,
        conf=CONF_SUSPICIOUS,
        stream=True
    )

    # Prepare video writer using properties of the input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video: " + INPUT_VIDEO)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w,h))
    cap.release()

    detections = []
    frame_id = 0
    print("Processing frames... (this can take time)")

    for result in tracked_results:
        # result.orig_img is the frame as numpy array (BGR)
        frame = result.orig_img.copy()

        # 1) draw tracked boxes from suspicious model (these include stable .id)
        # result.boxes is a Boxes object; iterate to get box tensors
        for box in result.boxes:
            try:
                xy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1,y1,x2,y2 = xy
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label = result.names[cls]
                # tracking id (ByteTrack) - may be None
                tid = int(box.id[0]) if getattr(box, "id", None) is not None else None
            except Exception:
                continue

            # We will show ID only for target labels
            show_id = tid if is_target_label(label) else None

            color = (0,128,255) if show_id is not None else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            text = f"{label} {conf:.2f}"
            if show_id is not None:
                text += f" | ID:{show_id}"
            cv2.putText(frame, text, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "frame": frame_id,
                "label": label,
                "confidence": conf,
                "bbox": [int(x1),int(y1),int(x2),int(y2)],
                "track_id": show_id
            })

        # 2) run weapon model on the SAME frame to get weapon/object boxes
        #    (these boxes will not have track IDs)
        res_weapon = weapon(frame, conf=CONF_WEAPON)
        # res_weapon is a Results list; we process the first (and only) result
        for r in res_weapon:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf[0])
                cls  = int(b.cls[0])
                label = r.names[cls] if hasattr(r, "names") else str(cls)

                # Draw weapon/object boxes (no ID)
                color = (0,200,200)  # different color for weapon/object if you like
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append({
                    "frame": frame_id,
                    "label": label,
                    "confidence": conf,
                    "bbox": [int(x1),int(y1),int(x2),int(y2)],
                    "track_id": None
                })

        out.write(frame)
        frame_id += 1

    out.release()
    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(detections, f, indent=2)

    print("Done. Output:", OUTPUT_VIDEO, "JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()

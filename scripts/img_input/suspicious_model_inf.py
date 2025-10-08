# suspicious_model_inf.py
# Part 1: load YOLOv8 best.pt, run on single image, compute 0-9 threat score, save annotated image + JSON.

import os
import json
import cv2
from ultralytics import YOLO

# ----------------- EDITABLE CONFIG (kept defaults to your sih folder paths) -----------------
BASE_DIR = r"/"
MODEL_PATH = os.path.join(BASE_DIR, "models", "suspicious_model4", "weights", "best.pt")
IMAGE_PATH = os.path.join(BASE_DIR, "dataset","suspicious-model-1", "test", "images", "OIP.jpg")
OUT_ANNOTATED = os.path.join(BASE_DIR, "results", "suspicious_model", "images", "res14.jpg")
OUT_JSON = os.path.join(BASE_DIR, "results","suspicious_model", "json", "res14.json")

# Scoring params (tune if needed)
CONF_THRESH = 0.35
AREA_WEIGHT = True
MAX_OBJECTS_EXPECTED = 3               # fallback cap if needed
MANUAL_WEIGHTS = {
    # Example suggestion for your model's classes (exact names must match model.names)
    # "weapon": 6.0,
    # "suspicious-suspect": 3.5,
    # "victim": 4.0,
    # "normal-action": 0.8
}
WEAPON_KEYWORDS = ("gun", "weapon", "rifle", "knife", "pistol", "ak", "handgun")
DEBUG = True   # set False to quiet debug prints
# ---------------------------------------------------------------------------------------------

def ensure_dirs():
    os.makedirs(os.path.join(BASE_DIR, "results", "suspicious_model", "images"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results", "suspicious_model", "json"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "test_images"), exist_ok=True)

def build_class_weights(model):
    class_weights = {}
    for idx, name in model.names.items():
        lname = str(name).lower()
        if name in MANUAL_WEIGHTS:
            class_weights[name] = MANUAL_WEIGHTS[name]
        elif any(k in lname for k in WEAPON_KEYWORDS):
            class_weights[name] = 6.0
        elif "suspicious" in lname or "suspect" in lname:
            class_weights[name] = 3.5
        elif "victim" in lname:
            class_weights[name] = 4.0
        elif "normal" in lname or "action" in lname:
            class_weights[name] = 0.8
        else:
            class_weights[name] = 1.0
    return class_weights

def box_area_fraction(xyxy, frame_w, frame_h):
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    fa = (bw * bh) / (frame_w * frame_h) if frame_w * frame_h > 0 else 0.0
    return max(0.001, fa)  # avoid exact zero; you can increase this floor if you want (e.g. 0.01)

def compute_raw_score(detections, frame_w, frame_h, class_weights):
    raw = 0.0
    contributions = []

    for cname, conf, xyxy in detections:
        if conf < CONF_THRESH:
            contributions.append((cname, conf, 0.0))
            continue

        w = class_weights.get(cname, 1.0)

        # special rule: don't scale weapons by area (we want even small weapons to count)
        if cname.lower() == "weapon":
            area_factor = 1.0
        else:
            area_factor = box_area_fraction(xyxy, frame_w, frame_h) if AREA_WEIGHT else 1.0

        contrib = w * conf * area_factor
        contributions.append((cname, conf, contrib))
        raw += contrib

    return raw, contributions

# ---------- CONFIG / TUNABLES for the new scoring ----------
WEAPON_CONF_THRESH = 0.25        # weapon presence threshold for override
WEAPON_SCORE_BASE = 7.0          # base for weapon mapping (maps to high 7..9)
WEAPON_SCORE_SCALE = 2.0         # scale applied to max weapon confidence

SUS_VICT_CONF_THRESH = 0.30     # threshold to consider suspect+victim combo
SUS_VICT_SCORE_BASE = 5.0
SUS_VICT_SCORE_SCALE = 4.0      # maps avg_conf -> added score (0..4) => 5..9

NORMAL_ONLY_MAX_SCORE = 3       # maximum score when only 'normal-action' appears

# Per-class expected maximum area (denominator). Use lowercase keys.
PER_CLASS_MAX_AREA = {
    "weapon": 1.0,
    "suspicious-suspect": 0.45,
    "victim": 0.45,
    "normal-action": 0.9
}
DEFAULT_CLASS_MAX_AREA = 0.6
# ------------------------------------------------------------

def _class_max_area_for(cname):
    return PER_CLASS_MAX_AREA.get(str(cname).lower(), DEFAULT_CLASS_MAX_AREA)

def _compute_base_cap(detections, class_weights):
    """cap = sum(weight * per_class_max_area) for detections above CONF_THRESH"""
    cap = 0.0
    details = []
    for cname, conf, xyxy in detections:
        if conf < CONF_THRESH:
            continue
        w = class_weights.get(cname, 1.0)
        max_area = _class_max_area_for(cname)
        max_contrib = w * max_area
        cap += max_contrib
        details.append((cname, w, max_area, max_contrib))
    return cap, details

def compute_base_score(raw, detections, class_weights):
    """
    Base score using per-class max-area cap (not current area).
    This prevents raw/cap degenerating into avg_conf.
    """
    cap, details = _compute_base_cap(detections, class_weights)
    if cap <= 0:
        # fallback conservative cap
        max_weight = max(class_weights.values()) if class_weights else 1.0
        cap = max_weight * MAX_OBJECTS_EXPECTED
    ratio = raw / cap
    ratio = max(0.0, min(1.0, ratio))
    base_score = int(round(ratio * 9))
    return base_score, cap, details

def compute_rule_based_score(detections, class_weights, frame_w, frame_h, raw):
    # collect best/conf by class (lowercased keys)
    best_conf = {}
    for cname, conf, xyxy in detections:
        key = str(cname).lower()
        best_conf[key] = max(conf, best_conf.get(key, 0.0))

    # 1) weapon override (categorical high threat)
    max_weapon_conf = 0.0
    for cname, conf, _ in detections:
        if "weapon" in str(cname).lower():
            max_weapon_conf = max(max_weapon_conf, conf)
    weapon_score = 0
    if max_weapon_conf >= WEAPON_CONF_THRESH:
        # map weapon confidence to 7..9 (tunable)
        weapon_score = min(9, int(round(WEAPON_SCORE_BASE + WEAPON_SCORE_SCALE * max_weapon_conf)))

    # 2) suspect+victim override (both present => strong suspicion)
    sus_conf = best_conf.get("suspicious-suspect", 0.0)
    vic_conf = best_conf.get("victim", 0.0)
    sus_vic_score = 0
    if sus_conf >= SUS_VICT_CONF_THRESH and vic_conf >= SUS_VICT_CONF_THRESH:
        avg = (sus_conf + vic_conf) / 2.0
        sus_vic_score = min(9, int(round(SUS_VICT_SCORE_BASE + SUS_VICT_SCORE_SCALE * avg)))

    # 3) base score (uses per-class max areas)
    base_score, cap, cap_details = compute_base_score(raw, detections, class_weights)

    # 4) If only normal-action detected, force a conservative lower cap
    classes_present = set(
        [str(cname).lower() for cname, conf, xyxy in detections if conf >= CONF_THRESH]
    )

    normal_only_score = 0
    if len(classes_present) > 0 and classes_present.issubset({"normal-action"}):
        normal_conf = [
            conf for cname, conf, xyxy in detections
            if str(cname).lower() == "normal-action" and conf >= CONF_THRESH
        ]
        if normal_conf:
            avg_conf = sum(normal_conf) / len(normal_conf)
            normal_only_score = min(NORMAL_ONLY_MAX_SCORE, int(round(NORMAL_ONLY_MAX_SCORE * avg_conf)))
        else:
            normal_only_score = 0

    final = max(base_score, weapon_score, sus_vic_score, normal_only_score)

    # debug details (if DEBUG True elsewhere)
    if 'DEBUG' in globals() and DEBUG:
        print("Rule-based scoring debug:")
        print("  base_score:", base_score, "cap:", cap)
        print("  cap_details:", cap_details)
        print("  weapon_score:", weapon_score, "max_weapon_conf:", max_weapon_conf)
        print("  sus_vic_score:", sus_vic_score, "sus_conf:", sus_conf, "vic_conf:", vic_conf)
        print("  normal_only_score:", normal_only_score)
        print("  final_score:", final)

    return final

# Usage in your flow (replace old normalize call):
# raw, contributions = compute_raw_score(detections, frame_w, frame_h, class_weights)
# score_0_9 = compute_rule_based_score(detections, class_weights, frame_w, frame_h, raw)


def main():
    ensure_dirs()

    if not os.path.exists(MODEL_PATH):
        print("ERROR: model file not found at:")
        print("  " + MODEL_PATH)
        return

    if not os.path.exists(IMAGE_PATH):
        print("ERROR: image file not found at:")
        print("  " + IMAGE_PATH)
        return

    print("Loading model from:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Model loaded. Classes:")
    for idx, name in model.names.items():
        print(f"  {idx} -> {name}")

    class_weights = build_class_weights(model)
    print("\nClass weights (auto):")
    for k, v in class_weights.items():
        print(f"  {k}: {v}")

    print("\nRunning inference on:", IMAGE_PATH)
    results = model.predict(source=IMAGE_PATH, conf=CONF_THRESH, imgsz=640, save=False)
    if len(results) == 0:
        print("No results returned by model.predict().")
        return

    res = results[0]
    try:
        frame_h, frame_w = int(res.orig_shape[0]), int(res.orig_shape[1])
    except Exception:
        im = cv2.imread(IMAGE_PATH)
        frame_h, frame_w = im.shape[:2]

    detections = []
    for box in res.boxes:
        try:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
        except Exception:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(x) for x in box.xyxy[0].cpu().numpy().tolist()]
        cname = model.names.get(cls_id, str(cls_id))
        detections.append((cname, conf, tuple(xyxy)))

    # raw, contributions = compute_raw_score(detections, frame_w, frame_h, class_weights)
    # score_0_9 = normalize_raw_to_0_9(raw, detections, class_weights, frame_w, frame_h)

    raw, contributions = compute_raw_score(detections, frame_w, frame_h, class_weights)
    score_0_9 = compute_rule_based_score(detections, class_weights, frame_w, frame_h, raw)

    print("\nDetections (class, conf, xyxy):")
    for d in detections:
        print(" ", d)
    print("\nContributions (class, conf, contrib):")
    for c in contributions:
        print(" ", c)
    print(f"\nRaw score: {raw:.6f}  => Final mapped score (0-9): {score_0_9}")

    try:
        annotated_rgb = res.plot()
        annotated_bgr = annotated_rgb[:, :, ::-1]
        cv2.imwrite(OUT_ANNOTATED, annotated_bgr)
        print("Annotated image saved to:", OUT_ANNOTATED)
    except Exception as e:
        print("res.plot() failed, using fallback drawing:", e)
        img = cv2.imread(IMAGE_PATH)
        img_annot = draw_fallback_annotated(img, detections)
        cv2.putText(img_annot, f"Threat: {score_0_9}/9", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(OUT_ANNOTATED, img_annot)
        print("Annotated image (fallback) saved to:", OUT_ANNOTATED)

    out_data = {
        "image_path": IMAGE_PATH,
        "model_path": MODEL_PATH,
        "raw_score": raw,
        "score_0_9": score_0_9,
        "detections": [
            {"class": d[0], "conf": float(d[1]), "xyxy": [float(x) for x in d[2]]}
            for d in detections
        ],
        "contributions": [
            {"class": c[0], "conf": float(c[1]), "contrib": float(c[2])}
            for c in contributions
        ]
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=2)
    print("JSON summary saved to:", OUT_JSON)

if __name__ == "__main__":
    main()

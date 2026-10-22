"""Capture our own training data from the webcam: images + HandLandmarker landmarks.

    python capture\\capture_custom.py --signer ana            # from ml/, webcam 0
    python capture\\capture_custom.py --signer ana --cam 1 --burst 15

Keys: 1-5 = capture that number (short countdown, then a burst of frames with a hand in view).
      Space = cancel a capture, Esc = quit.

Output (outside the repo, raw images never go to git), one folder per person, <name>_p<id>:
    DATASETS/custom_data/<name>_p<id>/Numbers/<n>/<name>_<n>_<time>_<k>.jpg   raw frame, no drawings
    DATASETS/custom_data/<name>_p<id>/Numbers/<n>/<name>_<n>_<time>_<k>.json  landmarks (see save())
    DATASETS/custom_data/<name>_p<id>/index.csv          one row per capture, paths relative to it
The id is reused if the name already has a folder, otherwise it is the next free one (p1, p2, ...).
Letters/ is ready for later (add keys to LABELS).

Landmarks are saved twice, like the training pipeline: on the full frame and on a hand crop
(120 px crop padded to 240, the LSC70ANH look). Train from the crop landmarks; fall back to full.
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(ML))  # project folder (holds DATASETS/)
BASE = os.path.join(ROOT, "DATASETS", "custom_data")
OUT = BASE  # set to the person's folder in main()
MODEL_PATH = os.path.join(ML, "models", "hand_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PAD, CROP, MARGIN = 60, 120, 1.25
# key -> (set folder, label). Add letters here later, e.g. "a": ("Letters", "A").
LABELS = {str(n): ("Numbers", str(n)) for n in range(1, 6)}


def landmarker(running_mode):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL_PATH), running_mode=running_mode,
        num_hands=1, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3))


def square_crop(rgb, lm):
    """Square crop around the landmarks, resized to 120 and padded to 240 with gray."""
    h, w = rgb.shape[:2]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    s = int(round(max(xs.max() - xs.min(), ys.max() - ys.min(), 20) * MARGIN))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = np.full((s, s, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + s, w), min(y0 + s, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    crop = cv2.resize(canvas, (CROP, CROP))
    return cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def hand_dict(res, w, h):
    if not res.hand_landmarks:
        return None
    hd = res.handedness[0][0]
    return {"img_w": w, "img_h": h, "handedness": hd.category_name, "handedness_score": round(hd.score, 4),
            "landmarks": [[q.x, q.y, q.z] for q in res.hand_landmarks[0]],
            "world_landmarks": [[q.x, q.y, q.z] for q in res.hand_world_landmarks[0]]}


def draw(img, lm, color, thick):
    h, w = img.shape[:2]
    for a, b in CONN:
        cv2.line(img, (int(lm[a][0] * w), int(lm[a][1] * h)), (int(lm[b][0] * w), int(lm[b][1] * h)), color, thick)
    for p in lm:
        cv2.circle(img, (int(p[0] * w), int(p[1] * h)), 3, (0, 0, 255), -1)


def person_folder(name):
    """<name>_p<id>: reuse the name's folder if it exists, else take the next free id."""
    ids = {}
    for d in os.listdir(BASE) if os.path.isdir(BASE) else []:
        m = re.fullmatch(r"(.+)_p(\d+)", d)
        if m and os.path.isdir(os.path.join(BASE, d)):
            ids[m.group(1)] = int(m.group(2))
    pid = ids.get(name, max(ids.values(), default=0) + 1)
    return f"{name}_p{pid}"


def count(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")]) if os.path.isdir(folder) else 0


def save(bgr, full, crop, signer, folder, label, k):
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    name = f"{signer}_{label}_{stamp}_{k:02d}"
    d = os.path.join(OUT, folder, label)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, name + ".jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = {"label": label, "set": folder.lower(), "signer": f"team:{signer}",
            "person": os.path.basename(OUT), "source": "custom",
            "time": stamp, "image": name + ".jpg", "full": full, "crop": crop}
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f)
    index = os.path.join(OUT, "index.csv")
    new = not os.path.exists(index)
    with open(index, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["path", "set", "label", "signer", "time", "handedness", "crop_ok"])
        wr.writerow([f"{folder}/{label}/{name}.jpg", folder.lower(), label, f"team:{signer}", stamp,
                     full["handedness"], int(crop is not None)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", help="who is signing (used for the signer split)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--burst", type=int, default=10, help="frames saved per key press")
    ap.add_argument("--gap", type=float, default=0.15, help="seconds between saved frames")
    ap.add_argument("--countdown", type=float, default=2.0, help="seconds before the burst starts")
    a = ap.parse_args()
    signer = (a.signer or input("Signer name: ")).strip().lower().replace(" ", "_") or "anon"
    global OUT
    OUT = os.path.join(BASE, person_folder(signer))

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(URL, MODEL_PATH)
    for folder, label in LABELS.values():
        os.makedirs(os.path.join(OUT, folder, label), exist_ok=True)
    print("saving to", OUT)

    find, refine = landmarker(vision.RunningMode.VIDEO), landmarker(vision.RunningMode.IMAGE)
    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_t, job, flash = -1, None, (0.0, "")  # job = {folder, label, start, saved, next}
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera read failed")
            break
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = max(int(time.monotonic() * 1000), last_t + 1)
        last_t = t
        res = find.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        full = hand_dict(res, w, h)
        crop, crop_view = None, np.full((240, 240, 3), 114, np.uint8)
        if full:
            c = square_crop(rgb, np.array(full["landmarks"]))
            crop = hand_dict(refine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(c))), 240, 240)
            crop_view = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
            if crop:
                draw(crop_view, crop["landmarks"], (0, 255, 0), 1)

        # capture job: countdown, then save one frame every `gap` s while a hand is in view
        now, status = time.monotonic(), ""
        if job:
            left = job["start"] - now
            if left > 0:
                status = f"{job['label']}: get ready {left:.1f}"
            elif not full:
                status = f"{job['label']}: no hand, waiting ({job['saved']}/{a.burst})"
            elif now >= job["next"]:
                save(bgr, full, crop, signer, job["folder"], job["label"], job["saved"])
                job["saved"] += 1
                job["next"] = now + a.gap
                if job["saved"] >= a.burst:
                    flash = (now + 1.5, f"saved {a.burst} x {job['label']}")
                    job = None
            if job and left <= 0:
                status = status or f"{job['label']}: capturing {job['saved']}/{a.burst}"

        view = bgr.copy()
        if full:
            draw(view, full["landmarks"], (0, 255, 0), 2)
        view = cv2.flip(view, 1)  # mirror for display only; saved images are not mirrored
        cv2.putText(view, f"signer: {os.path.basename(OUT)} | hand: {full['handedness'] if full else '-'}"
                    f" | crop: {'ok' if crop else '-'}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if job:
            cv2.putText(view, status, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)
        elif now < flash[0]:
            cv2.putText(view, flash[1], (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 3)
        counts = "  ".join(f"{lb}:{count(os.path.join(OUT, fd, lb))}" for fd, lb in LABELS.values())
        cv2.putText(view, f"saved  {counts}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, "1-5 = capture that number | Space = cancel | Esc = quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        view[10:250, w - 250:w - 10] = crop_view
        cv2.imshow("Soman capture", view)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord(" "):
            job = None
        elif k < 128 and chr(k).lower() in LABELS and not job:
            folder, label = LABELS[chr(k).lower()]
            job = {"folder": folder, "label": label, "start": now + a.countdown, "saved": 0, "next": 0.0}
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

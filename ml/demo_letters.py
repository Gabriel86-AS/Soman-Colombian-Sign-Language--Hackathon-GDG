"""Live webcam demo of the letter model (A B D I K M Q R U V W X).

    python demo_letters.py [camera_index]      # q or Esc to quit

Same pipeline as training: HandLandmarker finds the hand, the frame is cropped around it (like the
LSC70ANH crops, padded 120 -> 240 with gray), HandLandmarker runs again on the crop, then
normalize.handshape -> out/letters_mlp.json. Needs mediapipe + opencv (Python 3.9-3.12).
"""
import json
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

import normalize as N

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PAD, CROP, MARGIN = 60, 120, 1.25  # crop side = MARGIN * hand box; LSC70ANH hands fill their 120 px crop


def forward(model, f):
    x = (np.asarray(f) - model["mean"]) / model["std"]
    for L in model["layers"]:
        x = np.asarray(L["W"]) @ x + np.asarray(L["b"])
        if L["act"] == "relu":
            x = np.maximum(x, 0)
        else:
            e = np.exp(x - x.max())
            x = e / e.sum()
    return x


def landmarker(running_mode, conf):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL_PATH), running_mode=running_mode,
        num_hands=1, min_hand_detection_confidence=conf, min_tracking_confidence=0.5))


def square_crop(rgb, lm):
    """Crop a square around the landmarks, resize to 120, pad to 240 with gray (training-set look)."""
    h, w = rgb.shape[:2]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min(), 20) * MARGIN
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    canvas = np.full((s, s, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + s, w), min(y0 + s, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    crop = cv2.resize(canvas, (CROP, CROP))
    return cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def main():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(URL, MODEL_PATH)
    model = json.load(open(os.path.join(HERE, "out", "letters_mlp.json")))
    labels, tau, hold = model["labels"], model["threshold"], model["holdFrames"]
    find = landmarker(vision.RunningMode.VIDEO, 0.5)
    refine = landmarker(vision.RunningMode.IMAGE, 0.3)
    cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    t0, last_t, streak, cand, shown = 0, -1, 0, None, ""
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = max(int(cv2.getTickCount() / cv2.getTickFrequency() * 1000), last_t + 1)
        last_t = t
        res = find.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        view, crop_view, text = bgr.copy(), np.full((240, 240, 3), 114, np.uint8), "no hand"
        best, p = None, 0.0
        if res.hand_landmarks:
            lm = np.array([[q.x, q.y] for q in res.hand_landmarks[0]])
            hh, ww = view.shape[:2]
            for a, b in CONN:
                cv2.line(view, (int(lm[a, 0] * ww), int(lm[a, 1] * hh)), (int(lm[b, 0] * ww), int(lm[b, 1] * hh)), (0, 255, 0), 2)
            crop = square_crop(rgb, lm)
            crop_view = crop.copy()
            r2 = refine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if r2.hand_landmarks:
                lm2 = np.array([[q.x, q.y, q.z] for q in r2.hand_landmarks[0]])
                left = r2.handedness[0][0].category_name == "Left"
                for a, b in CONN:
                    cv2.line(crop_view, (int(lm2[a, 0] * 240), int(lm2[a, 1] * 240)), (int(lm2[b, 0] * 240), int(lm2[b, 1] * 240)), (0, 255, 0), 1)
                probs = forward(model, N.handshape(lm2, 240, 240, left))
                i = int(probs.argmax())
                best, p = labels[i], float(probs[i])
                text = f"{best if best != 'none' else '-'}  {p:.2f}"
        crop_bgr = cv2.cvtColor(crop_view, cv2.COLOR_RGB2BGR) if res.hand_landmarks else crop_view
        # a letter counts after `hold` frames above the threshold
        if best and best != "none" and p >= tau:
            streak = streak + 1 if best == cand else 1
            cand = best
            if streak >= hold:
                shown = best
        else:
            streak, cand = 0, None
            if best is None or best == "none":
                shown = shown if best is None else ""
        cv2.putText(view, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(view, f"letter: {shown}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 0), 3)
        cv2.putText(view, "hold your hand close; q = quit", (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        view[10:250, view.shape[1] - 250:view.shape[1] - 10] = crop_bgr
        cv2.imshow("Soman letters demo", view)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Live webcam demo of the letter model (out/letters_mlp.json, format v2).

    python demo_letters.py [camera_index]

Keys: a letter key = target mode for that letter (like the app: "is this the letter I asked for?"),
Space = free mode (guess any letter), Esc = quit.

Same pipeline as training: HandLandmarker finds the hand, the frame is cropped around it (like the
LSC70ANH crops, padded 120 -> 240 with gray), HandLandmarker runs again on the crop, then
normalize.hand_points -> letter_features -> MLP. If the crop pass loses the hand, the full-frame
landmarks are used. Needs mediapipe + opencv (Python 3.9-3.12).
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
ALPHA = 0.4  # probability smoothing across frames (one bad frame does not break the hold)


def forward(model, f):
    """Mean of the members' softmax outputs (one member unless the model is an ensemble)."""
    ps = []
    for mb in model["members"]:
        x = (np.asarray(f) - mb["mean"]) / mb["std"]
        for L in mb["layers"]:
            x = np.asarray(L["W"]) @ x + np.asarray(L["b"])
            if L["act"] == "relu":
                x = np.maximum(x, 0)
            else:
                e = np.exp(x - x.max())
                x = e / e.sum()
        ps.append(x)
    return np.mean(ps, 0)


def landmarker(running_mode, conf):
    # low thresholds: webcams are darker/blurrier than the dataset, keep tracking through hard frames
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL_PATH), running_mode=running_mode,
        num_hands=1, min_hand_detection_confidence=conf, min_hand_presence_confidence=conf,
        min_tracking_confidence=conf))


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
    if model.get("version") != 2:
        sys.exit("out/letters_mlp.json is not format v2: run train_letters.py and copy its output")
    labels, tau, per, hold, spec = model["labels"], model["threshold"], model["thresholds"], model["holdFrames"], model["features"]
    print("letters:", " ".join(labels[1:]))
    find = landmarker(vision.RunningMode.VIDEO, 0.3)
    refine = landmarker(vision.RunningMode.IMAGE, 0.3)
    cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # more pixels on the hand -> steadier landmarks
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_t, streak, cand, shown, avg, target = -1, 0, None, "", None, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = max(int(cv2.getTickCount() / cv2.getTickFrequency() * 1000), last_t + 1)
        last_t = t
        res = find.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        view, crop_view, text = bgr.copy(), np.full((240, 240, 3), 114, np.uint8), "no hand"
        hh, ww = view.shape[:2]
        if res.hand_landmarks:
            lm = np.array([[q.x, q.y, q.z] for q in res.hand_landmarks[0]])
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
                pts, src = N.hand_points(lm2, 240, 240, left), "crop"
            else:
                # crop pass lost the hand: use the full-frame landmarks (features are scale-free)
                left = res.handedness[0][0].category_name == "Left"
                pts, src = N.hand_points(lm, ww, hh, left), "full"
            probs = forward(model, N.letter_features(pts, spec))
            avg = probs if avg is None else ALPHA * probs + (1 - ALPHA) * avg
        else:
            avg = None

        # decide: target mode = p(target) over its own threshold; free mode = argmax over tau
        if avg is None:
            hit = None
        elif target:
            pt = float(avg[labels.index(target)])
            i = int(avg.argmax())
            text = f"{target}: {pt:.2f} / {per[target]:.2f}   (top {labels[i]} {avg[i]:.2f}, {src})"
            hit = target if pt >= per[target] else None
        else:
            i = int(avg.argmax())
            text = f"{labels[i] if labels[i] != 'none' else '-'}  {avg[i]:.2f}  ({src})"
            hit = labels[i] if labels[i] != "none" and avg[i] >= tau else None
        if hit:
            streak = streak + 1 if hit == cand else 1
            cand = hit
            if streak >= hold:
                shown = hit
        else:
            streak, cand = 0, None
            if avg is not None:
                shown = ""

        crop_bgr = cv2.cvtColor(crop_view, cv2.COLOR_RGB2BGR) if res.hand_landmarks else crop_view
        mode = f"target {target}" if target else "free"
        cv2.putText(view, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(view, f"{'OK ' if target and shown else ''}letter: {shown}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 0), 3)
        cv2.putText(view, f"mode: {mode} | letter key = target, Space = free, Esc = quit", (10, hh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        view[10:250, ww - 250:ww - 10] = crop_bgr
        cv2.imshow("Soman letters demo", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord(" "):
            target, shown = None, ""
        elif chr(k).upper() in labels[1:]:
            target, shown = chr(k).upper(), ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

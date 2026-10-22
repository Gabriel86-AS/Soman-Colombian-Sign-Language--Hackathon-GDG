"""Live webcam demo of the numbers model retrained with our team captures (2026-10-22).

    python demo.py [camera_index]
    python demo.py --check      # no camera: check the model against its golden cases

Uses its own frozen copy of the model (model/numbers_mlp.json), so retraining ml/out does not change it.
12 numbers: 1-10, MIL, MILLON (2 and 3 learned only from our captures) + none.

Keys (target mode, like the app: "is this the number I asked for?"):
    1-9 = that number, 0 = 10, M = MIL, N = MILLON, Space = free mode (guess any), Esc = quit.

Pipeline = training: HandLandmarker on the frame -> crop around the hand (LSC70ANH look) ->
HandLandmarker on the crop -> normalize.hand_points -> letter_features -> MLP. Falls back to the
full-frame landmarks if the crop pass loses the hand.
"""
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)
import normalize as N  # noqa: E402

MODEL_JSON = os.path.join(HERE, "model", "numbers_mlp.json")
GOLDEN_JSON = os.path.join(HERE, "model", "golden_numbers.json")
TASK_PATH = os.path.join(ML, "models", "hand_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PAD, CROP, MARGIN = 60, 120, 1.25
ALPHA = 0.4  # probability smoothing across frames
KEYS = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
        "0": "10", "m": "MIL", "n": "MILLON"}
PANEL_W = 300


def forward(model, f):
    """Mean of the members' softmax outputs."""
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


def load_model():
    model = json.load(open(MODEL_JSON))
    if model.get("version") != 2:
        sys.exit("model/numbers_mlp.json is not format v2")
    return model


def check():
    model = load_model()
    gold = json.load(open(GOLDEN_JSON))
    err = 0.0
    for c in gold["cases"]:
        p = N.hand_points(np.array(c["landmarks"]), c["img_w"], c["img_h"], c["handedness"] == "Left")
        probs = forward(model, N.letter_features(p, gold["features"]))
        err = max(err, float(np.abs(probs - np.array(c["probs"])).max()))
    print(f"labels: {' '.join(model['labels'])}")
    print(f"{len(gold['cases'])} golden cases, max error {err:.2e} (tolerance {gold['tolerance']})")
    sys.exit(0 if err <= gold["tolerance"] else 1)


def landmarker(vision, mpt, running_mode, conf):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=TASK_PATH), running_mode=running_mode,
        num_hands=1, min_hand_detection_confidence=conf, min_hand_presence_confidence=conf,
        min_tracking_confidence=conf))


def square_crop(cv2, rgb, lm):
    """Square around the landmarks, resized to 120, padded to 240 with gray (training-set look)."""
    h, w = rgb.shape[:2]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min(), 20) * MARGIN
    x0, y0, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    canvas = np.full((s, s, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + s, w), min(y0 + s, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    crop = cv2.resize(canvas, (CROP, CROP))
    return cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def draw_panel(cv2, h, labels, avg, target, per, tau, crop_bgr):
    """Right-side panel: hand crop on top, one probability bar per class with its threshold tick."""
    panel = np.full((h, PANEL_W, 3), 32, np.uint8)
    panel[10:250, 30:270] = crop_bgr
    y, bar_h = 270, max(14, min(26, (h - 280) // len(labels) - 4))
    for i, lab in enumerate(labels):
        p = float(avg[i]) if avg is not None else 0.0
        thr = tau if lab == "none" else (per[lab] if target == lab else tau)
        on_target = lab == target
        color = (60, 200, 255) if on_target else ((0, 200, 0) if p >= thr and lab != "none" else (150, 150, 150))
        cv2.putText(panel, lab, (8, y + bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if on_target else (200, 200, 200), 1)
        x0, x1 = 80, PANEL_W - 12
        cv2.rectangle(panel, (x0, y), (x1, y + bar_h), (70, 70, 70), 1)
        cv2.rectangle(panel, (x0, y), (x0 + int((x1 - x0) * p), y + bar_h), color, -1)
        tx = x0 + int((x1 - x0) * thr)
        cv2.line(panel, (tx, y - 2), (tx, y + bar_h + 2), (0, 0, 255), 1)
        y += bar_h + 4
        if y > h - bar_h:
            break
    return panel


def main():
    if "--check" in sys.argv:
        check()
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not os.path.exists(TASK_PATH):
        os.makedirs(os.path.dirname(TASK_PATH), exist_ok=True)
        print("downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(URL, TASK_PATH)
    model = load_model()
    labels, tau, per, hold, spec = model["labels"], model["threshold"], model["thresholds"], model["holdFrames"], model["features"]
    print("numbers:", " ".join(l for l in labels if l != "none"))
    find = landmarker(vision, mpt, vision.RunningMode.VIDEO, 0.3)
    refine = landmarker(vision, mpt, vision.RunningMode.IMAGE, 0.3)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cap = cv2.VideoCapture(int(args[0]) if args else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_t, streak, cand, shown, avg, target, src = -1, 0, None, "", None, None, ""
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera gave no frame (try another index: python demo.py 1)")
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
            crop = square_crop(cv2, rgb, lm)
            crop_view = crop.copy()
            r2 = refine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if r2.hand_landmarks:
                lm2 = np.array([[q.x, q.y, q.z] for q in r2.hand_landmarks[0]])
                left = r2.handedness[0][0].category_name == "Left"
                for a, b in CONN:
                    cv2.line(crop_view, (int(lm2[a, 0] * 240), int(lm2[a, 1] * 240)), (int(lm2[b, 0] * 240), int(lm2[b, 1] * 240)), (0, 255, 0), 1)
                pts, src = N.hand_points(lm2, 240, 240, left), "crop"
            else:
                left = res.handedness[0][0].category_name == "Left"
                pts, src = N.hand_points(lm, ww, hh, left), "full"
            probs = forward(model, N.letter_features(pts, spec))
            avg = probs if avg is None else ALPHA * probs + (1 - ALPHA) * avg
            crop_view = cv2.cvtColor(crop_view, cv2.COLOR_RGB2BGR)
        else:
            avg = None

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

        mode = f"target {target}" if target else "free"
        cv2.putText(view, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        big = (f"Bien! {shown}" if target and shown else f"number: {shown}")
        cv2.putText(view, big, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 0), 3)
        if not shown and streak:
            cv2.rectangle(view, (10, 110), (10 + int(300 * streak / hold), 122), (0, 200, 0), -1)
        cv2.putText(view, f"mode: {mode} | 1-9, 0=10, M=MIL, N=MILLON | Space = free, Esc = quit", (10, hh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        frame = np.hstack([view, draw_panel(cv2, hh, labels, avg, target, per, tau, crop_view)])
        cv2.imshow("Soman numbers demo (team model)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord(" "):
            target, shown = None, ""
        elif chr(k).lower() in KEYS and KEYS[chr(k).lower()] in labels:
            target, shown = KEYS[chr(k).lower()], ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

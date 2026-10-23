"""Live webcam demo of the moving-letter model (G H J N Ñ S Z), like the app's "Haz la seña".

    python demo_moving\\demo.py [camera_index]     # from ml/
    python demo_moving\\demo.py --check            # no camera: golden test

Keys: G H J N S Z = try that letter, 1 = Ñ, Space = try the same letter again, Esc = quit.

Flow = training data: countdown, then a 1 s window. Every frame: HandLandmarker on the frame -> crop around the
hand (LSC70ANH look) -> HandLandmarker on the crop. Then the 6 frames closest to 0, 1/6 ... 5/6 s are kept
(LSC70 is 6 fps), sequences.moving_shape -> 480 features -> MLP. Pass = p(target) >= its threshold.
Uses ml/out/moving_mlp.json (always the latest export).
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)
import sequences as S  # noqa: E402

MODEL_JSON = os.path.join(ML, "out", "moving_mlp.json")
GOLDEN_JSON = os.path.join(ML, "out", "golden", "moving_letters.json")
TASK_PATH = os.path.join(ML, "models", "hand_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PAD, CROP, MARGIN = 60, 120, 1.25
KEYS = {"g": "G", "h": "H", "j": "J", "n": "N", "1": "NN", "s": "S", "z": "Z"}
COUNTDOWN = 3.0
PANEL_W = 300


def forward(model, f):
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


def nan_array(rows):
    return np.array([[np.nan if v is None else v for v in p] for p in rows], float).reshape(-1, 21, 3)


def check():
    model, gold = json.load(open(MODEL_JSON)), json.load(open(GOLDEN_JSON))
    ef = ep = 0.0
    for c in gold["cases"]:
        f = S.moving_shape(nan_array(c["landmarks"]), c["img_w"], c["img_h"], c["is_left"], c["t_ms"], model["T"]).reshape(-1)
        ef = max(ef, float(np.abs(f - np.array(c["features"])).max()))
        ep = max(ep, float(np.abs(forward(model, f) - np.array(c["probs"])).max()))
    print(f"labels: {' '.join(model['labels'])}; ship: {model.get('gate3')}")
    print(f"{len(gold['cases'])} golden cases, max error features {ef:.2e}, probs {ep:.2e} (tolerance {gold['tolerance']})")
    sys.exit(0 if max(ef, ep) <= gold["tolerance"] else 1)


def landmarker(vision, mpt, running_mode):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=TASK_PATH), running_mode=running_mode,
        num_hands=1, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))


def square_crop(cv2, rgb, lm):
    h, w = rgb.shape[:2]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min(), 20) * MARGIN
    x0, y0, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    canvas = np.full((s, s, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + s, w), min(y0 + s, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    return cv2.copyMakeBorder(cv2.resize(canvas, (CROP, CROP)), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def score(model, frames):
    """frames: list of (t_ms from window start, crop landmarks [21,3] or None, is_left).
    Keep the 6 frames nearest LSC70's times, then features + MLP. Returns (probs, n_with_hand, strip) or None."""
    if not frames:
        return None
    n = model.get("frames", 6)
    win = model.get("window_ms", 1000)
    ts = np.array([f[0] for f in frames])
    pick = [int(np.abs(ts - k * win / n).argmin()) for k in range(n)]
    lm = np.full((n, 21, 3), np.nan)
    lefts = []
    for k, i in enumerate(pick):
        if frames[i][1] is not None:
            lm[k] = frames[i][1]
            lefts.append(frames[i][2])
    if len(lefts) < S.MOVING_MIN_FRAMES:
        return None
    f = S.moving_shape(lm, 240, 240, np.mean(lefts) > 0.5, np.arange(n) * win / n, model["T"])
    return forward(model, f.reshape(-1)), len(lefts), pick


def panel(cv2, h, model, probs, target, strip):
    p = np.full((h, PANEL_W, 3), 32, np.uint8)
    if strip is not None:
        p[10:10 + strip.shape[0], 10:10 + strip.shape[1]] = strip
    labels, thr = model["labels"], model["thresholds"]
    y = 160
    for i, lab in enumerate(labels):
        v = float(probs[i]) if probs is not None else 0.0
        name = model["display"].get(lab, lab)
        on = lab == target
        t = thr.get(lab, 0.5)
        col = (60, 200, 255) if on else (150, 150, 150)
        cv2.putText(p, name, (8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if on else (200, 200, 200), 1)
        x0, x1 = 80, PANEL_W - 12
        cv2.rectangle(p, (x0, y), (x1, y + 22), (70, 70, 70), 1)
        cv2.rectangle(p, (x0, y), (x0 + int((x1 - x0) * v), y + 22), col, -1)
        if lab in thr:
            tx = x0 + int((x1 - x0) * t)
            cv2.line(p, (tx, y - 2), (tx, y + 24), (0, 0, 255), 1)
        y += 30
    return p


def main():
    if "--check" in sys.argv:
        check()
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not os.path.exists(TASK_PATH):
        os.makedirs(os.path.dirname(TASK_PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, TASK_PATH)
    model = json.load(open(MODEL_JSON))
    disp = lambda l: model["display"].get(l, l)
    print("letters:", " ".join(disp(l) for l in model["labels"] if l != "static"), "| ship:", model.get("gate3"))
    find, refine = landmarker(vision, mpt, vision.RunningMode.VIDEO), landmarker(vision, mpt, vision.RunningMode.IMAGE)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cap = cv2.VideoCapture(int(args[0]) if args else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_t, target, phase, t0, frames, crops = -1, None, "idle", 0.0, [], []
    result, probs, strip = "Pick a letter: G H J N S Z, 1 = Ñ", None, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera gave no frame (try another index)")
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = max(int(time.monotonic() * 1000), last_t + 1)
        last_t = t
        res = find.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        view = bgr.copy()
        hh, ww = view.shape[:2]
        lm2, left, crop = None, False, None
        if res.hand_landmarks:
            lm = np.array([[q.x, q.y, q.z] for q in res.hand_landmarks[0]])
            for a, b in CONN:
                cv2.line(view, (int(lm[a, 0] * ww), int(lm[a, 1] * hh)), (int(lm[b, 0] * ww), int(lm[b, 1] * hh)), (0, 255, 0), 2)
            crop = square_crop(cv2, rgb, lm)
            r2 = refine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if r2.hand_landmarks:
                lm2 = np.array([[q.x, q.y, q.z] for q in r2.hand_landmarks[0]])
                left = r2.handedness[0][0].category_name == "Left"
        now = time.monotonic()
        if phase == "count" and now >= t0:
            phase, frames, crops = "rec", [], []
        if phase == "rec":
            el = (now - t0) * 1000
            frames.append((el, lm2, left))
            crops.append(crop)
            if el >= model.get("window_ms", 1000):
                phase = "idle"
                out = score(model, frames)
                if out is None:
                    result, probs, strip = "No hand seen enough. Try again (Space).", None, None
                else:
                    probs, nh, pick = out
                    ti = model["labels"].index(target)
                    ok_ = probs[ti] >= model["thresholds"][target]
                    top = model["labels"][int(probs.argmax())]
                    result = (f"Eso! {disp(target)} p={probs[ti]:.2f}" if ok_ else
                              f"Casi... {disp(target)} p={probs[ti]:.2f} < {model['thresholds'][target]:.2f} (top {disp(top)})")
                    tiles = [cv2.resize(cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR), (46, 46)) if crops[i] is not None
                             else np.zeros((46, 46, 3), np.uint8) for i in pick]
                    strip = np.hstack(tiles)
        view = cv2.flip(view, 1)  # mirror for display only
        if phase == "count":
            cv2.putText(view, f"{disp(target)} in {t0 - now:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4)
        elif phase == "rec":
            cv2.putText(view, f"GO! {disp(target)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        cv2.putText(view, result, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(view, "G H J N S Z, 1 = N~ | Space = again | Esc = quit (hand close to the camera)", (10, hh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Soman moving letters demo", np.hstack([view, panel(cv2, hh, model, probs, target, strip)]))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        ch = chr(k).lower() if k < 128 else ""
        if phase == "idle" and (ch in KEYS or (ch == " " and target)):
            if ch in KEYS:
                target = KEYS[ch]
            phase, t0, probs, strip, result = "count", now + COUNTDOWN, None, None, ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

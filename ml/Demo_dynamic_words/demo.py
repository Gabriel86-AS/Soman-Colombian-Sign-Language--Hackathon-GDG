"""Live webcam demo of the word model (8 MVP words), like the app's "Haz la seña" for words.

    python Demo_dynamic_words\\demo.py [camera_index]    # from ml/
    python Demo_dynamic_words\\demo.py --check           # no camera: golden test
    python Demo_dynamic_words\\demo.py --model words_mlp # use another export from ml/out/

Keys: 1-8 = try that word, Space = try the same word again, Esc = quit.

Flow = training pipeline: countdown, then a 2.5 s window. Every frame: HandLandmarker (2 hands) + PoseLandmarker lite
(nose, shoulders) on the full frame, rows like extract.py. Then sequences.clip_to_sequence (tracks, idle trim,
resample to 30), each block resampled to the model's T, flattened, MLP. hand_order "both": the clip is scored with
the hands as found and with the two hands swapped, and the probs are averaged. Pass = p(word) >= its threshold.
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

NAME = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "words_mlp_swap"
MODEL_JSON = os.path.join(ML, "out", f"{NAME}.json")
GOLDEN_JSON = os.path.join(ML, "out", "golden", f"{NAME}.json")
MODELS = os.path.join(ML, "models")
TASKS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
DISPLAY = {"GRACIAS": "Gracias", "BUENOS_DIAS": "Buenos dias", "BUENAS_TARDES": "Buenas tardes",
           "BUENAS_NOCHES": "Buenas noches", "BIEN": "Bien", "HOLA": "Hola", "ADIOS": "Adios",
           "POR_FAVOR": "Por favor", "other": "otra sena"}
COUNTDOWN = 3.0
PANEL_W = 340
POSE_COLS = ["nose_x", "nose_y", "lsh_x", "lsh_y", "rsh_x", "rsh_y"]


# ---------------------------------------------------------------- model (plain numpy, same as the app)
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


def resample_T(a, T):
    """[30, ...] -> [T, ...] linear on a uniform grid (same as train_words.resample_T)."""
    n = a.shape[0]
    if n == T:
        return a
    dst = np.linspace(0, 1, T)
    i0 = np.minimum(np.floor(dst * (n - 1)).astype(int), n - 2)
    w = (dst * (n - 1) - i0).reshape((T,) + (1,) * (a.ndim - 1))
    return (1 - w) * a[i0] + w * a[i0 + 1]


TIPS = [3, 7, 11, 15, 19]


def features(b, model):
    """blocks at T=30 (shape [30,40], locB, locN, shape2, locB2, locN2 [30,2]) -> flat features for the MLP."""
    T = model["T"]
    r = {k: resample_T(np.asarray(v, float), T) for k, v in b.items()}
    r["shape2_tips"] = r["shape2"].reshape(T, 20, 2)[:, TIPS].reshape(T, 10)
    m2 = float(np.abs(r["locB2"]).sum() > 0)
    rel = np.stack([-(r["locB"][:, 0] + r["locB2"][:, 0]), r["locB2"][:, 1] - r["locB"][:, 1]], -1)
    r["hrel"], r["hdist"] = rel * m2, np.linalg.norm(rel, axis=-1, keepdims=True) * m2
    for k in ("locB", "locN", "locB2"):
        r["d" + k] = np.vstack([np.zeros((1, 2)), np.diff(r[k], axis=0)])
    return np.concatenate([r[k] for k, _ in model["layout"]], 1).reshape(-1)


def swapped(b):
    if not has2(b):
        return b
    return dict(b, shape=b["shape2"], shape2=b["shape"], locB=b["locB2"], locB2=b["locB"], locN=b["locN2"], locN2=b["locN"])


def has2(b):
    return float(np.abs(np.asarray(b["shape2"])).sum()) > 0


def order(b, model):
    if model.get("hand_order") == "higher_main" and has2(b) and \
            np.mean(np.asarray(b["locB2"])[:, 1]) < np.mean(np.asarray(b["locB"])[:, 1]):
        return swapped(b)
    return b


def predict(model, b):
    b = order(b, model)
    views = [b, swapped(b)] if model.get("hand_order") == "both" else [b]
    return np.mean([forward(model, features(v, model)) for v in views], 0)


def check():
    model, gold = json.load(open(MODEL_JSON)), json.load(open(GOLDEN_JSON))
    ef = ep = 0.0
    for c in gold["cases"]:
        b = {k: np.asarray(v) for k, v in c["blocks"].items()}  # golden blocks are already in hand order
        ef = max(ef, float(np.abs(features(b, model) - np.array(c["features"])).max()))
        if c.get("features_swapped") is not None:
            ef = max(ef, float(np.abs(features(swapped(b), model) - np.array(c["features_swapped"])).max()))
        views = [b, swapped(b)] if model.get("hand_order") == "both" else [b]
        p = np.mean([forward(model, features(v, model)) for v in views], 0)
        ep = max(ep, float(np.abs(p - np.array(c["probs"])).max()))
    print(f"model {NAME}: hand order {model.get('hand_order')}, floor {model.get('floor')}, ship {model.get('gate5')}")
    print(f"{len(gold['cases'])} golden cases, max error features {ef:.2e}, probs {ep:.2e} (tolerance {gold['tolerance']})")
    sys.exit(0 if max(ef, ep) <= gold["tolerance"] else 1)


# ---------------------------------------------------------------- live
def score(model, rows):
    """rows: per-frame dicts like extract.py -> (probs, blocks) or None."""
    import pandas as pd
    if not rows:
        return None
    g = pd.DataFrame(rows)
    for c in S.HAND_COLS + POSE_COLS:
        if c not in g:
            g[c] = np.nan
    # extract.py puts the clip-level pose on every row; clip_to_sequence takes the median
    if g[POSE_COLS].isna().all().any():
        return "pose"
    r = S.clip_to_sequence(g, 30, model.get("idle_thr"))
    if r is None:
        return None
    b = dict(shape=r["X"][:, :40], locB=r["locB"], locN=r["locN"], shape2=r["X2"][:, :40],
             locB2=r["locB2"], locN2=r["locN2"])
    if any(np.isnan(v).any() for v in b.values()):
        return "pose"
    return predict(model, b), b


def panel(cv2, h, model, probs, target):
    p = np.full((h, PANEL_W, 3), 32, np.uint8)
    thr = model["thresholds"]
    y = 30
    for i, lab in enumerate(model["labels"]):
        v = float(probs[i]) if probs is not None else 0.0
        on = lab == target
        col = (60, 200, 255) if on else (150, 150, 150)
        key = f"{i + 1} " if lab in thr else "  "
        cv2.putText(p, key + DISPLAY.get(lab, lab), (8, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if on else (200, 200, 200), 1)
        x0, x1 = 150, PANEL_W - 12
        cv2.rectangle(p, (x0, y), (x1, y + 22), (70, 70, 70), 1)
        cv2.rectangle(p, (x0, y), (x0 + int((x1 - x0) * v), y + 22), col, -1)
        if lab in thr:
            tx = x0 + int((x1 - x0) * thr[lab])
            cv2.line(p, (tx, y - 2), (tx, y + 24), (0, 0, 255), 2)
        if lab in model.get("gate5", []):
            cv2.putText(p, "*", (x1 - 2, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 32
    cv2.putText(p, "red tick = pass line, * = passed Gate 5", (8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(p, f"hand order: {model.get('hand_order')}", (8, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return p


def main():
    if "--check" in sys.argv:
        check()
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    os.makedirs(MODELS, exist_ok=True)
    for f, url in TASKS.items():
        if not os.path.exists(os.path.join(MODELS, f)):
            print("downloading", f)
            urllib.request.urlretrieve(url, os.path.join(MODELS, f))
    model = json.load(open(MODEL_JSON))
    words = [l for l in model["labels"] if l in model["thresholds"]]
    keys = {str(i + 1): w for i, w in enumerate(words)}
    print(f"model {NAME} | hand order {model.get('hand_order')} | ship {model.get('gate5')}")
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and sys.argv[i] != "--model"]
    cap = cv2.VideoCapture(int(args[0]) if args else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_t, target, phase, t0, rows, fi = -1, None, "idle", 0.0, [], 0
    result, probs = "Pick a word: 1-8 (stand back: head, shoulders and hands in view)", None
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera gave no frame (try another index)")
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hh, ww = rgb.shape[:2]
        t = max(int(time.monotonic() * 1000), last_t + 1)
        last_t = t
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hr, pr = hands.detect_for_video(mpi, t), pose.detect_for_video(mpi, t)
        view = bgr.copy()
        pv = {c: np.nan for c in POSE_COLS}
        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            pv = dict(nose_x=lm[0].x, nose_y=lm[0].y, lsh_x=lm[11].x, lsh_y=lm[11].y, rsh_x=lm[12].x, rsh_y=lm[12].y)
            for a, b in (("nose", "lsh"), ("nose", "rsh"), ("lsh", "rsh")):
                cv2.line(view, (int(pv[a + "_x"] * ww), int(pv[a + "_y"] * hh)),
                         (int(pv[b + "_x"] * ww), int(pv[b + "_y"] * hh)), (255, 160, 0), 2)
        now = time.monotonic()
        if phase == "count" and now >= t0:
            phase, rows, fi, t_start = "rec", [], 0, t
        frame_rows = []
        base = dict(frame=fi, t_ms=t - (t_start if phase == "rec" else 0), img_w=ww, img_h=hh)
        if hr.hand_landmarks:
            for hi, lms in enumerate(hr.hand_landmarks):
                cat = hr.handedness[hi][0]
                r = {**base, "hand_idx": hi, "handedness": cat.category_name, "hand_score": cat.score, **pv}
                for i, q in enumerate(lms):
                    r[f"x{i}"], r[f"y{i}"], r[f"z{i}"] = q.x, q.y, q.z
                frame_rows.append(r)
                pts = [(int(q.x * ww), int(q.y * hh)) for q in lms]
                for a, b in CONN:
                    cv2.line(view, pts[a], pts[b], (0, 255, 0) if hi == 0 else (0, 200, 255), 2)
        else:
            frame_rows.append({**base, "hand_idx": -1, "handedness": None, "hand_score": np.nan, **pv})
        if phase == "rec":
            rows += frame_rows
            fi += 1
            if (now - t0) * 1000 >= model.get("window_ms", 2500):
                phase = "idle"
                out = score(model, rows)
                if out is None:
                    result, probs = "No hands seen enough. Try again (Space).", None
                elif out == "pose":
                    result, probs = "Body not seen (need head + shoulders). Step back, Space.", None
                else:
                    probs, b = out
                    ti = model["labels"].index(target)
                    thr = model["thresholds"][target]
                    top = model["labels"][int(probs.argmax())]
                    two = " (2 hands)" if has2(b) else ""
                    result = (f"Eso! {DISPLAY[target]} p={probs[ti]:.2f}{two}" if probs[ti] >= thr else
                              f"Casi... {DISPLAY[target]} p={probs[ti]:.2f} < {thr:.2f} (top {DISPLAY.get(top, top)}){two}")
        view = cv2.flip(view, 1)  # mirror for display only
        if phase == "count":
            cv2.putText(view, f"{DISPLAY[target]} in {t0 - now:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 4)
        elif phase == "rec":
            left = model.get("window_ms", 2500) / 1000 - (now - t0)
            cv2.putText(view, f"GO! {DISPLAY[target]} {left:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        cv2.putText(view, result, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(view, "1-8 = word | Space = again | Esc = quit", (10, hh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Soman dynamic words demo", np.hstack([view, panel(cv2, hh, model, probs, target)]))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        ch = chr(k) if k < 128 else ""
        if phase == "idle" and (ch in keys or (ch == " " and target)):
            if ch in keys:
                target = keys[ch]
            phase, t0, probs, result = "count", now + COUNTDOWN, None, ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

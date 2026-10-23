"""Always-on fingerspelling demo: static letters AND moving letters, recognized continuously (no countdown).

    python demo_live\\demo.py [camera_index]         # from ml/
    python demo_live\\demo.py --all-moving           # also try H and S (they failed Gate 3)

Every frame: HandLandmarker on the frame -> crop around the hand (LSC70ANH look) -> HandLandmarker on the crop.
  - Static letters (out/letters_mlp.json): that frame's hand_points -> MLP, smoothed. A letter fires when it is the
    top guess over its threshold for HOLD frames AND the hand is steady (so a moving letter in progress does not
    fire as a static one).
  - Moving letters (out/moving_mlp.json): sliding 1 s window ending now -> the 6 frames nearest LSC70's times ->
    sequences.moving_shape -> MLP. A letter fires when it is the top class over max(its threshold, MOVE_TAU) for
    MOVE_HOLD frames in a row. The model's "static" class rejects held hands.
After a letter fires: COOLDOWN with no new letters; the same static letter needs the hand to change before repeating.

Keys: Space = add a space, Backspace = delete last, Delete or 0 = clear text, Esc = quit.
"""
import json
import os
import sys
import time
import urllib.request
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path.insert(0, ML)
import normalize as N  # noqa: E402
import sequences as S  # noqa: E402

LETTERS_JSON = os.path.join(ML, "out", "letters_mlp.json")
MOVING_JSON = os.path.join(ML, "out", "moving_mlp.json")
TASK_PATH = os.path.join(ML, "models", "hand_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PAD, CROP, MARGIN = 60, 120, 1.25

# tuning (shown on screen)
ALPHA = 0.4          # static: probability smoothing across frames
STEADY_MS = 400      # static: the hand must be steady over this long ...
STEADY_MAX = 0.35    # ... = mean fingertip/joint displacement (palm units) from the current frame below this
MOVE_TAU = 0.80      # moving: minimum probability (free mode is stricter than the app's target mode)
MOVE_ALPHA = 0.5     # moving: smoothing across consecutive windows
MOVE_HOLD = 3        # moving: frames in a row over the threshold
COOLDOWN = 0.8       # seconds without new letters after one fires
REARM = 6            # frames of another (or no) letter before the same static letter can fire again
BUFFER_MS = 1500
PANEL_W = 320


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


def moving_window(buf, now, model):
    """Probabilities for the 1 s window ending now, or None (window not full / too few hands)."""
    win, n = model.get("window_ms", 1000), model.get("frames", 6)
    if not buf or buf[0][0] > now - win + 60:  # need about a full window of history
        return None
    ts = np.array([b[0] for b in buf])
    lm = np.full((n, 21, 3), np.nan)
    lefts = []
    for k in range(n):
        i = int(np.abs(ts - (now - win + k * win / n)).argmin())
        if buf[i][1] is not None:
            lm[k] = buf[i][1]
            lefts.append(buf[i][2])
    if len(lefts) < S.MOVING_MIN_FRAMES:
        return None
    f = S.moving_shape(lm, 240, 240, np.mean(lefts) > 0.5, np.arange(n) * win / n, model["T"])
    return forward(model, f.reshape(-1))


def steadiness(buf, now):
    """Max mean point displacement (palm units) between the latest hand shape and those in the last STEADY_MS."""
    shapes = [b[3] for b in buf if b[0] >= now - STEADY_MS and b[3] is not None]
    if len(shapes) < 3:
        return np.inf
    cur = shapes[-1]
    return max(float(np.sqrt(((s - cur) ** 2).sum(-1)).mean()) for s in shapes)


def bars(cv2, panel, y, title, labels, probs, disp, hi, n=5):
    cv2.putText(panel, title, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 8
    if probs is None:
        cv2.putText(panel, "-", (8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return y + 30 * n
    for i in np.argsort(-probs)[:n]:
        lab, p = labels[i], float(probs[i])
        col = (0, 200, 0) if lab == hi else (150, 150, 150)
        cv2.putText(panel, disp(lab), (8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.rectangle(panel, (70, y + 4), (PANEL_W - 12, y + 24), (70, 70, 70), 1)
        cv2.rectangle(panel, (70, y + 4), (70 + int((PANEL_W - 82) * p), y + 24), col, -1)
        y += 28
    return y + 8


def main():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not os.path.exists(TASK_PATH):
        os.makedirs(os.path.dirname(TASK_PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, TASK_PATH)
    SL = json.load(open(LETTERS_JSON))
    MV = json.load(open(MOVING_JSON))
    s_labels, s_tau, s_per, s_hold, s_spec = SL["labels"], SL["threshold"], SL["thresholds"], SL["holdFrames"], SL["features"]
    m_labels, m_thr = MV["labels"], MV["thresholds"]
    disp = lambda l: MV.get("display", {}).get(l, l)
    m_allowed = set(m_thr) if "--all-moving" in sys.argv else set(MV.get("gate3", m_thr))
    print("static:", " ".join(l for l in s_labels if l != "none"))
    print("moving:", " ".join(disp(l) for l in m_labels if l in m_allowed))

    find, refine = landmarker(vision, mpt, vision.RunningMode.VIDEO), landmarker(vision, mpt, vision.RunningMode.IMAGE)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cap = cv2.VideoCapture(int(args[0]) if args else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buf = deque()  # (t_ms, crop landmarks [21,3] or None, is_left, shape [20,2] or None)
    last_t, s_avg, m_avg = -1, None, None
    s_cand, s_streak, s_armed_block, rearm = None, 0, None, 0
    m_cand, m_streak = None, 0
    text, last_fire, flash, cool_until = "", "", (0.0, ""), 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera gave no frame (try another index)")
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = max(int(time.monotonic() * 1000), last_t + 1)
        last_t = t
        now_s = t / 1000
        res = find.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        view = bgr.copy()
        hh, ww = view.shape[:2]
        lm2, left, shape, s_probs = None, False, None, None
        crop_view = np.full((240, 240, 3), 114, np.uint8)
        if res.hand_landmarks:
            lm = np.array([[q.x, q.y, q.z] for q in res.hand_landmarks[0]])
            for a, b in CONN:
                cv2.line(view, (int(lm[a, 0] * ww), int(lm[a, 1] * hh)), (int(lm[b, 0] * ww), int(lm[b, 1] * hh)), (0, 255, 0), 2)
            crop = square_crop(cv2, rgb, lm)
            crop_view = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            r2 = refine.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if r2.hand_landmarks:
                lm2 = np.array([[q.x, q.y, q.z] for q in r2.hand_landmarks[0]])
                left = r2.handedness[0][0].category_name == "Left"
                pts = N.hand_points(lm2, 240, 240, left)
                shape = pts[1:, :2]
                s_probs = forward(SL, N.letter_features(pts, s_spec))
                for a, b in CONN:
                    cv2.line(crop_view, (int(lm2[a, 0] * 240), int(lm2[a, 1] * 240)), (int(lm2[b, 0] * 240), int(lm2[b, 1] * 240)), (0, 255, 0), 1)
        buf.append((t, lm2, left, shape))
        while buf and buf[0][0] < t - BUFFER_MS:
            buf.popleft()

        # ---- static letters: smoothed, held, hand steady
        s_avg = None if s_probs is None else (s_probs if s_avg is None else ALPHA * s_probs + (1 - ALPHA) * s_avg)
        steady = steadiness(buf, t)
        s_hit = None
        if s_avg is not None:
            i = int(s_avg.argmax())
            lab = s_labels[i]
            if lab != "none" and s_avg[i] >= max(s_tau, s_per.get(lab, 0)) and steady <= STEADY_MAX:
                s_hit = lab
        if s_hit and s_hit == s_cand:
            s_streak += 1
        else:
            s_cand, s_streak = s_hit, (1 if s_hit else 0)
        if s_armed_block is not None:  # re-arm the last static letter after REARM frames of something else
            rearm = rearm + 1 if s_hit != s_armed_block else 0
            if rearm >= REARM:
                s_armed_block, rearm = None, 0

        # ---- moving letters: sliding 1 s window
        m_probs = moving_window(buf, t, MV)
        m_avg = None if m_probs is None else (m_probs if m_avg is None else MOVE_ALPHA * m_probs + (1 - MOVE_ALPHA) * m_avg)
        m_hit = None
        if m_avg is not None:
            i = int(m_avg.argmax())
            lab = m_labels[i]
            if lab in m_allowed and m_avg[i] >= max(m_thr[lab], MOVE_TAU):
                m_hit = lab
        if m_hit and m_hit == m_cand:
            m_streak += 1
        else:
            m_cand, m_streak = m_hit, (1 if m_hit else 0)

        # ---- fire (moving first; static needs the hand to change before the same letter repeats)
        fired = None
        if now_s >= cool_until:
            if m_hit and m_streak >= MOVE_HOLD:
                fired = disp(m_hit)
                buf.clear()  # a new moving letter needs a fresh window
                m_avg, m_cand, m_streak = None, None, 0
            elif s_hit and s_streak >= s_hold and s_hit != s_armed_block:
                fired = s_hit
                s_armed_block, rearm = s_hit, 0
        if fired:
            text += fired
            last_fire = fired
            flash = (now_s + 1.0, fired)
            cool_until = now_s + COOLDOWN
            s_streak = 0

        # ---- draw
        view = cv2.flip(view, 1)  # mirror for display only
        if now_s < flash[0]:
            cv2.putText(view, flash[1], (ww - 200, 160), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 200, 0), 10)
        cv2.rectangle(view, (0, hh - 90), (ww, hh), (0, 0, 0), -1)
        cv2.putText(view, (text[-28:] or " ") + "_", (14, hh - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(view, "Space = space | Backspace = delete | 0 = clear | Esc = quit  (hand close to the camera)",
                    (14, hh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        status = "cooldown" if now_s < cool_until else ("steady" if steady <= STEADY_MAX else "moving")
        cv2.putText(view, f"{status} | motion {min(steady, 9.99):.2f} (static fires < {STEADY_MAX})", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if s_cand and s_streak < s_hold:
            cv2.rectangle(view, (14, 46), (14 + int(260 * s_streak / s_hold), 56), (0, 200, 0), -1)
        if m_cand:
            cv2.rectangle(view, (14, 62), (14 + int(260 * min(m_streak, MOVE_HOLD) / MOVE_HOLD), 72), (0, 165, 255), -1)

        panel = np.full((hh, PANEL_W, 3), 32, np.uint8)
        panel[10:250, 40:280] = crop_view
        y = bars(cv2, panel, 280, "static letters", s_labels, s_avg, disp, s_cand, 4)
        bars(cv2, panel, y + 6, "moving letters (last 1 s)", m_labels, m_avg, disp, m_cand, 4)
        cv2.imshow("Soman live fingerspelling", np.hstack([view, panel]))

        k = cv2.waitKeyEx(1)
        if k == 27:
            break
        if k == 32:
            text += " "
        elif k == 8:
            text = text[:-1]
        elif k in (ord("0"), 3014656):  # 0 or Delete
            text = ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

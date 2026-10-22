import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import normalize as N  # noqa: E402
import sequences as S  # noqa: E402

rng = np.random.default_rng(1)
W, H = 640, 480


def hand():
    lm = np.zeros((21, 3))
    lm[:, :2] = 0.4 + 0.1 * rng.random((21, 2))
    return lm


BASE = hand()


def test_mirror_right_equals_left():
    lm = hand()
    mir = lm.copy()
    mir[:, 0] = 1 - mir[:, 0]  # image flipped horizontally
    a = N.handshape(lm, W, H, False)
    b = N.handshape(mir, W, H, True)
    assert np.allclose(a, b, atol=1e-9)


def test_scale_translate_invariant():
    lm = hand()
    a = N.handshape(lm, W, H, False)
    lm2 = lm.copy()
    lm2[:, :2] = lm2[:, :2] * 0.5 + 0.2  # smaller and shifted
    b = N.handshape(lm2, W, H, False)
    assert np.allclose(a, b, atol=1e-9)


def _clip(fps, n_sec=2.0):
    """Same motion sampled at `fps`: wrist moves right then stops, handshape fixed."""
    base = BASE
    rows = []
    for i in range(int(n_sec * fps)):
        t = i / fps
        lm = base.copy()
        lm[:, 0] += 0.2 * min(t, 1.0)
        r = dict(dataset="x", sample_id="c", signer="s", label="L", rep=0, frame=i,
                 t_ms=round(t * 1000), img_w=W, img_h=H, hand_idx=0, handedness="Right",
                 hand_score=1.0, nose_x=np.nan, nose_y=np.nan, lsh_x=np.nan, lsh_y=np.nan,
                 rsh_x=np.nan, rsh_y=np.nan)
        for k in range(21):
            r[f"x{k}"], r[f"y{k}"], r[f"z{k}"] = lm[k]
        rows.append(r)
    return pd.DataFrame(rows)


def test_resample_fps_invariant():
    a = S.clip_to_sequence(_clip(24), 30)["X"]
    b = S.clip_to_sequence(_clip(30), 30)["X"]
    assert np.abs(a - b).max() < 0.15
    assert np.allclose(a[:, :40], b[:, :40], atol=1e-6)


def test_idle_trim():
    out = S.clip_to_sequence(_clip(30), 30, idle_thr=0.5)
    assert out is not None and out["X"].shape == (30, 42)

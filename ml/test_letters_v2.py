"""Smoke test for the v2 letter features and the train/export path, on fake data.

    python test_letters_v2.py
"""
import os
import tempfile

os.environ.setdefault("SOMAN_EXPORT", tempfile.mkdtemp())
import numpy as np

import normalize as N
import train_letters as T

SPECS = ("xy", "xy+z", "xy+geo", "xy+z+geo3")


def test_features():
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.3, 0.7, (21, 3))
    lm[:, 2] *= 0.1
    for left in (False, True):  # 'xy' must equal the v1 handshape exactly
        assert np.allclose(N.letter_features(N.hand_points(lm, 240, 200, left), "xy"), N.handshape(lm, 240, 200, left))
    lm2 = lm.copy()
    lm2[:, 0] = 1 - lm2[:, 0]
    for spec in SPECS:  # mirrored image + flipped handedness -> same features; batch == single
        a = N.letter_features(N.hand_points(lm, 240, 200, False), spec)
        b = N.letter_features(N.hand_points(lm2, 240, 200, True), spec)
        assert np.allclose(a, b), spec
        assert np.allclose(N.letter_features(N.hand_points(np.stack([lm, lm]), 240, 200, False), spec)[0], a)
    print("dims", {s: len(N.letter_features(N.hand_points(lm, 1, 1, False), s)) for s in SPECS})


def test_pipeline():
    rng = np.random.default_rng(0)
    cls = ["A", "B", "C", "none"]
    protos = {c: rng.uniform(0.3, 0.7, (21, 3)) for c in cls}
    lms, ys, gs, sids = [], [], [], []
    for s in range(10):
        for c in cls:
            for k in range(6):
                lms.append(protos[c] + rng.normal(0, 0.01, (21, 3)))
                ys.append(c); gs.append(f"s{s}"); sids.append(f"{s}{c}" if c != "none" else f"n{s}{k}")
    lms = np.array(lms)
    D = dict(P=N.hand_points(lms, 240, 240, False), y=np.array(ys), g=np.array(gs), sid=np.array(sids))
    classes = np.unique(D["y"])
    cfg = dict(T.BASE, feats="xy+z+geo3", aug3d=True, seeds=2)
    r = T.score(T.cv(cfg, D, None), D, classes)
    assert r["macro"] > 0.9, r["macro"]
    tau, per, rec = T.thresholds(r["probs"], D["y"], classes)
    members = [T.fit(cfg, D["P"], D["y"], s) for s in range(2)]
    model = T.to_json(members, cfg["feats"], tau, per)
    L = dict(lm=lms, w=np.full(len(lms), 240), h=np.full(len(lms), 240), hand=np.array(["Right"] * len(lms)), y=D["y"])
    gold, err = T.golden(model, L, members, cfg["feats"])
    assert err < 1e-9 and len(gold["cases"]) == 6, (err, len(gold["cases"]))
    print("pipeline ok: macro", round(r["macro"], 3), "tau", round(tau, 3), "golden err", err)


if __name__ == "__main__":
    test_features()
    test_pipeline()
    print("OK")

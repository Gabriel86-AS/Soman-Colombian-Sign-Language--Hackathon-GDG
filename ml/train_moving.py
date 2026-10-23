"""Moving letters G H J N Ñ S Z: small MLP on the hand-crop shape sequence (Phase 6; replaces DTW, user decision 2026-10-22).

    python train_moving.py           # variant search (signers held out) + export
    python train_moving.py --quick   # base config only + export

Input per clip: LSC70ANH crops, 6 frames (first second, 6 fps) -> sequences.moving_shape -> [12, 40] -> 480 dims.
Classes: the 7 moving letters + "static" (clips of the 20 static letters: a held K must not pass as G).
The app knows the target letter: pass = p(target) >= thresholds[target] (and >= 0.5, so it is the top guess).
Outputs: ml/out/moving_mlp.json, golden/moving_letters.json, moving_report.md. Run log: Drive soman-ml/moving_runs.json.
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import normalize as N
import sequences as S
import moving_dtw as M

EXPORT = M.EXPORT
RUNS = os.path.join(os.path.dirname(M.OUT.rstrip("/")), "moving_runs.json")
CLASSES = M.classes(True)  # 7 letters + "static"
T = M.T
BASE = dict(hidden=[256, 128], alpha=1e-3, copies=4, rot=15, aug3d=False, warp=False, seeds=3)
FEATURES = "moving_shape_m1"  # sequences.moving_shape on the hand crop, T=12, flattened [t][point 1..20][x,y]


# ---------------------------------------------------------------- data
def clip_points(raw):
    """Crop landmarks [6,21,3] -> [T,21,3]: pixels (z * img_w), left hands mirrored, wrist origin,
    divided by the clip's mean 2D palm size, resampled by time. x,y of points 1..20 == moving_shape()."""
    t = raw["t_ms"]
    lm = S._fill_nan(raw["crop"], t)
    w, h = raw["crop_wh"]
    p = np.stack([lm[..., 0] * w, lm[..., 1] * h, lm[..., 2] * w], -1)
    if raw["is_left"]:
        p[..., 0] = -p[..., 0]
    p = p - p[:, :1]
    return S._resample(p / N.palm_size(p[..., :2]).mean(), t, T)


def load():
    C = M.load_clips()
    P = np.stack([clip_points(r) for r in C["raw"]])
    assert np.allclose(P[:, :, 1:, :2].reshape(len(P), T, 40), C["shape"], atol=1e-9), "points != moving_shape"
    y = C["label"]
    yi = np.array([CLASSES.index(c if c in M.LETTERS else "static") for c in y])
    return dict(P=P, y=y, yi=yi, g=C["signer"], raw=C["raw"], shape=C["shape"])


def feats(P):
    return P[:, :, 1:, :2].reshape(len(P), -1)


# ---------------------------------------------------------------- augmentation
def _rot(yaw, pitch, roll):
    c, s, n = np.cos, np.sin, len(yaw)
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = c(yaw); Ry[:, 0, 2] = s(yaw); Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -s(yaw); Ry[:, 2, 2] = c(yaw)
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = c(pitch); Rx[:, 1, 2] = -s(pitch); Rx[:, 2, 1] = s(pitch); Rx[:, 2, 2] = c(pitch)
    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = c(roll); Rz[:, 0, 1] = -s(roll); Rz[:, 1, 0] = s(roll); Rz[:, 1, 1] = c(roll); Rz[:, 2, 2] = 1
    return Rz @ Rx @ Ry


def _warp(Q, rng):
    """Time stretch 0.85-1.15x and shift +-10% of the window (edges held)."""
    n = len(Q)
    a, b = rng.uniform(0.85, 1.15, n), rng.uniform(-0.1, 0.1, n)
    s = np.linspace(0, 1, T)
    pos = np.clip(a[:, None] * (s - 0.5) + 0.5 + b[:, None], 0, 1) * (T - 1)
    i0 = np.minimum(np.floor(pos).astype(int), T - 2)
    w = (pos - i0)[..., None, None]
    idx = np.arange(n)[:, None]
    return (1 - w) * Q[idx, i0] + w * Q[idx, i0 + 1]


def augment(P, cfg, rng):
    """One rotation (roll, optional 3D yaw/pitch +-15), aspect +-10% and shear +-0.1 per clip; optional time warp;
    jitter; then re-normalized like clip_points (wrist origin, clip-mean palm size)."""
    out = [P]
    n = len(P)
    for _ in range(cfg["copies"]):
        r = np.deg2rad(rng.uniform(-cfg["rot"], cfg["rot"], n))
        a3 = np.deg2rad(rng.uniform(-15, 15, (2, n))) if cfg["aug3d"] else np.zeros((2, n))
        Q = np.einsum("nij,ntkj->ntki", _rot(a3[0], a3[1], r), P)
        Q[..., 0] = Q[..., 0] * rng.uniform(0.9, 1.1, n)[:, None, None] + rng.uniform(-0.1, 0.1, n)[:, None, None] * Q[..., 1]
        if cfg["warp"]:
            Q = _warp(Q, rng)
        Q = Q + rng.normal(0, 0.02, Q.shape)
        Q = Q - Q[:, :, :1]
        out.append(Q / N.palm_size(Q[..., :2]).mean(1)[:, None, None, None])
    return np.concatenate(out)


# ---------------------------------------------------------------- model
def fit(cfg, P, yi, seed):
    rng = np.random.default_rng(seed)
    X = feats(augment(P, cfg, rng))
    sc = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]), alpha=cfg["alpha"], early_stopping=True,
                      max_iter=300, random_state=seed).fit(sc.transform(X), np.tile(yi, len(X) // len(P)))
    return dict(sc=sc, m=m)


def predict(members, P):
    X = feats(P)
    return np.mean([mb["m"].predict_proba(mb["sc"].transform(X)) for mb in members], 0)


def cv(cfg, D):
    def run(tr, te):
        return te, predict([fit(cfg, D["P"][tr], D["yi"][tr], s) for s in range(cfg["seeds"])], D["P"][te])
    probs = np.zeros((len(D["y"]), len(CLASSES)))
    for te, p in Parallel(n_jobs=-1)(delayed(run)(tr, te) for tr, te in GroupKFold(5).split(D["P"], D["y"], D["g"])):
        probs[te] = p
    return probs


def evaluate(probs, y):
    """Target-mode verification, letters rule: every other single letter passes as the target <= 5%, p >= 0.5."""
    res, thr = M.verify(-probs, y, True, rule="worst5", max_thr=-0.5)
    return res, {c: round(-v, 4) for c, v in thr.items()}, float(np.mean([res[c]["recall"] for c in M.LETTERS]))


# ---------------------------------------------------------------- export
def to_json(members, thresholds):
    R = lambda a: np.round(a, 6).tolist()
    out = []
    for mb in members:
        m = mb["m"]
        layers = [dict(W=R(m.coefs_[i].T), b=R(m.intercepts_[i]), act="relu") for i in range(len(m.coefs_) - 1)]
        layers.append(dict(W=R(m.coefs_[-1].T), b=R(m.intercepts_[-1]), act="softmax"))
        out.append(dict(mean=R(mb["sc"].mean_), std=R(mb["sc"].scale_), layers=layers))
    # W is [out][in]: y[o] = sum_i W[o][i]*x[i] + b[o]. Final probs = mean of the members' softmax.
    return dict(version=2, kind="moving_letters", features=FEATURES, T=T, frames=6, window_ms=1000,
                labels=CLASSES, display=M.DISPLAY, members=out, thresholds=thresholds)


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


def golden(model, D, members):
    rng = np.random.default_rng(0)
    idx = [int(i) for c in M.LETTERS for i in rng.choice(np.where(D["y"] == c)[0], 2, replace=False)]
    idx += [int(i) for i in rng.choice(np.where(~np.isin(D["y"], M.LETTERS))[0], 4, replace=False)]
    cases = []
    for i in idx:
        r = D["raw"][i]
        f = S.moving_shape(r["crop"], r["crop_wh"][0], r["crop_wh"][1], r["is_left"], r["t_ms"], T).reshape(-1)
        cases.append(dict(label=str(D["y"][i]), landmarks=M._nan_list(r["crop"]), img_w=r["crop_wh"][0], img_h=r["crop_wh"][1],
                          is_left=r["is_left"], t_ms=r["t_ms"].tolist(), features=np.round(f, 8).tolist(),
                          probs=forward(model, f).tolist()))
    sk = predict(members, D["P"][idx])
    err = float(np.abs(sk - np.array([c["probs"] for c in cases])).max())
    return dict(tolerance=1e-4, features=FEATURES, T=T, labels=CLASSES,
                note="landmarks: hand-crop HandLandmarker output per frame (null = no hand); fill gaps by linear "
                     "interpolation in time (edges held), then moving_shape, then the MLP", cases=cases), err


def report(trials, chosen, res, thr, ship):
    L = ["# Moving letters: MLP report", "",
         "Model: MLP on the hand-crop shape sequence (12 x 40 = 480 dims), classes = G H J N Ñ S Z + static. "
         f"Config: `{json.dumps(chosen)}`.",
         "Data: LSC70ANH crops, 6 frames per clip (first second, 6 fps), 70 signers; static = the 20 static letters' clips.",
         "All numbers are on held-out signers (GroupKFold 5). Target mode: pass = p(target) >= its threshold; threshold = "
         "lowest value (>= 0.5) at which every other single letter passes as the target <= 5%.",
         "Gate 3: recall >= 80% and the contact sheet (`qc/sheets/moving_<X>.png`) shows the motion in the first second.", "",
         "## Variant search (mean recall over the 7 letters)", "", "| trial | mean recall | kept |", "|---|---|---|"]
    L += [f"| {t['name']} | {t['mean_recall']:.3f} | {'yes' if t['kept'] else ''} |" for t in trials]
    L += ["", "## Per letter (chosen config)", "",
          "| letter | n | recall | worst other letter passing (rate) | FA static (avg) | threshold p | Gate 3 |", "|---|---|---|---|---|---|---|"]
    for c in M.LETTERS:
        r = res[c]
        L.append(f"| {M.DISPLAY.get(c, c)} | {r['n']} | {r['recall']:.2f} | {r['worst_static']} ({r['worst_static_fa']:.2f}) | "
                 f"{r['fa_static']:.3f} | {thr[c]} | {'pass' if r['gate3'] else 'fail'} |")
    L += ["", "DTW comparison (same rule): crop shape DTW 0.36 mean recall, with static reject templates 0.65, 0/7 pass "
          "(see `ml/NOTES.md`, phase 6).", "",
          f"Ship (camera-graded): {', '.join(M.DISPLAY.get(c, c) for c in ship) or 'none'}. The others stay video-only.", "",
          "## Caveats", "- LSC70 covers only the first second at 6 fps: the app must use the same window "
          "(about 1 s after the countdown, hand crop, resample to 12 by time).",
          "- No team clips yet: this is the dataset number, not the new-user/new-camera number.",
          "- `worst other letter` covers static letters only; moving-vs-moving confusions are inside the worst5 rule too."]
    open(f"{EXPORT}/moving_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main
def main():
    D = load()
    print(f"{len(D['y'])} clips, {len(set(D['g']))} signers", flush=True)
    trials, cfg = [], dict(BASE)

    def trial(name, c):
        res, thr, rec = evaluate(cv(c, D), D["y"])
        t = dict(name=name, cfg=c, mean_recall=rec, recall={k: res[k]["recall"] for k in M.LETTERS}, kept=False)
        trials.append(t)
        print(f"{name:14s} mean recall {rec:.3f}  " + " ".join(f"{M.DISPLAY.get(k, k)}={res[k]['recall']:.2f}" for k in M.LETTERS), flush=True)
        json.dump(trials, open(RUNS, "w"), indent=1)
        return t

    best = trial("base", cfg)
    best["kept"] = True
    if "--quick" not in sys.argv:
        for name, change in (("aug3d", dict(aug3d=True)), ("warp", dict(warp=True)), ("copies8", dict(copies=8)),
                             ("rot25", dict(rot=25)), ("small", dict(hidden=[128, 64])), ("alpha1e-2", dict(alpha=1e-2)),
                             ("ensemble5", dict(seeds=5))):
            t = trial(name, {**cfg, **change})
            if t["mean_recall"] >= best["mean_recall"] + 0.005:  # keep only real gains
                cfg, best = {**cfg, **change}, t
                t["kept"] = True
        json.dump(trials, open(RUNS, "w"), indent=1)
    print("chosen:", cfg, flush=True)
    probs = cv(cfg, D)
    res, thr, rec = evaluate(probs, D["y"])
    ship = [c for c in M.LETTERS if res[c]["gate3"]]
    members = [fit(cfg, D["P"], D["yi"], s) for s in range(cfg["seeds"])]
    model = to_json(members, thr)
    model["gate3"] = ship
    os.makedirs(f"{EXPORT}/golden", exist_ok=True)
    json.dump(model, open(f"{EXPORT}/moving_mlp.json", "w"), separators=(",", ":"))
    g, err = golden(model, D, members)
    json.dump(g, open(f"{EXPORT}/golden/moving_letters.json", "w"), separators=(",", ":"))
    report(trials, cfg, res, thr, ship)
    for c in M.LETTERS:
        r = res[c]
        print(f"  {M.DISPLAY.get(c, c):2s} recall={r['recall']:.2f} worst static {r['worst_static']}={r['worst_static_fa']:.2f} "
              f"thr={thr[c]} gate3={r['gate3']}", flush=True)
    print(f"final mean recall {rec:.3f}; ship {ship}; golden JSON vs sklearn max err {err:.2e}", flush=True)


if __name__ == "__main__":
    main()

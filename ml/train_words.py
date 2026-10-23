"""Words (moving signs): small MLP on the variant-C word sequence. Same move as moving letters (DTW -> MLP),
because kNN-DTW failed Gate 5 (top-1 0.81, 0/8 words at 80% with impostor FA <= 10%).

    python train_words.py           # variant search (signers held out) + cross-dataset + export
    python train_words.py --quick   # base config only + export

Input per clip (from sequences_words.npz, same app pipeline as DTW: 2.5 s window, idle trim, resample to 30 by time,
PoseLandmarker lite for shoulders + nose): per time step
    [shape 40 | locB 2 | locN 2 | shape2 40 | locB2 2] = 86 dims  (shape = handshape of the dominant hand,
    locB = shoulder-relative wrist, locN = nose-relative wrist, *2 = second hand, zeros if none).
Flattened time-major: [t][86]. Optional T=15: linear resample of the 30-step sequence (np.interp per dim).
Classes: the 8 MVP words + "other" (LSC50 impostor words, the only impostors with pose).
Target mode (the app knows the word): pass = p(word) >= thresholds[word] (>= 0.5, so it is the top guess).
Threshold = lowest value in [0.50, 0.95] where impostors pass as the word <= 10% AND every other single
MVP word passes as it <= 10%. Gate 5: >= 6 of 8 words with recall >= 0.80 under that rule.
Outputs: ml/out/words_mlp.json, golden/words_mlp.json, words_mlp_report.md. Run log: FEATS/words_runs.json.
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from words_dtw import FEATS, OUTDIR, WORDS, SOURCE, folds

EXPORT = OUTDIR
RUNS = f"{FEATS}/words_runs.json"
CLASSES = WORDS + ["other"]
FEATURES = "words_c_m1"

BASE = dict(T=30, hidden=[256, 128], alpha=1e-3, copies=4, rot=15, loc_shift=0.1, warp=True, shape2=True,
            seeds=3)
MAX_FA, MAX_OTHER, GATE, N_GATE = 0.10, 0.10, 0.80, 6
GRID = np.round(np.arange(0.50, 0.951, 0.01), 2)


# ---------------------------------------------------------------- data
def load():
    z = np.load(f"{FEATS}/sequences_words.npz", allow_pickle=True)
    n, T = z["X"].shape[:2]
    S1 = z["X"][:, :, :40].reshape(n, T, 20, 2).astype(np.float64)
    S2 = z["X2"][:, :, :40].reshape(n, T, 20, 2).astype(np.float64)
    LB, LN, LB2 = (z[k].astype(np.float64) for k in ("locB", "locN", "locB2"))
    ok = ~(np.isnan(S1).any((1, 2, 3)) | np.isnan(LB).any((1, 2)) | np.isnan(LN).any((1, 2))
           | np.isnan(S2).any((1, 2, 3)) | np.isnan(LB2).any((1, 2)))
    lab = z["label"].astype(str)
    ds = z["dataset"].astype(str)
    signer = np.array([f"{d}_{s}" for d, s in zip(z["dataset"], z["signer"])])
    mvp = np.isin(lab, WORDS)
    imp = (ds == "lsc50") & ~mvp
    keep = ok & (mvp | imp)
    y = np.where(mvp, lab, "other")[keep]
    D = dict(S1=S1[keep], LB=LB[keep], LN=LN[keep], S2=S2[keep], LB2=LB2[keep], y=y,
             yi=np.array([CLASSES.index(c) for c in y]), g=signer[keep], ds=ds[keep], lab=lab[keep],
             sid=z["sample_id"].astype(str)[keep])
    D["has2"] = np.abs(D["S2"]).sum((1, 2, 3)) > 0
    D["fold"] = folds(D["g"], D["ds"])
    return D


def subset(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(D["y"]) else v) for k, v in D.items()}


def resample_T(a, T):
    """[n, 30, ...] -> [n, T, ...] by linear interpolation on a uniform time grid (identity for T=30)."""
    if a.shape[1] == T:
        return a
    src, dst = np.linspace(0, 1, a.shape[1]), np.linspace(0, 1, T)
    i0 = np.minimum(np.floor(dst * (a.shape[1] - 1)).astype(int), a.shape[1] - 2)
    w = (dst * (a.shape[1] - 1) - i0).reshape((1, T) + (1,) * (a.ndim - 2))
    return (1 - w) * a[:, i0] + w * a[:, i0 + 1]


TIPS = [3, 7, 11, 15, 19]  # landmarks 4, 8, 12, 16, 20 (the shape block starts at landmark 1)


def layout(cfg):
    """[(block, dims)] per time step, in order. shape2: True (40), "tips" (fingertips, 10) or False.
    vel: per-step change of locB, locN, locB2 after resampling (first step 0)."""
    s2 = cfg.get("shape2", True)
    lay = [("shape", 40), ("locB", 2), ("locN", 2)]
    lay += [("shape2", 40)] if s2 is True else [("shape2_tips", 10)] if s2 == "tips" else []
    lay += [("locB2", 2)]
    if cfg.get("vel"):
        lay += [("dlocB", 2), ("dlocN", 2), ("dlocB2", 2)]
    return lay


def feats(D, cfg):
    """-> [n, T * sum(layout dims)] time-major."""
    n, T = len(D["S1"]), cfg["T"]
    blk = dict(shape=D["S1"].reshape(n, -1, 40), locB=D["LB"], locN=D["LN"], shape2=D["S2"].reshape(n, -1, 40),
               shape2_tips=D["S2"][:, :, TIPS].reshape(n, -1, 10), locB2=D["LB2"])
    blk = {k: resample_T(v, T) for k, v in blk.items()}
    for k in ("locB", "locN", "locB2"):
        blk["d" + k] = np.concatenate([np.zeros((n, 1, 2)), np.diff(blk[k], axis=1)], 1)
    return np.concatenate([blk[k] for k, _ in layout(cfg)], 2).reshape(n, -1)


def balance(D, cfg):
    """Oversample (whole clips) every MVP word up to the largest MVP word count."""
    if not cfg.get("balance"):
        return D
    cnt = {w: int((D["y"] == w).sum()) for w in WORDS}
    top = max(cnt.values())
    rng = np.random.default_rng(0)
    idx = [np.arange(len(D["y"]))]
    for w, c in cnt.items():
        if 0 < c < top:
            idx.append(rng.choice(np.where(D["y"] == w)[0], top - c))
    return subset(D, np.concatenate(idx))


# ---------------------------------------------------------------- augmentation
def _rot2(P, ang):
    c, s = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
    x, y = P[..., 0], P[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], -1)


def _warp(arrs, rng):
    """Same time stretch 0.85-1.15x and shift +-10% for every array of a clip (edges held)."""
    n, T = len(arrs[0]), arrs[0].shape[1]
    a, b = rng.uniform(0.85, 1.15, n), rng.uniform(-0.1, 0.1, n)
    pos = np.clip(a[:, None] * (np.linspace(0, 1, T) - 0.5) + 0.5 + b[:, None], 0, 1) * (T - 1)
    i0 = np.minimum(np.floor(pos).astype(int), T - 2)
    idx = np.arange(n)[:, None]
    out = []
    for A in arrs:
        w = (pos - i0).reshape((n, T) + (1,) * (A.ndim - 2))
        out.append((1 - w) * A[idx, i0] + w * A[idx, i0 + 1])
    return out


def augment(D, cfg, rng):
    """Per clip: in-plane rotation +-rot and aspect +-10% of each hand's shape (re-normalized to palm size via the
    mean distance wrist->middle MCP, which is point 9 -> index 8 here), body-position scale +-10% and shift
    +-loc_shift for each hand, optional joint time warp, jitter. A missing second hand stays all zeros."""
    keys = ("S1", "LB", "LN", "S2", "LB2")
    out = {k: [D[k]] for k in keys}
    n = len(D["S1"])
    m2 = D["has2"].astype(float)
    for _ in range(cfg["copies"]):
        new = {}
        for sk in ("S1", "S2"):
            P = _rot2(D[sk], np.deg2rad(rng.uniform(-cfg["rot"], cfg["rot"], n)))
            P[..., 0] *= rng.uniform(0.9, 1.1, n)[:, None, None]
            P = P + rng.normal(0, 0.02, P.shape)
            palm = np.linalg.norm(P[:, :, 8], axis=-1).mean(1)[:, None, None, None]
            new[sk] = P / np.maximum(palm, 1e-9)
        sc = rng.uniform(0.9, 1.1, n)[:, None, None]
        sh = rng.uniform(-cfg["loc_shift"], cfg["loc_shift"], (n, 1, 2))
        new["LB"] = D["LB"] * sc + sh + rng.normal(0, 0.02, D["LB"].shape)
        new["LN"] = D["LN"] * sc + sh + rng.normal(0, 0.02, D["LN"].shape)
        sh2 = rng.uniform(-cfg["loc_shift"], cfg["loc_shift"], (n, 1, 2))
        new["LB2"] = D["LB2"] * sc + sh2 + rng.normal(0, 0.02, D["LB2"].shape)
        if cfg["warp"]:
            new = dict(zip(keys, _warp([new[k] for k in keys], rng)))
        new["S2"] = new["S2"] * m2[:, None, None, None]
        new["LB2"] = new["LB2"] * m2[:, None, None]
        for k in keys:
            out[k].append(new[k])
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- model
def fit(cfg, D, seed):
    D = balance(D, cfg)
    A = augment(D, cfg, np.random.default_rng(seed)) if cfg["copies"] else D
    X = feats(A, cfg)
    yi = np.tile(D["yi"], len(X) // len(D["yi"]))
    sc = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]), alpha=cfg["alpha"], early_stopping=True,
                      max_iter=300, random_state=seed).fit(sc.transform(X), yi)
    return dict(sc=sc, m=m)


def predict(members, D, cfg):
    X = feats(D, cfg)
    return np.mean([mb["m"].predict_proba(mb["sc"].transform(X)) for mb in members], 0)


def cv(cfg, D):
    def run(f):
        tr, te = np.where(D["fold"] != f)[0], np.where(D["fold"] == f)[0]
        Dtr = subset(D, tr)
        return te, predict([fit(cfg, Dtr, s) for s in range(cfg["seeds"])], subset(D, te), cfg)
    probs = np.zeros((len(D["y"]), len(CLASSES)))
    for te, p in Parallel(n_jobs=-1)(delayed(run)(f) for f in range(5)):
        probs[te] = p
    return probs


def evaluate(probs, y, thr=None):
    """Target-mode verification per word. Returns per-word results, thresholds, mean recall, n words passing."""
    thr = dict(thr or {})
    imp = y == "other"
    top = probs[:, :len(WORDS)].argmax(1)
    res = {}
    for i, w in enumerate(WORDS):
        s, pos = probs[:, i], y == w
        others = [o for o in WORDS if o != w and (y == o).any()]
        if w not in thr:
            thr[w] = float(GRID[-1])
            for t in GRID:
                if (s[imp] >= t).mean() <= MAX_FA and max((s[y == o] >= t).mean() for o in others) <= MAX_OTHER:
                    thr[w] = float(t)
                    break
        t = thr[w]
        per_other = {o: float((s[y == o] >= t).mean()) for o in others}
        worst = max(per_other, key=per_other.get)
        rec = float((s[pos] >= t).mean())
        res[w] = dict(n=int(pos.sum()), top1=float((top[pos] == i).mean()), recall=rec,
                      imp_fa=float((s[imp] >= t).mean()), worst_other=worst, worst_other_fa=per_other[worst],
                      threshold=t, gate5=bool(rec >= GATE))
    mean = float(np.mean([res[w]["recall"] for w in WORDS]))
    return res, thr, mean, int(sum(res[w]["gate5"] for w in WORDS))


def cross(cfg, D):
    """Train on one dataset's MVP clips (+ all LSC50 impostors from other signers), test top-1 on the other."""
    out = {}
    for a, b in (("lsc50", "lsc54"), ("lsc54", "lsc50")):
        te = np.where((D["ds"] == b) & (D["y"] != "other"))[0]
        test_signers = set(D["g"][te])
        tr = np.where(((D["ds"] == a) | (D["y"] == "other")) & ~np.isin(D["g"], list(test_signers)))[0]
        p = predict([fit(cfg, subset(D, tr), s) for s in range(cfg["seeds"])], subset(D, te), cfg)
        yt = np.array([WORDS.index(c) for c in D["y"][te]])
        pr = p[:, :len(WORDS)].argmax(1)
        out[f"{a}->{b}"] = dict(top1=float((pr == yt).mean()),
                                per_word={w: float((pr[yt == i] == i).mean()) for i, w in enumerate(WORDS) if (yt == i).any()})
    return out


# ---------------------------------------------------------------- export
def to_json(members, cfg, thresholds, gate5):
    R = lambda a: np.round(a, 6).tolist()
    out = []
    for mb in members:
        m = mb["m"]
        layers = [dict(W=R(m.coefs_[i].T), b=R(m.intercepts_[i]), act="relu") for i in range(len(m.coefs_) - 1)]
        layers.append(dict(W=R(m.coefs_[-1].T), b=R(m.intercepts_[-1]), act="softmax"))
        out.append(dict(mean=R(mb["sc"].mean_), std=R(mb["sc"].scale_), layers=layers))
    lay = [[k, d] for k, d in layout(cfg)]
    # W is [out][in]: y[o] = sum_i W[o][i]*x[i] + b[o]. Final probs = mean of the members' softmax.
    return dict(version=2, kind="words", features=FEATURES, T_source=30, T=cfg["T"], window_ms=2500,
                layout=lay, needs_pose=True, labels=CLASSES, members=out, thresholds=thresholds, gate5=gate5,
                note="per time step concatenate the layout blocks (from sequences.clip_to_sequence at T=30; shape2_tips = landmarks 4,8,12,16,20 of shape2); "
                     "if T < 30 resample each block linearly on a uniform grid; dloc* = step-to-step change of the resampled loc (first step 0); flatten time-major; then the MLP")


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


def golden(model, D, members, cfg):
    """2 clips per word + 4 impostors: the T=30 blocks (input), the flat features and the expected probs."""
    rng = np.random.default_rng(0)
    idx = [int(i) for w in WORDS for i in rng.choice(np.where(D["y"] == w)[0], 2, replace=False)]
    idx += [int(i) for i in rng.choice(np.where(D["y"] == "other")[0], 4, replace=False)]
    Dq = subset(D, np.array(idx))
    F = feats(Dq, cfg)
    n = len(idx)
    cases = []
    for r, i in enumerate(idx):
        blocks = dict(shape=Dq["S1"][r].reshape(30, 40), locB=Dq["LB"][r], locN=Dq["LN"][r],
                      shape2=Dq["S2"][r].reshape(30, 40), locB2=Dq["LB2"][r])
        cases.append(dict(label=str(D["lab"][i]), sample_id=str(D["sid"][i]),
                          blocks={k: np.round(v, 8).tolist() for k, v in blocks.items()},
                          features=np.round(F[r], 8).tolist(), probs=forward(model, F[r]).tolist()))
    sk = predict(members, Dq, cfg)
    err = float(np.abs(sk - np.array([c["probs"] for c in cases])).max())
    assert n == len(cases)
    return dict(tolerance=1e-4, features=FEATURES, T=cfg["T"], labels=CLASSES, cases=cases), err


def report(trials, cfg, res, thr, ship, cr):
    L = ["# Words: MLP report", "",
         "Model: MLP on the variant-C word sequence (hand shape + shoulder/nose-relative position, both hands), "
         f"classes = 8 MVP words + other (LSC50 impostors). Config: `{json.dumps(cfg)}`.",
         "All numbers are on held-out signers (same 5 signer folds as the DTW eval). Target mode: pass = p(word) >= "
         f"its threshold; threshold = lowest value in [0.50, 0.95] where impostors pass as the word <= {MAX_FA:.0%} "
         f"and every other single MVP word passes as it <= {MAX_OTHER:.0%}. Thresholds are fit on the same "
         "out-of-fold scores, so recall is slightly optimistic.", "",
         "## Variant search", "", "| trial | mean recall | words >= 80% | mean top-1 | kept |", "|---|---|---|---|---|"]
    L += [f"| {t['name']} | {t['mean_recall']:.3f} | {t['n_pass']}/8 | {t['top1']:.3f} | {'yes' if t['kept'] else ''} |"
          for t in trials]
    L += ["", "## Per word (chosen config)", "",
          "| word | n | top-1 | recall @thr | impostor FA | worst other word (rate) | threshold | Gate 5 |",
          "|---|---|---|---|---|---|---|---|"]
    for w in WORDS:
        r = res[w]
        L.append(f"| {w} | {r['n']} | {r['top1']:.2f} | {r['recall']:.2f} | {r['imp_fa']:.3f} | "
                 f"{r['worst_other']} ({r['worst_other_fa']:.2f}) | {thr[w]} | {'pass' if r['gate5'] else 'fail'} |")
    L += ["", "## Cross-dataset (top-1, no reject)", ""]
    L += [f"- {k}: {v['top1']:.2f}  " + ", ".join(f"{w} {a:.2f}" for w, a in v["per_word"].items()) for k, v in cr.items()]
    L += ["", "DTW comparison (variant C, `ml/NOTES.md` phase 5): top-1 0.81, 0/8 words at 80% with impostor FA <= 10%, "
          "cross-dataset 0.56 / 0.58.", "",
          f"Gate 5 ({N_GATE} of 8 needed): {len(ship)}/8 pass -> {'PASS' if len(ship) >= N_GATE else 'FAIL'}. "
          f"Camera-graded: {', '.join(ship) or 'none'}. The others stay video-only (\"Mira y elige\").", "",
          "## Caveats", "- Impostors are LSC50 only (5 people, 42 words); LSC-54 non-MVP signs have no pose.",
          "- No team clips yet: the cross-dataset number is the closest to \"new user, new phone\".",
          "- The app pipeline is the DTW one: 2.5 s window, idle trim, resample to 30, PoseLandmarker lite."]
    open(f"{EXPORT}/words_mlp_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main
def main():
    D = load()
    print(f"{len(D['y'])} clips ({(D['y'] != 'other').sum()} MVP, {(D['y'] == 'other').sum()} impostors), "
          f"{len(set(D['g']))} signers", flush=True)
    base = dict(BASE)
    if "--base" in sys.argv:
        base.update(json.loads(sys.argv[sys.argv.index("--base") + 1]))
    round2 = "--round2" in sys.argv
    runs = RUNS.replace(".json", "_r2.json") if round2 else RUNS
    trials, cfg = [], base

    def trial(name, c):
        res, thr, rec, npass = evaluate(cv(c, D), D["y"])
        top1 = float(np.mean([res[w]["top1"] for w in WORDS]))
        t = dict(name=name, cfg=c, mean_recall=rec, n_pass=npass, top1=top1,
                 recall={w: res[w]["recall"] for w in WORDS}, kept=False)
        trials.append(t)
        print(f"{name:12s} recall {rec:.3f} pass {npass}/8 top1 {top1:.3f}  "
              + " ".join(f"{w}={res[w]['recall']:.2f}" for w in WORDS), flush=True)
        json.dump(trials, open(runs, "w"), indent=1)
        return t

    ROUND1 = (("no_aug", dict(copies=0)), ("no_warp", dict(warp=False)), ("T15", dict(T=15)),
              ("no_shape2", dict(shape2=False)), ("copies8", dict(copies=8)),
              ("loc_shift0.2", dict(loc_shift=0.2)), ("small", dict(hidden=[128, 64])),
              ("alpha1e-2", dict(alpha=1e-2)), ("ensemble5", dict(seeds=5)))
    ROUND2 = (("vel", dict(vel=True)), ("shape2_tips", dict(shape2="tips")), ("balance", dict(balance=True)),
              ("loc_shift0.3", dict(loc_shift=0.3)), ("warp", dict(warp=True)), ("T15", dict(T=15)))
    best = trial("base", cfg)
    best["kept"] = True
    if "--quick" not in sys.argv:
        for name, change in (ROUND2 if round2 else ROUND1):
            t = trial(name, {**cfg, **change})
            # keep only real gains in mean recall (n_pass jumps on noise at the 0.80 line); T15 kept if no loss (half size)
            if t["mean_recall"] >= best["mean_recall"] + (-0.003 if name == "T15" else 0.005):
                cfg, best = {**cfg, **change}, t
                t["kept"] = True
        json.dump(trials, open(runs, "w"), indent=1)
    print("chosen:", cfg, flush=True)
    res, thr, rec, npass = evaluate(cv(cfg, D), D["y"])
    cr = cross(cfg, D)
    for k, v in cr.items():
        print(f"cross {k}: top1 {v['top1']:.2f}", flush=True)
    ship = [w for w in WORDS if res[w]["gate5"]]
    members = [fit(cfg, D, s) for s in range(cfg["seeds"])]
    model = to_json(members, cfg, {w: round(thr[w], 4) for w in WORDS}, ship)
    os.makedirs(f"{EXPORT}/golden", exist_ok=True)
    json.dump(model, open(f"{EXPORT}/words_mlp.json", "w"), separators=(",", ":"))
    g, err = golden(model, D, members, cfg)
    json.dump(g, open(f"{EXPORT}/golden/words_mlp.json", "w"), separators=(",", ":"))
    report(trials, cfg, res, thr, ship, cr)
    for w in WORDS:
        r = res[w]
        print(f"  {w:14s} top1={r['top1']:.2f} recall={r['recall']:.2f} impFA={r['imp_fa']:.3f} "
              f"worst {r['worst_other']}={r['worst_other_fa']:.2f} thr={thr[w]} gate5={r['gate5']}", flush=True)
    print(f"final mean recall {rec:.3f}; Gate 5 {npass}/8 ({'PASS' if npass >= N_GATE else 'FAIL'}); ship {ship}; "
          f"golden JSON vs sklearn max err {err:.2e}", flush=True)


if __name__ == "__main__":
    main()

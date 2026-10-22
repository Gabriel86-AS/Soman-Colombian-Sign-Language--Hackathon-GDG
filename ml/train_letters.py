"""Phase 4 (v2): static-letter MLP on all 20 static letters + none. Compares variants on held-out
signers, keeps a change only when it helps, exports letters_mlp.json (v2) + golden vectors + report.

    python train_letters.py                  # explore variants, then export the best
    python train_letters.py --quick          # skip exploring, export the saved best config (runs.json)
    python train_letters.py --set numbers    # same process on the LSC70 numbers -> numbers_mlp.json
    python train_letters.py --set numbers --custom   # + our own captures (frames/custom, adds 2 and 3)

Our captures (capture/capture_custom.py -> capture/custom_to_frames.py) are dataset "custom", one signer
per person (team:<name>_p<id>); they join the same signer-split cross-validation as LSC70.

Env: SOMAN_OUT (frames dir), SOMAN_QC, SOMAN_EXPORT (default ml/out), SOMAN_CACHE (/content/cache).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import normalize as N
import sequences as S
from extract import OUT, STATIC, NUMBERS
from build_features import QC

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.environ.get("SOMAN_EXPORT", os.path.join(HERE, "out"))
CACHE = os.environ.get("SOMAN_CACHE", "/content/cache")
# name -> (labels, description for the report). Both use LSC70ANH hand crops (dataset lsc70h).
SETS = {"letters": (STATIC, "20 static letters"), "numbers": (NUMBERS, "10 numbers (1, 4-10, MIL, MILLON)")}
SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "letters"
LABELS, SET_DESC = SETS[SET]
CUSTOM = "--custom" in sys.argv
if CUSTOM and SET == "numbers":  # LSC70 has no 2 or 3; our captures do
    LABELS, SET_DESC = NUMBERS | {"2", "3"}, "12 numbers (1-10, MIL, MILLON; 2 and 3 only from our captures)"
RUNS = os.path.join(os.path.dirname(OUT.rstrip("/")), f"{SET}{'_custom' if CUSTOM else ''}_runs.json")
N_NONE = 2000
HOLD = 8
BASE = dict(feats="xy", hidden=[128, 64], alpha=1e-4, copies=2, rot=15, aug3d=False, extra=False, seeds=1)


# ---------------------------------------------------------------- data
def _read(ds, keep):
    n = len(os.listdir(os.path.join(OUT, ds)))  # part count in the name: new clips -> new cache
    path = f"{CACHE}/{ds}_{n}.parquet"
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        t = time.time()
        pd.read_parquet(os.path.join(OUT, ds)).to_parquet(path)
        print(f"cached {ds} in {time.time() - t:.0f}s", flush=True)
    d = pd.read_parquet(path)
    d["label"] = d.label.astype(str)
    return d[d.hand_idx >= 0].merge(keep, on=["dataset", "sample_id"]).reset_index(drop=True)


def load_keep(ds="lsc70h"):
    """Kept clips from qc/clips.csv; clips of this set missing there (extracted later) get the same rule."""
    keep = pd.read_csv(f"{QC}/clips.csv", dtype={"label": str})
    d = pd.read_parquet(os.path.join(OUT, ds))
    d["label"] = d.label.astype(str)
    d = d[d.label.isin(LABELS)]
    new = d[~d.sample_id.isin(keep[keep.dataset == ds].sample_id)]
    if len(new):
        from qc import clip_table
        new = new.assign(has_hand=new.hand_idx >= 0)
        keep = pd.concat([keep, clip_table(new)], ignore_index=True)
        print(f"qc keep rule applied to {new.sample_id.nunique()} new {ds} clips", flush=True)
    return keep[keep.keep.astype(bool)][["dataset", "sample_id"]]


def to_points(d):
    lm = d[S.HAND_COLS].to_numpy(float).reshape(-1, 21, 3)
    left = (d.handedness == "Left").to_numpy()
    P = np.empty_like(lm)
    for (w, h), idx in d.groupby(["img_w", "img_h"]).indices.items():
        for lf in (False, True):
            k = idx[left[idx] == lf]
            if len(k):
                P[k] = N.hand_points(lm[k], w, h, lf)
    return P, lm


def frame_filter(P, sample_ids):
    """Same rule as build_features: drop frames > 2x the sample's median distance from its median shape."""
    X = P[:, 1:, :2].reshape(len(P), -1)
    dist = np.zeros(len(P))
    for _, idx in pd.Series(sample_ids).groupby(sample_ids).indices.items():
        dist[idx] = np.linalg.norm(X[idx] - np.median(X[idx], 0), axis=1)
    medd = pd.Series(dist).groupby(sample_ids).transform("median").to_numpy()
    return ~((medd > 0) & (dist > 2 * medd))


def _letters(d, ds):
    d = d[d.label.isin(LABELS)].reset_index(drop=True)
    P, lm = to_points(d)
    ok = frame_filter(P, d.sample_id.to_numpy())
    d = d[ok].reset_index(drop=True)
    return dict(P=P[ok], lm=lm[ok], y=d.label.to_numpy(str), g=d.signer.to_numpy(str),
                sid=d.sample_id.to_numpy(str), w=d.img_w.to_numpy(), h=d.img_h.to_numpy(),
                hand=d.handedness.to_numpy(str), src=np.full(len(d), ds))


def load_letters(ds, keep):
    return _letters(_read(ds, keep), ds)


def load_custom():
    """Our captures: every frame has a hand (the capture only saves those), so no qc keep list."""
    d = pd.read_parquet(os.path.join(OUT, "custom"))
    d["label"] = d.label.astype(str)
    d[S.HAND_COLS] = d[S.HAND_COLS].astype(float)
    return _letters(d, "custom")


def load_none(keep):
    """Hands mid-sign (middle 60% of each clip) from LSC50 + LSC-54 word clips."""
    d = pd.concat([_read(ds, keep) for ds in ("lsc50", "lsc54")]).reset_index(drop=True)
    rel = d.groupby("sample_id").frame.transform(lambda f: (f - f.min()) / max(f.max() - f.min(), 1))
    d = d[(rel > 0.2) & (rel < 0.8)].sample(N_NONE, random_state=0).reset_index(drop=True)
    P, _ = to_points(d)
    return dict(P=P, y=np.full(len(d), "none"), g=d.signer.to_numpy(str),
                sid=np.array([f"none{i}" for i in range(len(d))]), src=np.full(len(d), "none"))


# ---------------------------------------------------------------- model
def _rot(yaw, pitch, roll):
    n = len(yaw)
    c, s = np.cos, np.sin
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = c(yaw); Ry[:, 0, 2] = s(yaw); Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -s(yaw); Ry[:, 2, 2] = c(yaw)
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = c(pitch); Rx[:, 1, 2] = -s(pitch); Rx[:, 2, 1] = s(pitch); Rx[:, 2, 2] = c(pitch)
    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = c(roll); Rz[:, 0, 1] = -s(roll); Rz[:, 1, 0] = s(roll); Rz[:, 1, 1] = c(roll); Rz[:, 2, 2] = 1
    return Rz @ Rx @ Ry


def augment(P, cfg, rng):
    """In-plane roll, optional 3D yaw/pitch (viewpoint), aspect +-10%, shear +-0.1, jitter; then
    re-normalized like hand_points (wrist origin, 2D palm size)."""
    out = [P]
    n = len(P)
    for _ in range(cfg["copies"]):
        r = np.deg2rad(rng.uniform(-cfg["rot"], cfg["rot"], n))
        a3 = np.deg2rad(rng.uniform(-15, 15, (2, n))) if cfg["aug3d"] else np.zeros((2, n))
        Q = np.einsum("nij,nkj->nki", _rot(a3[0], a3[1], r), P)
        Q[..., 0] = Q[..., 0] * rng.uniform(0.9, 1.1, n)[:, None] + rng.uniform(-0.1, 0.1, n)[:, None] * Q[..., 1]
        Q = Q + rng.normal(0, 0.02, Q.shape)
        Q = Q - Q[:, :1]
        out.append(Q / N.palm_size(Q[..., :2])[:, None, None])
    return np.concatenate(out)


def fit(cfg, P, y, seed):
    rng = np.random.default_rng(seed)
    Pa = augment(P, cfg, rng)
    X = N.letter_features(Pa, cfg["feats"])
    cls, yi = np.unique(np.tile(y, len(Pa) // len(P)), return_inverse=True)  # ints: sklearn early stopping
    sc = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]), alpha=cfg["alpha"], early_stopping=True,
                      max_iter=300, random_state=seed).fit(sc.transform(X), yi)
    return dict(sc=sc, m=m, cls=cls)


def predict(members, P, feats):
    X = N.letter_features(P, feats)
    return np.mean([mb["m"].predict_proba(mb["sc"].transform(X)) for mb in members], 0)


def cv(cfg, D, extra):
    folds = list(GroupKFold(5).split(D["P"], D["y"], D["g"]))

    def run(tr, te):
        P, y = D["P"][tr], D["y"][tr]
        if cfg["extra"]:  # full-frame LSC70 landmarks of training signers only
            k = np.isin(extra["g"], np.unique(D["g"][tr]))
            P, y = np.concatenate([P, extra["P"][k]]), np.r_[y, extra["y"][k]]
        members = [fit(cfg, P, y, s) for s in range(cfg["seeds"])]
        return te, predict(members, D["P"][te], cfg["feats"])

    probs = np.zeros((len(D["y"]), len(np.unique(D["y"]))))
    for te, p in Parallel(n_jobs=-1)(delayed(run)(tr, te) for tr, te in folds):
        probs[te] = p
    return probs


def score(probs, D, classes):
    y = D["y"]
    pred = classes[probs.argmax(1)]
    acc = {c: float((pred[y == c] == c).mean()) for c in classes}
    # sample level (letters): average the frames of one sample, like the app averaging over time
    df = pd.DataFrame(probs).assign(sid=D["sid"], y=y)
    ps = df.groupby("sid").mean(numeric_only=True)
    ys = df.groupby("sid").y.first().loc[ps.index].to_numpy()
    ps_pred = classes[ps.to_numpy().argmax(1)]
    acc_s = {c: float((ps_pred[ys == c] == c).mean()) for c in classes if c != "none"}
    return dict(macro=float(np.mean(list(acc.values()))), acc=acc, acc_sample=acc_s,
                macro_sample=float(np.mean(list(acc_s.values()))), pred=pred, probs=probs)


def thresholds(probs, y, classes):
    """Free mode: tau so <= 5% of none frames have a letter prob above it (floor 0.70).
    Target mode (the app knows the letter): per letter, the lowest tau_l in [0.50, 0.95] such that, for
    EVERY other class c (each other letter and none), <= 5% of c's frames reach p_l >= tau_l. The worst
    class matters (a learner making E when asked for C), not the average. 0.50 keeps the target the top guess."""
    ni = list(classes).index("none")
    tau = float(max(0.70, np.percentile(np.delete(probs[y == "none"], ni, 1).max(1), 95)))
    per, rec, worst = {}, {}, {}
    for i, c in enumerate(classes):
        if c == "none":
            continue
        t = 0.95
        for cand in np.arange(0.50, 0.951, 0.01):
            fa = {o: float((probs[y == o, i] >= cand).mean()) for o in classes if o != c}
            if max(fa.values()) <= 0.05:
                t = float(cand)
                break
        fa = {o: float((probs[y == o, i] >= t).mean()) for o in classes if o != c}
        o = max(fa, key=fa.get)
        per[c], rec[c], worst[c] = round(t, 2), float((probs[y == c, i] >= t).mean()), (o, fa[o])
    return tau, per, rec, worst


# ---------------------------------------------------------------- export
def to_json(members, feats, tau, per):
    cls = list(members[0]["cls"])
    labels = ["none"] + [c for c in cls if c != "none"]
    out = []
    for mb in members:
        m = mb["m"]
        order = [cls.index(l) for l in labels]
        R = lambda a: np.round(a, 6).tolist()  # halves the file; probs move ~1e-6 (golden uses the rounded model)
        layers = [dict(W=R(m.coefs_[i].T), b=R(m.intercepts_[i]), act="relu") for i in range(len(m.coefs_) - 1)]
        layers.append(dict(W=R(m.coefs_[-1][:, order].T), b=R(m.intercepts_[-1][order]), act="softmax"))
        out.append(dict(mean=R(mb["sc"].mean_), std=R(mb["sc"].scale_), layers=layers))
    # W is [out][in]: y[o] = sum_i W[o][i]*x[i] + b[o]. Final probs = mean of the members' softmax.
    return dict(version=2, points="hand_points_v1", features=feats, labels=labels, members=out,
                threshold=tau, thresholds=per, holdFrames=HOLD)


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


def golden(model, D, members, feats):
    rng = np.random.default_rng(1)
    idx = []
    for c in sorted(set(D["y"]) - {"none"}):
        idx.append(rng.choice(np.where(D["y"] == c)[0]))
    cases = [dict(landmarks=D["lm"][i].tolist(), img_w=int(D["w"][i]), img_h=int(D["h"][i]), handedness=D["hand"][i])
             for i in idx]
    for c in cases[:4]:  # phone-mirrored copies: must give the same features
        lm = np.array(c["landmarks"]); lm[:, 0] = 1 - lm[:, 0]
        cases.append(dict(landmarks=lm.tolist(), img_w=c["img_w"], img_h=c["img_h"],
                          handedness="Left" if c["handedness"] == "Right" else "Right"))
    for c in cases:
        p = N.hand_points(np.array(c["landmarks"]), c["img_w"], c["img_h"], c["handedness"] == "Left")
        c["points"] = p.tolist()
        c["features"] = N.letter_features(p, feats).tolist()
        c["probs"] = forward(model, c["features"]).tolist()
    cls = list(members[0]["cls"])
    order = [cls.index(l) for l in model["labels"]]
    sk = predict(members, np.array([c["points"] for c in cases]), feats)[:, order]
    err = float(np.abs(sk - np.array([c["probs"] for c in cases])).max())
    return dict(tolerance=1e-4, points="hand_points_v1", features=feats, labels=model["labels"], cases=cases), err


# ---------------------------------------------------------------- main
def main():
    quick = "--quick" in sys.argv
    os.makedirs(f"{EXPORT}/golden", exist_ok=True)
    keep = load_keep()
    L = load_letters("lsc70h", keep)
    if CUSTOM:
        C = load_custom()
        print(f"custom: {len(C['y'])} frames, signers {sorted(set(C['g']))}, labels {sorted(set(C['y']))}", flush=True)
        L = {k: np.concatenate([L[k], C[k]]) for k in L}
    Nn = load_none(keep)
    D = {k: np.concatenate([L[k], Nn[k]]) for k in ("P", "y", "g", "sid", "src")}
    classes = np.unique(D["y"])
    print(f"{SET} {len(L['y'])} frames ({len(set(L['y']))} classes, {len(set(L['g']))} signers), none {len(Nn['y'])}", flush=True)
    missing = sorted(LABELS - set(L["y"]))
    if missing:
        print("WARNING: no frames for", missing, flush=True)
    extra = None

    saved = json.load(open(RUNS)) if os.path.exists(RUNS) else dict(trials=[], chosen=None)
    log = []

    def load_extra():
        nonlocal extra
        if extra is None:
            try:
                E = load_letters("lsc70", keep)
            except (FileNotFoundError, ValueError):
                E = dict(P=np.zeros((0, 21, 3)), y=np.array([]), g=np.array([]))
            extra = dict(P=E["P"], y=E["y"], g=E["g"])
            print(f"extra full-frame lsc70: {len(E['y'])} frames", flush=True)

    def trial(name, cfg):
        if cfg["extra"]:
            load_extra()
            if len(extra["y"]) == 0:
                print(f"{name:28s} skipped (no full-frame lsc70 data for {SET})", flush=True)
                return dict(macro=-1.0)
        t = time.time()
        r = score(cv(cfg, D, extra), D, classes)
        log.append(dict(name=name, cfg=cfg, macro=r["macro"], macro_sample=r["macro_sample"]))
        saved["trials"].append(log[-1])
        json.dump(saved, open(RUNS, "w"), indent=1)
        print(f"{name:28s} macro {r['macro']:.4f}  sample {r['macro_sample']:.4f}  ({time.time() - t:.0f}s)", flush=True)
        return r

    if quick:
        # --quick [json]: rerun the saved choice; with json, also try choice+json and keep it if >= 0.5 point better
        best_cfg = saved["chosen"] or dict(BASE)
        prior = [e for e in saved["trials"] if not e["name"].startswith("quick:")]
        best = trial("quick: chosen", best_cfg)
        args = [a for a in sys.argv[1:] if a not in ("--quick", "--set", SET, "--custom")]
        if args:
            cfg = dict(best_cfg, **json.loads(args[0]))
            r = trial(f"quick: chosen + {args[0]}", cfg)
            if r["macro"] >= best["macro"] + 0.005:
                best_cfg, best = cfg, r
        log = prior + log
    else:
        # stage 1: features (simplest first; a richer set must win by >= 1 point: more Dart code)
        best_cfg, best = dict(BASE), trial("feats=xy", dict(BASE))
        for f in ("xy+z", "xy+geo", "xy+z+geo3"):
            cfg = dict(BASE, feats=f)
            r = trial(f"feats={f}", cfg)
            if r["macro"] >= best["macro"] + 0.01:
                best_cfg, best = cfg, r
        # stage 2: model and data, greedy, keep a change if it adds >= 0.5 point
        for name, change in (("wider (256,128) a=1e-3", dict(hidden=[256, 128], alpha=1e-3)),
                             ("aug x4 rot 25", dict(copies=4, rot=25)),
                             ("aug 3D viewpoint", dict(aug3d=True)),
                             ("+full-frame LSC70", dict(extra=True))):
            cfg = dict(best_cfg, **change)
            r = trial(name, cfg)
            if r["macro"] >= best["macro"] + 0.005:
                best_cfg, best = cfg, r
        # stage 3: 5-model ensemble (5x the app compute, still tiny); keep if >= 1 point
        cfg = dict(best_cfg, seeds=5)
        r = trial("ensemble x5", cfg)
        if r["macro"] >= best["macro"] + 0.01:
            best_cfg, best = cfg, r
    print("best config:", best_cfg, flush=True)
    saved["chosen"] = best_cfg
    json.dump(saved, open(RUNS, "w"), indent=1)

    # final: held-out results of the chosen config give the report + thresholds; then fit on everyone
    r = best
    tau, per, rec, worst = thresholds(r["probs"], D["y"], classes)
    P, y = D["P"], D["y"]
    if best_cfg["extra"]:
        load_extra()
        P, y = np.concatenate([P, extra["P"]]), np.r_[y, extra["y"]]
    members = Parallel(n_jobs=-1)(delayed(fit)(best_cfg, P, y, s) for s in range(best_cfg["seeds"]))
    model = to_json(members, best_cfg["feats"], tau, per)
    json.dump(model, open(f"{EXPORT}/{SET}_mlp.json", "w"))
    gold, err = golden(model, L, members, best_cfg["feats"])
    json.dump(gold, open(f"{EXPORT}/golden/{SET}.json", "w"))

    letters = [c for c in classes if c != "none"]
    y = D["y"]
    # motion check: a static model only fits signs whose handshape holds still over the 6 frames.
    # spread = mean distance of a frame to its sample's median shape (xy features, palm units)
    X = N.letter_features(L["P"], "xy")
    spread = np.zeros(len(X))
    for _, idx in pd.Series(L["sid"]).groupby(L["sid"]).indices.items():
        spread[idx] = np.linalg.norm(X[idx] - np.median(X[idx], 0), axis=1)
    spread = pd.Series(spread).groupby(L["y"]).mean()
    n_own = int((L["src"] == "custom").sum())
    own = f" + {n_own} frames of our own captures" if CUSTOM else ""
    conf = sorted(((float((r["pred"][y == a] == b).mean()), a, b) for a in letters for b in classes if a != b), reverse=True)[:12]
    rep = [f"# {SET.capitalize()} model report (v2)", "",
           f"Frames: {len(L['y'])} {SET} frames ({SET_DESC}, {len(set(L['g']))} signers, LSC70ANH crops{own}) + {len(Nn['y'])} none frames.",
           "All numbers are on held-out signers (GroupKFold 5 by signer).", "",
           f"## Variants tried (macro accuracy over {len(letters)} {SET} + none, frame level / sample level)", "",
           "| variant | frame | sample |", "|---|---|---|"]
    rep += [f"| {e['name']} | {e['macro']:.3f} | {e['macro_sample']:.3f} |" for e in log]
    rep += ["", f"Chosen: `{json.dumps(best_cfg)}`", "",
            "## Per sign", "",
            "Free = argmax over all classes. Target = the app knows the sign: frame passes if p(sign) >= its own threshold.",
            "Shape spread = how much the handshape changes within one sample (high = the sign moves; a static model may not fit).", "",
            "| sign | frame acc | sample acc | target threshold | target recall | worst false accept | shape spread |", "|---|---|---|---|---|---|---|"]
    rep += [f"| {c} | {r['acc'][c]:.3f} | {r['acc_sample'][c]:.3f} | {per[c]:.2f} | {rec[c]:.3f} | {worst[c][0]} {worst[c][1]:.3f} | {spread[c]:.3f} |" for c in letters]
    rep += [f"| none | {r['acc']['none']:.3f} | | | | |", "",
            f"Macro: frame {r['macro']:.3f}, sample {r['macro_sample']:.3f}.", "",
            "## Most confused (true -> predicted, frame level)", ""] + [f"- {a} -> {b}: {v:.3f}" for v, a, b in conf]
    if CUSTOM:
        # our captures, each person held out (their fold never saw them), plus LSC70 alone for comparison
        cm_ = D["src"] == "custom"
        ppl = sorted(set(D["g"][cm_]))
        cl = sorted(set(D["y"][cm_]), key=lambda c: (len(c), c))
        rep += ["", "## Our captures (each person held out)", "",
                "| sign | " + " | ".join(p.split(":")[-1] for p in ppl) + " | all | frames |",
                "|---|" + "---|" * (len(ppl) + 2)]
        for c in cl:
            cells = []
            for p in ppl:
                k = cm_ & (D["g"] == p) & (y == c)
                cells.append(f"{(r['pred'][k] == c).mean():.2f}" if k.any() else "")
            k = cm_ & (y == c)
            rep.append(f"| {c} | " + " | ".join(cells) + f" | {(r['pred'][k] == c).mean():.3f} | {int(k.sum())} |")
        lm_ = D["src"] != "custom"
        old = [c for c in classes if c not in ("2", "3")]
        rep += ["", f"Our captures macro {np.mean([(r['pred'][cm_ & (y == c)] == c).mean() for c in cl]):.3f}; "
                f"LSC70 + none frames only, macro {np.mean([(r['pred'][lm_ & (y == c)] == c).mean() for c in old]):.3f}."]
    rep += ["", f"Free-mode threshold tau = {tau:.3f} (<= 5% of none frames above it, floor 0.70). holdFrames = {HOLD}.",
            "Target thresholds: for every other class, <= 5% of its frames reach the threshold (per frame, before the 8-frame hold).", "",
            f"Golden: {len(gold['cases'])} cases (4 mirrored); JSON forward pass vs sklearn max error {err:.2e}"]
    open(f"{EXPORT}/{SET}_report.md", "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print("\n".join(rep), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        labs = letters + ["none"]
        cm = confusion_matrix(y, r["pred"], labels=labs, normalize="true")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cm, cmap="Blues", vmax=1)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if cm[i, j] >= 0.03:
                    ax.text(j, i, f"{cm[i, j]:.2f}"[1:], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labs)), labs)
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.savefig(f"{EXPORT}/{SET}_confusion.png", dpi=110, bbox_inches="tight")
    except Exception as e:
        print("no confusion png:", e)


if __name__ == "__main__":
    main()

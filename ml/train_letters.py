"""Phase 4: static-letter MLP. Train, choose letters, export letters_mlp.json + golden vectors.

    python train_letters.py

Env: SOMAN_OUT (frames dir), SOMAN_QC, SOMAN_FEATS (as build_features.py), SOMAN_EXPORT (default ml/out).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import normalize as N
import sequences as S
from build_features import FEATS, QC
from extract import OUT

EXPORT = os.environ.get("SOMAN_EXPORT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
rng = np.random.default_rng(0)
N_NONE = 2000
MAX_LETTERS = 12


def load_none(keep):
    """Per-frame features of hands mid-sign from word clips (LSC50 + LSC-54)."""
    parts = []
    for ds in ("lsc50", "lsc54"):
        d = pd.read_parquet(os.path.join(OUT, ds))
        d = d[d.hand_idx >= 0].merge(keep, on=["dataset", "sample_id"])
        parts.append(d)
    d = pd.concat(parts)
    # frames from the middle 60% of each clip (mid-sign), not hands entering/leaving
    rel = d.groupby("sample_id").frame.transform(lambda f: (f - f.min()) / max(f.max() - f.min(), 1))
    d = d[(rel > 0.2) & (rel < 0.8)]
    d = d.sample(min(N_NONE, len(d)), random_state=0)
    lm = d[S.HAND_COLS].to_numpy(float).reshape(-1, 21, 3)
    left = (d.handedness == "Left").to_numpy()
    X = np.stack([N.handshape(lm[i], d.img_w.iat[i], d.img_h.iat[i], left[i]) for i in range(len(d))])
    return X, d.signer.to_numpy()


def augment(X, copies=2):
    """Rotation +-15 deg, scale +-10%, shear +-0.1, jitter sigma 0.02 palm units."""
    out = [X]
    for _ in range(copies):
        P = X.reshape(len(X), 20, 2)
        a = np.deg2rad(rng.uniform(-15, 15, len(X)))
        c, s = np.cos(a), np.sin(a)
        sc = rng.uniform(0.9, 1.1, len(X))
        sh = rng.uniform(-0.1, 0.1, len(X))
        M = np.zeros((len(X), 2, 2))
        M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1] = c, -s, s, c
        Sh = np.tile(np.eye(2), (len(X), 1, 1))
        Sh[:, 0, 1] = sh
        Q = np.einsum("nij,njk,nlk->nli", M, Sh, P) * sc[:, None, None]
        Q = Q + rng.normal(0, 0.02, Q.shape)
        out.append(Q.reshape(len(X), 40))
    return np.vstack(out)


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300, random_state=0)


def fit_scaled(model, Xtr, ytr, aug=True):
    if aug:
        n = len(Xtr)
        Xa = augment(Xtr)
        ytr = np.tile(ytr, len(Xa) // n)
        Xtr = Xa
    sc = StandardScaler().fit(Xtr)
    model.fit(sc.transform(Xtr), ytr)
    return sc, model


def cv(X, y, groups, make, aug):
    oof = np.empty(len(y), dtype=object)
    probs = None
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc, m = fit_scaled(make(), X[tr], y[tr], aug)
        p = m.predict_proba(sc.transform(X[te]))
        if probs is None:
            probs = np.zeros((len(y), len(m.classes_)))
            classes = m.classes_
        probs[te] = p
        oof[te] = m.classes_[p.argmax(1)]
    return oof, probs, classes


def choose(letters, acc, conf):
    """Accuracy >= 0.90, then drop the weaker letter of any pair confused > 10%; keep the best 12."""
    keep = [l for l in letters if acc[l] >= 0.90]
    changed = True
    while changed:
        changed = False
        for a in keep:
            for b in keep:
                if a != b and conf.get((a, b), 0) > 0.10:
                    keep.remove(a if acc[a] <= acc[b] else b)
                    changed = True
                    break
            if changed:
                break
    return sorted(sorted(keep, key=lambda l: -acc[l])[:MAX_LETTERS])


def main():
    os.makedirs(EXPORT, exist_ok=True)
    os.makedirs(f"{EXPORT}/golden", exist_ok=True)
    z = np.load(f"{FEATS}/letters.npz", allow_pickle=True)
    keep = pd.read_csv(f"{QC}/clips.csv")
    keep = keep[keep.keep][["dataset", "sample_id"]]
    Xl, yl, gl = z["X"], z["label"].astype(str), z["signer"].astype(str)
    Xn, gn = load_none(keep)
    X = np.vstack([Xl, Xn])
    y = np.r_[yl, np.full(len(Xn), "none")]
    g = np.r_[gl, gn]
    letters = sorted(set(yl))
    report = ["# Letter model report", "", f"Frames: {len(Xl)} letters ({len(letters)} classes, 70 signers) + {len(Xn)} none."]

    # baselines (no augmentation) and MLP (augmented), signer-grouped 5-fold
    base = {}
    for name, mk, aug in (("kNN k=5", lambda: KNeighborsClassifier(5), False),
                          ("logreg", lambda: LogisticRegression(max_iter=500), False),
                          ("MLP (128,64) + aug", make_mlp, True)):
        oof, probs, classes = cv(X, y, g, mk, aug)
        base[name] = float((oof == y).mean())
        print(name, base[name], flush=True)
        last = (oof, probs, classes)
    oof, probs, classes = last
    report += ["", "## Held-out-signer accuracy (all frames)", ""] + [f"- {k}: {v:.3f}" for k, v in base.items()]

    acc = {l: float((oof[y == l] == l).mean()) for l in letters}
    conf = {}
    for a in letters:
        for b in set(letters) | {"none"}:
            if a != b:
                conf[(a, b)] = float((oof[y == a] == b).mean())
    pairs = sorted(((v, a, b) for (a, b), v in conf.items() if v > 0.02), reverse=True)[:10]
    report += ["", "## Per-letter accuracy (MLP, held-out signers)", ""] + [f"- {l}: {acc[l]:.3f}" for l in letters]
    report += ["", "## Most confused (true -> predicted)", ""] + [f"- {a} -> {b}: {v:.3f}" for v, a, b in pairs]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        labs = letters + ["none"]
        cm = confusion_matrix(y, oof, labels=labs, normalize="true")
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labs)), labs)
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.savefig(f"{EXPORT}/letters_confusion.png", dpi=110, bbox_inches="tight")
    except Exception as e:
        print("no confusion png:", e)

    chosen = choose(letters, acc, conf)
    n90 = sum(a >= 0.90 for a in acc.values())
    report += ["", f"Letters at >= 90%: {n90} of {len(letters)}. **Gate 2 (>= 8): {'PASS' if n90 >= 8 else 'FAIL'}**",
               f"Chosen MVP letters ({len(chosen)}): {' '.join(chosen)}"]
    print("chosen", chosen, flush=True)

    # final set: chosen letters + none (dropped letters go into none, subsampled)
    dropped = [l for l in letters if l not in chosen]
    mask_c = np.isin(yl, chosen)
    Xd = Xl[np.isin(yl, dropped)]
    Xd = Xd[rng.choice(len(Xd), min(len(Xd), 600), replace=False)] if len(Xd) else Xd
    Xf = np.vstack([Xl[mask_c], Xn, Xd])
    yf = np.r_[yl[mask_c], np.full(len(Xn) + len(Xd), "none")]
    gf = np.r_[gl[mask_c], gn, np.full(len(Xd), "dropped")]
    oof2, probs2, classes2 = cv(Xf, yf, gf, make_mlp, True)
    acc2 = {l: float((oof2[yf == l] == l).mean()) for l in chosen}
    report += ["", "## Final set, held-out signers", ""] + [f"- {l}: {acc2[l]:.3f}" for l in chosen]
    report += [f"- none: {float((oof2[yf == 'none'] == 'none').mean()):.3f}"]
    ni = list(classes2).index("none")
    nonenone = np.delete(probs2[yf == "none"], ni, axis=1).max(1)
    tau = float(max(0.70, np.percentile(nonenone, 95)))
    report += ["", f"Threshold tau = {tau:.3f} (<= 5% of none frames above it; floor 0.70). holdFrames = 8."]

    sc, m = fit_scaled(make_mlp(), Xf, yf, True)
    labels = list(m.classes_)
    labels = ["none"] + [l for l in labels if l != "none"]
    order = [list(m.classes_).index(l) for l in labels]
    Wout = m.coefs_[-1][:, order]
    bout = m.intercepts_[-1][order]
    layers = [dict(W=m.coefs_[i].T.tolist(), b=m.intercepts_[i].tolist(), act="relu") for i in range(len(m.coefs_) - 1)]
    layers.append(dict(W=Wout.T.tolist(), b=bout.tolist(), act="softmax"))
    # W is stored [out][in] so Dart does y[o] = sum_i W[o][i]*x[i] + b[o]
    json.dump(dict(version=1, features="xy_1to20_v1", labels=labels, mean=sc.mean_.tolist(),
                   std=sc.scale_.tolist(), layers=layers, threshold=tau, holdFrames=8),
              open(f"{EXPORT}/letters_mlp.json", "w"))

    # golden vectors: raw landmarks -> features -> probabilities, using the exported JSON only
    model = json.load(open(f"{EXPORT}/letters_mlp.json"))
    d = pd.concat([pd.read_parquet(os.path.join(OUT, "lsc70"), filters=[("label", "in", chosen)])])
    d = d[d.hand_idx >= 0].sample(16, random_state=1)
    cases = []
    for _, r in d.iterrows():
        lm = r[S.HAND_COLS].to_numpy(float).reshape(21, 3)
        cases.append(dict(landmarks=lm.tolist(), img_w=int(r.img_w), img_h=int(r.img_h), handedness=r.handedness))
    for c in cases[:4]:  # phone-mirrored versions of 4 cases: must give the same features
        lm = np.array(c["landmarks"])
        lm[:, 0] = 1 - lm[:, 0]
        cases.append(dict(landmarks=lm.tolist(), img_w=c["img_w"], img_h=c["img_h"],
                          handedness="Left" if c["handedness"] == "Right" else "Right"))
    for c in cases:
        f = N.handshape(np.array(c["landmarks"]), c["img_w"], c["img_h"], c["handedness"] == "Left")
        c["features"] = f.tolist()
        c["probs"] = forward(model, f).tolist()
    json.dump(dict(tolerance=1e-4, labels=model["labels"], cases=cases), open(f"{EXPORT}/golden/letters.json", "w"))
    sk = m.predict_proba(sc.transform(np.array([c["features"] for c in cases])))[:, order]
    err = np.abs(sk - np.array([c["probs"] for c in cases])).max()
    report += ["", f"Golden: {len(cases)} cases (4 mirrored); JSON forward pass vs sklearn max error {err:.2e}"]
    open(f"{EXPORT}/letters_report.md", "w", encoding="utf-8").write("\n".join(report) + "\n")
    print("\n".join(report))


def forward(model, f):
    x = (np.asarray(f) - model["mean"]) / model["std"]
    for L in model["layers"]:
        x = np.asarray(L["W"]) @ x + np.asarray(L["b"])
        if L["act"] == "relu":
            x = np.maximum(x, 0)
        else:
            e = np.exp(x - x.max())
            x = e / e.sum()
    return x


if __name__ == "__main__":
    main()

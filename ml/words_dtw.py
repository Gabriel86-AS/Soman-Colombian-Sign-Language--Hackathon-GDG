"""Phase 5: kNN-DTW word recognition. Evaluate (signer-held-out), pick thresholds, export templates.

    python words_dtw.py eval          # variants A, B, C + cross-dataset  -> FEATS/words_eval.json
    python words_dtw.py export A      # templates, thresholds, golden for the chosen variant -> ml/out/

Env: SOMAN_FEATS (features dir, has sequences_words.npz, two_handed_words.json).
"""
import json
import os
import sys

import numpy as np

import dtw as D

FEATS = os.environ.get("SOMAN_FEATS", "/content/drive/MyDrive/soman-ml/features")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
WORDS = ["GRACIAS", "BUENOS_DIAS", "BUENAS_TARDES", "BUENAS_NOCHES", "BIEN", "HOLA", "ADIOS", "POR_FAVOR"]
K, N_TEMPL, BAND_FRAC, T = 5, 40, 0.10, 30
BAND = max(1, round(BAND_FRAC * T))
FEATURE_VERSION = "v2"
SOURCE = {"lsc50": "LSC50", "lsc54": "LSC-54", "team": "team"}


def layout(variant):
    """[(feature, source array, weight)] per variant; weights are part of the feature contract."""
    if variant == "A":
        return [("shape", "X[:, :40]", 1.0), ("locA", "X[:, 40:]", 1.0)]
    if variant == "B":
        return [("shape", "X[:, :40]", 1.0), ("locB", "locB", 3.0)]
    if variant == "C":
        return [("shape", "X[:, :40]", 1.0), ("locB", "locB", 3.0), ("locN", "locN", 3.0),
                ("shape2", "X2[:, :40]", 0.3), ("locB2", "locB2", 3.0)]
    raise ValueError(variant)


def load(variant):
    z = np.load(f"{FEATS}/sequences_words.npz", allow_pickle=True)
    src = dict(X=z["X"], X2=z["X2"], locB=z["locB"], locN=z["locN"], locB2=z["locB2"])
    parts = []
    for _, expr, wgt in layout(variant):
        name, _, sl = expr.partition("[")
        a = src[name].astype(np.float64)
        if sl:
            a = a[:, :, 40:] if sl.startswith(":, 40:") else a[:, :, :40]
        parts.append(wgt * a)
    F = np.concatenate(parts, 2)
    ok = ~np.isnan(F).any((1, 2))
    meta = dict(label=z["label"].astype(str), ds=z["dataset"].astype(str), sid=z["sample_id"].astype(str),
                signer=np.array([f"{d}_{s}" for d, s in zip(z["dataset"], z["signer"])]))
    return F, meta, ok


def folds(signer, ds, n=5, seed=0):
    """LSC50 persons are spread one per fold; LSC-54 signers are shuffled round-robin."""
    rng = np.random.default_rng(seed)
    a = sorted({s for s, d in zip(signer, ds) if d == "lsc50"})
    b = sorted({s for s, d in zip(signer, ds) if d == "lsc54"})
    rng.shuffle(b)
    fold = {s: i % n for i, s in enumerate(a)}
    fold.update({s: i % n for i, s in enumerate(b)})
    return np.array([fold[s] for s in signer])


def medoids(F, k):
    """k medoids by farthest-first init + a few PAM-style updates on the DTW matrix. Returns indices."""
    if len(F) <= k:
        return np.arange(len(F))
    dm = D.dist_matrix(F, F, BAND)
    dm = (dm + dm.T) / 2
    sel = [int(dm.sum(1).argmin())]
    while len(sel) < k:
        sel.append(int(dm[:, sel].min(1).argmax()))
    for _ in range(10):
        asg = dm[:, sel].argmin(1)
        new = []
        for c in range(k):
            m = np.where(asg == c)[0]
            new.append(int(m[dm[np.ix_(m, m)].sum(1).argmin()]) if len(m) else sel[c])
        if new == sel:
            break
        sel = new
    return np.array(sel)


def build_templates(F, labels, train_idx, n_templ=N_TEMPL):
    """Indices (into F) of the templates: n_templ medoids per word from the training clips."""
    out = []
    for w in WORDS:
        idx = train_idx[labels[train_idx] == w]
        out += list(idx[medoids(F[idx], n_templ)])
    return np.array(out)


def score(F, labels, tidx, qidx):
    dm = D.dist_matrix(F[qidx], F[tidx], BAND)
    return D.class_scores(dm, labels[tidx], WORDS, K)


def oof(variant, n_templ=N_TEMPL):
    F, m, ok = load(variant)
    lab = m["label"]
    fold = folds(m["signer"], m["ds"])
    S = np.full((len(F), len(WORDS)), np.nan)
    for f in range(5):
        tr = np.where(ok & (fold != f) & np.isin(lab, WORDS))[0]
        te = np.where(ok & (fold == f))[0]
        tidx = build_templates(F, lab, tr, n_templ)
        S[te] = score(F, lab, tidx, te)
    return S, m, ok


def _best(S):
    pred = np.nanargmin(np.where(np.isnan(S), np.inf, S), 1)
    return pred, np.take_along_axis(S, pred[:, None], 1)[:, 0]


def thresholds(S, lab, keep, start_pct=95.0, max_fa=0.10):
    """Per word: 95th percentile of the correct-match distances, lowered until impostor false-accept <= max_fa."""
    mvp = keep & np.isin(lab, WORDS)
    imp = keep & ~np.isin(lab, WORDS)
    pred, best = _best(S)
    pct = {w: start_pct for w in WORDS}
    thr = {}

    def set_thr(w):
        sel = mvp & (lab == w) & (pred == WORDS.index(w))
        thr[w] = float(np.percentile(best[sel], max(pct[w], 0))) if sel.any() else 0.0

    def accepted():
        limit = np.array([thr[WORDS[p]] for p in pred])
        return keep & (best <= limit)

    for w in WORDS:
        set_thr(w)
    acc = imp & accepted()
    fa = acc.sum() / max(imp.sum(), 1)
    while fa > max_fa and min(pct.values()) > 0:
        cnt = {w: int((acc & (pred == WORDS.index(w))).sum()) for w in WORDS}
        w = max(cnt, key=cnt.get)
        pct[w] -= 2.5
        set_thr(w)
        acc = imp & accepted()
        fa = acc.sum() / max(imp.sum(), 1)
    return thr, pct, float(fa)


def metrics(S, lab, keep, thr):
    mvp = keep & np.isin(lab, WORDS)
    imp = keep & ~np.isin(lab, WORDS)
    pred, best = _best(S)
    acc = keep & (best <= np.array([thr[WORDS[p]] for p in pred]))
    res = {}
    for i, w in enumerate(WORDS):
        sel = mvp & (lab == w)
        res[w] = dict(n=int(sel.sum()), top1=float((pred[sel] == i).mean()),
                      recall=float(((pred[sel] == i) & acc[sel]).mean()),
                      confused_with=sorted({WORDS[p] for p in pred[sel] if p != i}))
    res["_impostor_false_accept"] = float((imp & acc).sum() / max(imp.sum(), 1))
    res["_n_impostors"] = int(imp.sum())
    return res


def cross(variant, thr):
    F, m, ok = load(variant)
    lab, ds = m["label"], m["ds"]
    out = {}
    for a, b in (("lsc50", "lsc54"), ("lsc54", "lsc50")):
        tr = np.where(ok & (ds == a) & np.isin(lab, WORDS))[0]
        te = np.where(ok & (ds == b) & np.isin(lab, WORDS))[0]
        tidx = build_templates(F, lab, tr, N_TEMPL)
        pred, best = _best(score(F, lab, tidx, te))
        yt = np.array([WORDS.index(l) for l in lab[te]])
        acc = best <= np.array([thr[WORDS[p]] for p in pred])
        out[f"{a}->{b}"] = dict(
            top1=float((pred == yt).mean()), recall=float(((pred == yt) & acc).mean()),
            per_word={w: float((pred[yt == i] == i).mean()) for i, w in enumerate(WORDS) if (yt == i).any()})
    return out


def evaluate():
    report = {}
    for v in ("A", "B", "C"):
        S, m, keep = oof(v)
        lab = m["label"]
        thr, pct, fa = thresholds(S, lab, keep)
        report[v] = dict(metrics=metrics(S, lab, keep, thr), thr=thr, pct=pct, cross=cross(v, thr))
        r = report[v]["metrics"]
        print(f"\n== variant {v}  impostor FA {r['_impostor_false_accept']:.3f} (n={r['_n_impostors']})")
        for w in WORDS:
            print(f"  {w:14s} n={r[w]['n']:3d} top1={r[w]['top1']:.2f} recall@thr={r[w]['recall']:.2f} "
                  f"thr={thr[w]:.3f} confused={r[w]['confused_with']}")
        for k, c in report[v]["cross"].items():
            print(f"  cross {k}: top1={c['top1']:.2f} recall={c['recall']:.2f} {c['per_word']}")
    json.dump(report, open(f"{FEATS}/words_eval.json", "w"), indent=1)


def export(variant):
    F, m, ok = load(variant)
    F = np.round(F, 3)  # the app stores 3 decimals; scores below are computed on exactly what it gets
    lab, ds = m["label"], m["ds"]
    S, _, keep = oof(variant)
    thr, pct, fa = thresholds(S, lab, keep)
    met = metrics(S, lab, keep, thr)
    two = set(json.load(open(f"{FEATS}/two_handed_words.json")))
    os.makedirs(f"{OUTDIR}/templates", exist_ok=True)
    os.makedirs(f"{OUTDIR}/golden", exist_ok=True)
    allidx = np.where(ok & np.isin(lab, WORDS))[0]
    tidx = build_templates(F, lab, allidx, N_TEMPL)
    for w in WORDS:
        ti = tidx[lab[tidx] == w]
        json.dump(dict(word=w, feature_version=FEATURE_VERSION, variant=variant, T=T, dim=int(F.shape[2]),
                       band=BAND, k=K, layout=[[n, wg] for n, _, wg in layout(variant)], threshold=round(thr[w], 5), two_handed=w in two,
                       templates=[dict(source=SOURCE[ds[i]], X=F[i].tolist()) for i in ti]),
                  open(f"{OUTDIR}/templates/{w}.json", "w"), separators=(",", ":"))
    # golden: queries (2 per word + 4 impostors) and their class scores against the exported templates
    rng = np.random.default_rng(0)
    q = []
    for w in WORDS:
        q += list(rng.choice(np.where(ok & (lab == w))[0], 2, replace=False))
    q += list(rng.choice(np.where(ok & ~np.isin(lab, WORDS))[0], 4, replace=False))
    sc = score(F, lab, tidx, np.array(q))
    cases = []
    for r, i in enumerate(q):
        p = int(sc[r].argmin())
        cases.append(dict(label=str(lab[i]), X=F[i].tolist(),
                          scores={w: round(float(sc[r, j]), 5) for j, w in enumerate(WORDS)},
                          predicted=WORDS[p], accepted=bool(sc[r, p] <= thr[WORDS[p]])))
    a, b = F[q[0]], F[q[3]]
    json.dump(dict(feature_version=FEATURE_VERSION, variant=variant, band=BAND, k=K, tolerance=1e-4,
                   dtw_pairs=[dict(a=a.tolist(), b=b.tolist(), dist=float(D.dtw(a, b, BAND)))], cases=cases),
              open(f"{OUTDIR}/golden/words.json", "w"), separators=(",", ":"))
    json.dump(dict(variant=variant, thresholds=thr, percentile=pct, metrics=met),
              open(f"{OUTDIR}/words_report.json", "w"), indent=1)
    print("exported", variant, "impostor FA", fa)


if __name__ == "__main__":
    {"eval": evaluate, "export": lambda: export(sys.argv[2])}[sys.argv[1]]()

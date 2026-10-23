"""Phase 6 (M5): moving letters G H J N Ñ S Z with kNN-DTW. Motion sheets, evaluation, export.

    python extract.py lsc70h_moving        # once: LSC70ANH hand crops of the moving letters (dataset lsc70h)
    python moving_dtw.py sheets            # 6 crops + wrist path per clip -> QC/sheets/moving_<X>.png (Gate 3 eye check)
    python moving_dtw.py eval              # all variants, signer-held-out -> FEATS/moving_eval.json
    python moving_dtw.py mlp               # comparison: small MLP on the crop-shape sequence -> FEATS/moving_mlp_eval.json
    python moving_dtw.py export <variant>  # templates, golden, report -> ml/out/  (add --all to skip Gate 3)

A clip = LSC70's 6 frames (first second, 6 fps). Shape comes from the hand crop (lsc70h), the wrist path
from the full frame (lsc70): same sample_id and frame index. Negatives = the 20 static letters' clips
(a held I must not pass as J). The app knows the target letter, so this is verification:
pass = the target wins the kNN vote AND its score is under the target's threshold.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import dtw as D
import sequences as S
from build_features import FEATS, QC
from extract import MANIFEST, MOVING, OUT, RAW, STATIC, _frames

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.environ.get("SOMAN_EXPORT", os.path.join(HERE, "out"))
CACHE = os.environ.get("SOMAN_CACHE", "/content/cache")
LETTERS = ["G", "H", "J", "N", "NN", "S", "Z"]
DISPLAY = {"NN": "Ñ"}
STATIC_L = sorted(STATIC)
T, K = 12, 5
N_TEMPL, N_REJ = 40, 10  # medoid templates per moving letter / per static letter (reject class)
MAX_FA, GATE = 0.10, 0.80
FEATURE_VERSION = "m1"
# variant -> [(feature, weight)]; features: shape (crop), shape_full (full frame), loc (full-frame wrist path)
VARIANTS = {
    "full": [("shape_full", 1.0), ("loc", 1.0)],
    "crop": [("shape", 1.0)],
    "crop_loc1": [("shape", 1.0), ("loc", 1.0)],
    "crop_loc3": [("shape", 1.0), ("loc", 3.0)],
    "crop_loc6": [("shape", 1.0), ("loc", 6.0)],
}
BANDS = (1, 2)  # 1 = 10% of 12 (the plan); 2 = a bit more warping


# ---------------------------------------------------------------- data
def read_ds(ds):
    d = os.path.join(OUT, ds)
    n = len(os.listdir(d))  # part count in the name: new clips -> new cache
    path = f"{CACHE}/{ds}_{n}.parquet"
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        pd.read_parquet(d).to_parquet(path)
    x = pd.read_parquet(path)
    x["label"] = x.label.astype(str)
    return x[x.label.isin(MOVING | STATIC)]


def clip_arrays(g):
    """One clip's rows -> (lm [6,21,3] NaN where no hand, share of Left labels, img_w, img_h, t_ms [6])."""
    lm = np.full((6, 21, 3), np.nan)
    h = g[(g.hand_idx == 0) & (g.frame < 6)]
    lm[h.frame.to_numpy(int)] = h[S.HAND_COLS].to_numpy(float).reshape(-1, 21, 3)
    left = float((h.handedness == "Left").mean()) if len(h) else 0.0
    return lm, left, float(g.img_w.iloc[0]), float(g.img_h.iloc[0]), np.arange(6) * 1000.0 / 6


def load_clips():
    """Clips that have shape (crop), shape_full and loc, so every variant is scored on the same clips."""
    cache = f"{FEATS}/moving_clips.npz"
    full, crop = read_ds("lsc70"), read_ds("lsc70h")
    crops = dict(tuple(crop.groupby("sample_id")))
    out = dict(shape=[], shape_full=[], loc=[], label=[], signer=[], sid=[], raw=[])
    miss = 0
    for sid, g in full.groupby("sample_id"):
        if sid not in crops:
            miss += 1
            continue
        flm, fleft, fw, fh, t = clip_arrays(g)
        clm, cleft, cw, ch, _ = clip_arrays(crops[sid])
        is_left = (cleft if not np.isnan(clm[:, 0, 0]).all() else fleft) > 0.5  # crop label is more reliable
        sh = S.moving_shape(clm, cw, ch, is_left, t, T)
        shf = S.moving_shape(flm, fw, fh, is_left, t, T)
        loc = S.moving_path(flm, fw, fh, is_left, t, T)
        if sh is None or shf is None or loc is None:
            miss += 1
            continue
        for k, v in (("shape", sh), ("shape_full", shf), ("loc", loc), ("label", g.label.iat[0]),
                     ("signer", g.signer.iat[0]), ("sid", sid),
                     ("raw", dict(crop=clm, crop_wh=(cw, ch), full=flm, full_wh=(fw, fh), is_left=bool(is_left), t_ms=t))):
            out[k].append(v)
    raw = out.pop("raw")
    C = {k: np.array(v) for k, v in out.items()}
    C["raw"] = raw
    print(f"clips: {len(C['label'])} kept, {miss} dropped (no crop or < {S.MOVING_MIN_FRAMES} frames with a hand); "
          f"moving {int(np.isin(C['label'], LETTERS).sum())}, static {int(np.isin(C['label'], STATIC_L).sum())}, "
          f"signers {len(set(C['signer']))}", flush=True)
    np.savez_compressed(cache, **{k: v for k, v in C.items() if k != "raw"})
    return C


def features(C, variant):
    return np.concatenate([w * C[f] for f, w in VARIANTS[variant]], 2).astype(np.float64)


# ---------------------------------------------------------------- matching
def templates(F, y, idx, rej, band):
    """Template indices and their class labels: N_TEMPL medoids per moving letter (+ N_REJ per static letter)."""
    ti, tl = [], []
    for c in LETTERS:
        i = idx[y[idx] == c]
        sel = i[D.medoids(F[i], N_TEMPL, band)]
        ti += list(sel)
        tl += [c] * len(sel)
    if rej:
        for c in STATIC_L:
            i = idx[y[idx] == c]
            sel = i[D.medoids(F[i], N_REJ, band)]
            ti += list(sel)
            tl += ["static"] * len(sel)
    return np.array(ti), np.array(tl)


def classes(rej):
    return LETTERS + (["static"] if rej else [])


def oof(F, y, g, rej, band):
    """Out-of-fold class scores [n, n_classes] with signers held out (GroupKFold 5)."""
    cl = classes(rej)
    Sc = np.full((len(F), len(cl)), np.inf)
    for tr, te in GroupKFold(5).split(F, y, g):
        ti, tl = templates(F, y, tr, rej, band)
        Sc[te] = D.class_scores(D.dist_matrix(F[te], F[ti], band), tl, cl, K)
    return Sc


def verify(Sc, y, rej, thr=None):
    """Per target letter: threshold (if not given), recall, false accepts. Pass = target wins and score <= thr."""
    pred = Sc.argmin(1)
    res, thr = {}, dict(thr or {})
    for i, c in enumerate(LETTERS):
        pos, neg = y == c, y != c
        s = Sc[:, i]
        win = pred == i
        if c not in thr:
            ok_pos = s[pos & win]
            cap = float(np.percentile(ok_pos, 95)) if len(ok_pos) else 0.0
            s_neg = np.sort(s[neg & win])
            m = int(np.floor(MAX_FA * neg.sum()))
            limit = float(np.nextafter(s_neg[m], -np.inf)) if m < len(s_neg) else np.inf
            thr[c] = min(cap, limit)
        acc = win & (s <= thr[c])
        per_static = {l: float(acc[y == l].mean()) for l in STATIC_L if (y == l).any()}
        worst = max(per_static, key=per_static.get)
        mv = Sc[:, :len(LETTERS)].argmin(1)
        res[c] = dict(n=int(pos.sum()), top1_moving=float((mv[pos] == i).mean()), recall=float(acc[pos].mean()),
                      fa_all=float(acc[neg].mean()), fa_static=float(acc[np.isin(y, STATIC_L)].mean()),
                      fa_moving=float(acc[neg & np.isin(y, LETTERS)].mean()),
                      worst_static=worst, worst_static_fa=per_static[worst],
                      gate3=bool(acc[pos].mean() >= GATE), threshold=round(float(thr[c]), 5))
    return res, thr


def evaluate():
    C = load_clips()
    y, g = C["label"], C["signer"]
    rep = {}
    for v in VARIANTS:
        F = features(C, v)
        for band in BANDS:
            for rej in (False, True):
                name = f"{v}_b{band}{'+rej' if rej else ''}"
                r, _ = verify(oof(F, y, g, rej, band), y, rej)
                rec = np.mean([r[c]["recall"] for c in LETTERS])
                rep[name] = dict(mean_recall=float(rec), n_pass=int(sum(r[c]["gate3"] for c in LETTERS)),
                                 mean_top1=float(np.mean([r[c]["top1_moving"] for c in LETTERS])), letters=r)
                print(f"{name:20s} recall {rec:.3f}  top1 {rep[name]['mean_top1']:.3f}  pass {rep[name]['n_pass']}/7  "
                      + " ".join(f"{DISPLAY.get(c, c)}={r[c]['recall']:.2f}" for c in LETTERS), flush=True)
    best = max(rep, key=lambda k: (rep[k]["n_pass"], rep[k]["mean_recall"]))
    print(f"\nbest: {best}")
    for c in LETTERS:
        r = rep[best]["letters"][c]
        print(f"  {DISPLAY.get(c, c):2s} n={r['n']:3d} top1={r['top1_moving']:.2f} recall={r['recall']:.2f} "
              f"FA all={r['fa_all']:.3f} static={r['fa_static']:.3f} moving={r['fa_moving']:.3f} "
              f"worst static {r['worst_static']}={r['worst_static_fa']:.2f} thr={r['threshold']} gate3={r['gate3']}")
    json.dump(dict(best=best, variants=rep), open(f"{FEATS}/moving_eval.json", "w"), indent=1)


# ---------------------------------------------------------------- comparison: small MLP (M9 rule: only if it beats DTW)
def _augment(X, rng, copies=4, rot=15, scale=0.1, jitter=0.02):
    """X [n,T,40] crop shape -> copies with one in-plane rotation/scale per clip and per-point jitter."""
    P = X.reshape(len(X), X.shape[1], 20, 2)
    out = [P]
    for _ in range(copies):
        a = np.deg2rad(rng.uniform(-rot, rot, len(P)))
        s = rng.uniform(1 - scale, 1 + scale, len(P))
        R = np.stack([np.cos(a), -np.sin(a), np.sin(a), np.cos(a)], 1).reshape(-1, 1, 1, 2, 2)
        Q = (R @ P[..., None])[..., 0] * s[:, None, None, None]
        out.append(Q + rng.normal(0, jitter, Q.shape))
    return np.concatenate(out).reshape(-1, X.shape[1] * 40)


def mlp_compare(hidden=(256, 128), seeds=3):
    """Out-of-fold MLP on the crop-shape sequence, classes = 7 moving letters + 'static'. Same verification
    metrics as DTW (score = -p(class)), so the numbers compare directly."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    C = load_clips()
    y, g = C["label"], C["signer"]
    yc = np.where(np.isin(y, LETTERS), y, "static")
    cl = classes(True)
    yi = np.array([cl.index(c) for c in yc])
    X = C["shape"]
    rep = {}
    for name, aug in (("mlp", False), ("mlp_aug", True)):
        P = np.zeros((len(X), len(cl)))
        for f, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
            rng = np.random.default_rng(f)
            Xtr = _augment(X[tr], rng) if aug else X[tr].reshape(len(tr), -1)
            ytr = np.tile(yi[tr], len(Xtr) // len(tr))
            sc = StandardScaler().fit(Xtr)
            for s in range(seeds):
                m = MLPClassifier(hidden, alpha=1e-3, max_iter=300, early_stopping=True, random_state=s)
                m.fit(sc.transform(Xtr), ytr)
                P[te] += m.predict_proba(sc.transform(X[te].reshape(len(te), -1))) / seeds
        r, _ = verify(-P, y, True)
        rec = np.mean([r[c]["recall"] for c in LETTERS])
        rep[name] = dict(mean_recall=float(rec), n_pass=int(sum(r[c]["gate3"] for c in LETTERS)), letters=r)
        print(f"{name:10s} recall {rec:.3f} pass {rep[name]['n_pass']}/7", flush=True)
        for c in LETTERS:
            q = r[c]
            print(f"  {DISPLAY.get(c, c):2s} top1={q['top1_moving']:.2f} recall={q['recall']:.2f} FA all={q['fa_all']:.3f} "
                  f"worst static {q['worst_static']}={q['worst_static_fa']:.2f}", flush=True)
    json.dump(rep, open(f"{FEATS}/moving_mlp_eval.json", "w"), indent=1)


# ---------------------------------------------------------------- export
def parse(name):
    """'crop_loc3_b1+rej' -> (variant, band, rej)."""
    rej = name.endswith("+rej")
    base = name[:-4] if rej else name
    v, _, b = base.rpartition("_b")
    return v, int(b), rej


def _nan_list(a):
    return [[None if np.isnan(x) else round(float(x), 6) for x in p] for p in np.asarray(a).reshape(-1, 3)]


def export(name, force_all=False):
    v, band, rej = parse(name)
    C = load_clips()
    y, g = C["label"], C["signer"]
    F = np.round(features(C, v), 3)  # the app stores 3 decimals
    res, thr = verify(oof(F, y, g, rej, band), y, rej)
    ship = [c for c in LETTERS if res[c]["gate3"] or force_all]
    os.makedirs(f"{EXPORT}/templates", exist_ok=True)
    os.makedirs(f"{EXPORT}/golden", exist_ok=True)
    ti, tl = templates(F, y, np.arange(len(F)), rej, band)
    common = dict(kind="moving_letter", feature_version=FEATURE_VERSION, variant=v, T=T, dim=int(F.shape[2]),
                  band=band, k=K, layout=[[f, w] for f, w in VARIANTS[v]], two_handed=False)
    for c in ship:
        sel = ti[tl == c]
        json.dump(dict(word=c, display=DISPLAY.get(c, c), threshold=thr[c], gate3=res[c]["gate3"], **common,
                       templates=[dict(source="LSC70", X=F[i].tolist()) for i in sel]),
                  open(f"{EXPORT}/templates/letter_{c}.json", "w"), separators=(",", ":"))
    if rej:
        sel = ti[tl == "static"]
        json.dump(dict(word="static", display="static", threshold=None, **common,
                       templates=[dict(source="LSC70", label=str(y[i]), X=F[i].tolist()) for i in sel]),
                  open(f"{EXPORT}/templates/letter_static_reject.json", "w"), separators=(",", ":"))
    # golden: raw landmarks -> features (Dart feature port) and class scores (Dart DTW port)
    rng = np.random.default_rng(0)
    q = [int(i) for c in LETTERS for i in rng.choice(np.where(y == c)[0], 2, replace=False)]
    q += [int(i) for i in rng.choice(np.where(np.isin(y, STATIC_L))[0], 4, replace=False)]
    cl = classes(rej)
    sc = D.class_scores(D.dist_matrix(F[q], F[ti], band), tl, cl, K)
    cases = []
    for r, i in enumerate(q):
        raw = C["raw"][i]
        p = int(sc[r].argmin())
        cases.append(dict(label=str(y[i]), crop=_nan_list(raw["crop"]), crop_wh=raw["crop_wh"],
                          full=_nan_list(raw["full"]), full_wh=raw["full_wh"], is_left=raw["is_left"],
                          t_ms=raw["t_ms"].tolist(),
                          features={f: np.round(C[f][i], 6).tolist() for f, _ in VARIANTS[v]},
                          X=F[i].tolist(), scores={c: round(float(sc[r, j]), 5) for j, c in enumerate(cl)},
                          predicted=cl[p], passed_as_target=bool(cl[p] == y[i] and y[i] in thr and sc[r, p] <= thr[y[i]])))
    json.dump(dict(feature_version=FEATURE_VERSION, variant=v, band=band, k=K, tolerance=1e-4, classes=cl,
                   note="templates are ALL exported letters (+ static reject); scores use every template, even of letters that failed Gate 3",
                   cases=cases), open(f"{EXPORT}/golden/moving_letters.json", "w"), separators=(",", ":"))
    report(name, res, ship)
    print("exported", name, "letters:", [DISPLAY.get(c, c) for c in ship])


def report(name, res, ship):
    ev = json.load(open(f"{FEATS}/moving_eval.json")) if os.path.exists(f"{FEATS}/moving_eval.json") else None
    L = ["# Moving letters: kNN-DTW report", "",
         f"Variant **{name}**: {VARIANTS[parse(name)[0]]}, T={T}, k={K}, {N_TEMPL} medoid templates per letter"
         + (f", plus {N_REJ} per static letter as a reject class" if parse(name)[2] else "") + ".",
         "Data: LSC70, 6 frames per clip (first second, 6 fps). Shape from the LSC70ANH hand crop, wrist path from the full frame.",
         f"All numbers are on held-out signers (GroupKFold 5). Threshold per letter: 95th percentile of correct scores, "
         f"lowered until <= {MAX_FA:.0%} of all non-target clips pass. Thresholds are fit on the same out-of-fold scores (slightly optimistic).",
         f"Gate 3: recall >= {GATE:.0%} **and** the contact sheet (`qc/sheets/moving_<X>.png`) shows the full motion in the first second (checked by eye).", ""]
    if ev:
        L += ["## Variants (mean recall over 7 letters / letters passing)", "", "| variant | recall | top-1 among moving | pass |", "|---|---|---|---|"]
        for k, r in sorted(ev["variants"].items(), key=lambda kv: -kv[1]["mean_recall"]):
            L.append(f"| {k} | {r['mean_recall']:.3f} | {r['mean_top1']:.3f} | {r['n_pass']}/7 |")
        L.append("")
    L += ["## Per letter", "", "| letter | n | top-1 | recall | FA all | FA static | FA moving | worst static (FA) | threshold | Gate 3 (numbers) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in LETTERS:
        r = res[c]
        L.append(f"| {DISPLAY.get(c, c)} | {r['n']} | {r['top1_moving']:.2f} | {r['recall']:.2f} | {r['fa_all']:.3f} | "
                 f"{r['fa_static']:.3f} | {r['fa_moving']:.3f} | {r['worst_static']} ({r['worst_static_fa']:.2f}) | {r['threshold']} | "
                 f"{'pass' if r['gate3'] else 'fail'} |")
    L += ["", f"Exported: {', '.join(DISPLAY.get(c, c) for c in ship) or 'none'}.", "",
          "## Caveats", "- LSC70 only covers the first second at 6 fps: long motions (e.g. Z, Ñ) may be cut off. "
          "The app must time-match this (1 s window, or team templates recorded with the app's window).",
          "- No team clips yet: this is the dataset number, not the new-user/new-camera number."]
    open(f"{EXPORT}/moving_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- contact sheets
def sheets():
    import cv2
    os.makedirs(f"{QC}/sheets", exist_ok=True)
    m = pd.read_csv(MANIFEST, dtype={"label": str})
    m = m[(m.dataset == "lsc70") & m.label.isin(MOVING) & m.sample_id.str.endswith("_0")]
    full = read_ds("lsc70")
    full = full[full.label.isin(MOVING)]
    fg = dict(tuple(full.groupby("sample_id")))
    for lab, gm in m.groupby("label"):
        rows = []
        for r in gm.sample(min(8, len(gm)), random_state=0).itertuples():
            try:
                tiles = [cv2.resize(rgb[60:-60, 60:-60] if rgb.shape[0] == 240 else rgb, (160, 160))
                         for _, _, rgb in _frames(os.path.join(RAW, r.path.replace("LSC70AN/", "LSC70ANH/")), "image", r.sample_id)]
                imgs = [rgb for _, _, rgb in _frames(os.path.join(RAW, r.path), "image", r.sample_id)]
                tiles += [np.zeros((160, 160, 3), np.uint8)] * (6 - len(tiles))
                path = np.zeros((160, 160, 3), np.uint8)
                g = fg.get(r.sample_id)
                if imgs and g is not None:
                    lm, _, w, h, _ = clip_arrays(g)
                    pts = lm[:, :, :2] * [w, h]
                    ok = ~np.isnan(pts[:, 0, 0])
                    if ok.any():
                        allp = pts[ok].reshape(-1, 2)
                        c, s = allp.mean(0), max(np.ptp(allp, 0).max() * 1.6, 60)
                        x0, y0 = int(c[0] - s / 2), int(c[1] - s / 2)
                        B = int(s) + 1
                        im = cv2.copyMakeBorder(imgs[0], B, B, B, B, cv2.BORDER_CONSTANT)
                        crop = im[y0 + B:y0 + B + int(s), x0 + B:x0 + B + int(s)].copy()
                        sc = 160 / s
                        path = cv2.resize(crop, (160, 160))
                        wp = [(int((p[0] - x0) * sc), int((p[1] - y0) * sc)) for p in pts[ok, 0]]
                        for a, b in zip(wp, wp[1:]):
                            cv2.line(path, a, b, (255, 255, 0), 2)
                        for k, p in enumerate(wp):
                            cv2.circle(path, p, 4, (255, 0, 0) if k == 0 else (0, 255, 0), -1)
                row = np.hstack(tiles[:6] + [path])
                cv2.putText(row, r.sample_id, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                rows.append(row)
            except Exception as e:  # a missing file must not stop the sheets
                print("skip", r.sample_id, e)
        if rows:
            out = f"{QC}/sheets/moving_{lab}.png"
            cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
            print("sheet", out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sheets":
        sheets()
    elif cmd == "eval":
        evaluate()
    elif cmd == "mlp":
        mlp_compare()
    elif cmd == "export":
        export(sys.argv[2], "--all" in sys.argv)
    else:
        raise SystemExit(__doc__)

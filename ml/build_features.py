"""Build letters.npz and sequences_{words,moving}.npz from the frames table + qc/clips.csv.

    python build_features.py

Env: SOMAN_OUT (frames dir), SOMAN_QC (qc dir), SOMAN_FEATS (output dir).
"""
import json
import os

import numpy as np
import pandas as pd

import normalize as N
import sequences as S
from extract import OUT, STATIC, MOVING

QC = os.environ.get("SOMAN_QC", os.path.join(os.path.dirname(OUT.rstrip("/")), "qc"))
FEATS = os.environ.get("SOMAN_FEATS", os.path.join(os.path.dirname(OUT.rstrip("/")), "features"))
T_WORDS, T_MOVING = 30, 12
LETTER_DS = os.environ.get("SOMAN_LETTER_DS", "lsc70h")  # hand crops; full frames give ~70% (hands ~40 px)


def build_letters(df, keep):
    d = df[(df.dataset == LETTER_DS) & df.label.isin(STATIC) & (df.hand_idx >= 0)]
    d = d.merge(keep, on=["dataset", "sample_id"])
    lm = d[S.HAND_COLS].to_numpy(float).reshape(-1, 21, 3)
    left = (d.handedness == "Left").to_numpy()
    X = np.stack([N.handshape(lm[i], d.img_w.iat[i], d.img_h.iat[i], left[i]) for i in range(len(d))])
    meta = d[["sample_id", "signer", "label", "frame"]].reset_index(drop=True)
    # frame filter: > 2x the median distance from the sample's own median handshape
    dist = np.zeros(len(d))
    for _, idx in meta.groupby("sample_id").indices.items():
        med = np.median(X[idx], 0)
        dist[idx] = np.linalg.norm(X[idx] - med, axis=1)
    medd = meta.assign(dist=dist).groupby("sample_id").dist.transform("median").to_numpy()
    ok = ~((medd > 0) & (dist > 2 * medd))
    # outliers: > 4 std from the letter's mean distance (flag only)
    flag = np.zeros(len(d), bool)
    for lab, idx in meta.groupby("label").indices.items():
        dd = np.linalg.norm(X[idx] - X[idx].mean(0), axis=1)
        flag[idx] = dd > dd.mean() + 4 * dd.std()
    print(f"letters: {len(d)} frames, {int((~ok).sum())} dropped by frame filter, {int(flag.sum())} flagged")
    np.savez_compressed(f"{FEATS}/letters.npz", X=X[ok], label=meta.label[ok].to_numpy(),
                        signer=meta.signer[ok].to_numpy(), sample_id=meta.sample_id[ok].to_numpy(),
                        frame=meta.frame[ok].to_numpy(), outlier=flag[ok],
                        source_tracker=np.full(ok.sum(), "hand_landmarker"))


def estimate_idle_thr(clips):
    """Wrist speed (palm units/s) of the first/last 10% of LSC50 MVP frames; threshold = 3x the median
    (the upper tail is hands entering/leaving the frame, not rest)."""
    sp = []
    for g in clips:
        out = _dominant_speed_edges(g)
        if out is not None:
            sp.append(out)
    sp = np.concatenate(sp)
    return float(3 * np.median(sp))


def _dominant_speed_edges(g):
    g = g.sort_values("frame")
    frames = np.array(sorted(g.frame.unique()))
    t = g.groupby("frame").t_ms.first().reindex(frames).to_numpy(float)
    w, h = float(g.img_w.iloc[0]), float(g.img_h.iloc[0])
    best = None
    for tk in S._tracks(g):
        f = S._fill(frames, t, {fr: v[0] for fr, v in tk["f"].items()})
        if f is None:
            continue
        sl, arr = f
        left = np.mean([v[1] for v in tk["f"].values()]) > 0.5
        wr, sc = N.wrist_px(S._smooth(arr), w, h, left)
        path = np.hypot(*np.diff(wr, axis=0).T).sum() / sc
        if best is None or path > best[0]:
            best = (path, wr, sc, t[sl])
    if best is None or len(best[1]) < 10:
        return None
    _, wr, sc, tt = best
    dt = np.maximum(np.diff(tt) / 1000.0, 1e-3)
    sp = np.hypot(*np.diff(wr, axis=0).T) / sc / dt
    k = max(1, len(sp) // 10)
    return np.r_[sp[:k], sp[-k:]]


def build_sequences(df, keep, name, mask, T, idle_thr):
    d = df.merge(keep, on=["dataset", "sample_id"])
    d = d[mask(d)]
    out = dict(X=[], X2=[], locB=[], label=[], signer=[], dataset=[], sample_id=[], dom=[], sec=[])
    for (ds, sid), g in d.groupby(["dataset", "sample_id"], sort=False):
        r = S.clip_to_sequence(g, T, idle_thr)
        if r is None:
            continue
        for k, v in (("X", r["X"]), ("X2", r["X2"]), ("locB", r["locB"]), ("label", g.label.iat[0]),
                     ("signer", g.signer.iat[0]), ("dataset", ds), ("sample_id", sid),
                     ("dom", r["dom_path"]), ("sec", r["second_path"])):
            out[k].append(v)
    dom, sec = np.array(out.pop("dom")), np.array(out.pop("sec"))
    labels = np.array(out["label"])
    two = []
    for lab in sorted(set(labels)):
        m = (labels == lab) & (dom > 0)
        if m.sum() and np.mean(sec[m] > 0.4 * dom[m]) > 0.5 and not lab.startswith("imp_"):
            two.append(lab)
    arrs = {k: np.array(v) for k, v in out.items()}
    np.savez_compressed(f"{FEATS}/sequences_{name}.npz", source_tracker=np.full(len(labels), "hand_landmarker"),
                        **arrs)
    json.dump([str(x) for x in two], open(f"{FEATS}/two_handed_{name}.json", "w"))
    print(f"sequences_{name}: {len(labels)} clips, X {arrs['X'].shape}, two_handed={two}")


def main():
    os.makedirs(FEATS, exist_ok=True)
    df = pd.read_parquet(OUT)
    df["has_hand"] = df.hand_idx >= 0
    keep = pd.read_csv(f"{QC}/clips.csv")
    keep = keep[keep.keep][["dataset", "sample_id"]]
    build_letters(df, keep)
    vid = df[df.dataset.isin(["lsc50", "lsc54"])].merge(keep, on=["dataset", "sample_id"])
    mvp50 = vid[(vid.dataset == "lsc50") & ~vid.label.str.startswith("imp_")]
    thr = estimate_idle_thr([g for _, g in mvp50.groupby("sample_id")])
    print(f"idle threshold (palm units/s): {thr:.3f}")
    json.dump(dict(idle_thr=thr), open(f"{FEATS}/idle_thr.json", "w"))
    build_sequences(df, keep, "words", lambda d: d.dataset.isin(["lsc50", "lsc54"]), T_WORDS, thr)
    build_sequences(df, keep, "moving", lambda d: (d.dataset == "lsc70") & d.label.isin(MOVING), T_MOVING, None)


if __name__ == "__main__":
    main()

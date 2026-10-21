"""Quality check on extracted landmarks.

    python qc.py rates      # detection tables + drop rules + Done-when counts -> OUT/../qc/
    python qc.py sheets     # contact sheets (12 random frames per label, landmarks drawn)

Env: SOMAN_RAW, SOMAN_OUT (frames dir), SOMAN_QC (qc dir), same defaults as extract.py.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

from extract import HAND_COLS, MANIFEST, OUT, RAW, STATIC, _frames, canon_label

QC = os.environ.get("SOMAN_QC", os.path.join(os.path.dirname(OUT.rstrip("/")), "qc"))
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def load():
    df = pd.read_parquet(OUT)
    df["has_hand"] = df.hand_idx >= 0
    return df


def clip_table(df):
    """One row per clip: frames, frames with a hand, detection span ratio, keep flag."""
    rows = []
    for (ds, sid), g in df.groupby(["dataset", "sample_id"], sort=False):
        fr = g.groupby("frame").has_hand.any()
        n, nh = len(fr), int(fr.sum())
        if ds == "lsc70":
            keep = nh >= 4
            span = nh / max(n, 1)
        else:
            det = fr[fr].index
            if len(det) == 0:
                keep, span = False, 0.0
            else:
                win = fr.loc[det.min():det.max()]
                span = float(win.mean())
                keep = span >= 0.6
        first = g.iloc[0]
        rows.append(dict(dataset=ds, sample_id=sid, signer=first.signer, label=first.label,
                         frames=n, hand_frames=nh, det_rate=nh / max(n, 1), span=span, keep=keep))
    return pd.DataFrame(rows)


def rates():
    os.makedirs(QC, exist_ok=True)
    df = load()
    ct = clip_table(df)
    ct.to_csv(f"{QC}/clips.csv", index=False)
    for by in ("dataset", "label", "signer"):
        t = ct.groupby(by).agg(clips=("keep", "size"), kept=("keep", "sum"),
                               det_rate=("det_rate", "mean")).round(3)
        t.to_csv(f"{QC}/rate_by_{by}.csv")
    print(ct.groupby("dataset").agg(clips=("keep", "size"), kept=("keep", "sum"),
                                    det_rate=("det_rate", "mean")).round(3))
    # Gate 1: LSC50 detection (clip-level mean of per-frame detection)
    g1 = ct[ct.dataset == "lsc50"].det_rate.mean()
    print(f"\nGate 1 (LSC50 detection, want >= 0.70): {g1:.3f}")
    # Gate 1b: static letters in LSC70
    l = ct[(ct.dataset == "lsc70") & ct.label.isin(STATIC)].groupby("label").det_rate.mean()
    bad = l[l < 0.75]
    print(f"Gate 1b (LSC70 static letters < 0.75): {dict(bad.round(3)) if len(bad) else 'none'}")
    # Done-when counts (kept clips only)
    k = ct[ct.keep]
    ls = df.merge(k[["dataset", "sample_id"]], on=["dataset", "sample_id"])
    ls = ls[(ls.dataset == "lsc70") & ls.label.isin(STATIC) & ls.has_hand]
    c = ls.groupby("label").agg(frames=("frame", "size"), signers=("signer", "nunique"))
    print("\nStatic letters (want >= 250 frames, >= 50 signers):")
    print(c.assign(ok=(c.frames >= 250) & (c.signers >= 50)).to_string())
    w = k[k.dataset != "lsc70"]
    w = w[~w.label.str.startswith("imp_")]
    mvp = w.groupby("label").agg(clips=("sample_id", "size"), signers=("signer", "nunique"))
    print("\nWords (want >= 15 signers, GRACIAS >= 9):")
    print(mvp.to_string())


def _draw(img, rows):
    h, w = img.shape[:2]
    for _, r in rows.iterrows():
        pts = [(int(r[f"x{i}"] * w), int(r[f"y{i}"] * h)) for i in range(21)]
        for a, b in CONN:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), max(1, w // 300))
        for p in pts:
            cv2.circle(img, p, max(2, w // 200), (255, 0, 0), -1)
    return img


def sheets():
    os.makedirs(f"{QC}/sheets", exist_ok=True)
    df = load()
    df = df[df.has_hand]
    m = pd.read_csv(MANIFEST)
    m["label"] = m.apply(canon_label, axis=1)
    paths = m.set_index(["dataset", "sample_id"])
    rng = np.random.default_rng(0)
    for (ds, lab), g in df.groupby(["dataset", "label"]):
        pick = g.sample(min(12, len(g)), random_state=0)
        tiles = []
        for _, r in pick.iterrows():
            try:
                p = paths.loc[(r.dataset, r.sample_id)]
                p = p.iloc[0] if isinstance(p, pd.DataFrame) else p
                path = os.path.join(RAW, p.path)
                for fi, _, rgb in _frames(path, p.kind, r.sample_id):
                    if fi == r.frame:
                        img = _draw(rgb.copy(), pick[(pick.sample_id == r.sample_id) & (pick.frame == fi)])
                        tiles.append(cv2.resize(img, (240, 180)))
                        break
            except Exception as e:  # a missing file must not stop the sheets
                print("skip", ds, r.sample_id, e)
        if not tiles:
            continue
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(f"{QC}/sheets/{ds}_{lab}.png", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print("sheets ->", f"{QC}/sheets")


if __name__ == "__main__":
    {"rates": rates, "sheets": sheets}[sys.argv[1] if len(sys.argv) > 1 else "rates"]()

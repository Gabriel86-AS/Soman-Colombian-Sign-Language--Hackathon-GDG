"""Turn our captures (DATASETS/custom_data/<name>_p<id>/) into frame tables like extract.py makes.

    python capture\\custom_to_frames.py OUT_DIR

Writes OUT_DIR/custom/<name>_p<id>.parquet (landmarks only, no images), one row per saved frame,
same columns as extract.py (dataset "custom"). Upload OUT_DIR/custom/ to Drive soman-ml/frames/custom/.
Uses the crop landmarks (240x240, the LSC70ANH look) when the crop pass found the hand, else the
full-frame ones (features are scale-free). One key press (a burst, k = 0..) = one sample.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(os.path.dirname(os.path.dirname(ML)), "DATASETS", "custom_data")
HAND_COLS = [f"{a}{i}" for i in range(21) for a in "xyz"]
POSE_COLS = ["nose_x", "nose_y", "lsh_x", "lsh_y", "rsh_x", "rsh_y"]
COLS = ["dataset", "sample_id", "signer", "label", "rep", "frame", "t_ms", "img_w", "img_h",
        "hand_idx", "handedness", "hand_score"] + HAND_COLS + POSE_COLS


def person_table(folder):
    person = os.path.basename(folder)
    metas = []
    for f in glob.glob(os.path.join(folder, "*", "*", "*.json")):
        m = json.load(open(f))
        m["k"] = int(os.path.basename(f)[:-5].rsplit("_", 1)[1])
        metas.append(m)
    metas.sort(key=lambda m: (m["set"], m["label"], m["time"]))
    recs, burst, prev = [], -1, None
    for m in metas:
        if m["k"] == 0 or prev is None or (m["set"], m["label"]) != prev:
            burst += 1
        prev = (m["set"], m["label"])
        h = m["crop"] or m["full"]
        r = dict(dataset="custom", sample_id=f"{person}_{m['label']}_{burst:04d}", signer=f"team:{person}",
                 label=m["label"], rep=burst, frame=m["k"], t_ms=m["k"] * 150, img_w=h["img_w"], img_h=h["img_h"],
                 hand_idx=0, handedness=h["handedness"], hand_score=h["handedness_score"],
                 src="crop" if m["crop"] else "full")
        for i, (x, y, z) in enumerate(h["landmarks"]):
            r[f"x{i}"], r[f"y{i}"], r[f"z{i}"] = x, y, z
        recs.append(r)
    d = pd.DataFrame(recs).reindex(columns=COLS + ["src"]).assign(**{c: np.nan for c in POSE_COLS})
    d[HAND_COLS] = d[HAND_COLS].astype(np.float32)  # MediaPipe outputs float32 anyway; halves the file
    return d


def main():
    out = os.path.join(sys.argv[1], "custom")
    os.makedirs(out, exist_ok=True)
    for folder in sorted(glob.glob(os.path.join(BASE, "*_p*"))):
        d = person_table(folder)
        p = os.path.join(out, os.path.basename(folder) + ".parquet")
        d.to_parquet(p, compression="zstd")
        print(f"{os.path.basename(folder)}: {len(d)} frames, {d.sample_id.nunique()} samples, "
              f"crop {int((d.src == 'crop').sum())}, labels {d.groupby('label').size().to_dict()} -> {p} "
              f"({os.path.getsize(p) // 1024} KB)")


if __name__ == "__main__":
    main()

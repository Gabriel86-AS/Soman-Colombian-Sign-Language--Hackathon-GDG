"""Extract HandLandmarker (and pose anchor) landmarks for every clip in manifest.csv.

    python extract.py                 # everything, in priority order
    python extract.py lsc70_static    # one group: lsc70_static lsc54_mvp lsc50_mvp lsc50_imp lsc70_moving rest

One parquet part per clip: OUT/<dataset>/<sample_id>.parquet. Existing parts are skipped (resumable).
Read all with pd.read_parquet(OUT).

Env: SOMAN_RAW (/content/raw), SOMAN_MANIFEST, SOMAN_OUT (frames dir), SOMAN_MODELS (/content/models).
"""
import os
import re
import sys
import unicodedata
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

RAW = os.environ.get("SOMAN_RAW", "/content/raw")
MANIFEST = os.environ.get("SOMAN_MANIFEST", "/content/drive/MyDrive/soman-ml/manifest.csv")
OUT = os.environ.get("SOMAN_OUT", "/content/drive/MyDrive/soman-ml/frames")
MODELS = os.environ.get("SOMAN_MODELS", "/content/models")

STATIC = set("ABCDEFIKLMOPQRTUVWXY")
MOVING = set("GHJNZ") | {"NN"}
LSC50_MVP = {"GRACIAS", "BUENOS_DIAS", "BUENAS_TARDES", "BUENAS_NOCHES", "BIEN", "HOLA", "ADIOS", "POR_FAVOR"}
# LSC-54 folder names (ascii, letters only, lowercase) -> canonical label. Chao == ADIOS.
LSC54_MVP = {"gracias": "GRACIAS", "buenosdias": "BUENOS_DIAS", "buenastardes": "BUENAS_TARDES",
             "buenasnoches": "BUENAS_NOCHES", "bien": "BIEN", "hola": "HOLA", "chao": "ADIOS",
             "adios": "ADIOS", "porfavor": "POR_FAVOR"}

GROUPS = ["lsc70_static", "lsc54_mvp", "lsc50_mvp", "lsc50_imp", "lsc70_moving", "rest"]
HAND_COLS = [f"{a}{i}" for i in range(21) for a in "xyz"]
POSE_COLS = ["nose_x", "nose_y", "lsh_x", "lsh_y", "rsh_x", "rsh_y"]


def _key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def canon_label(row):
    if row["dataset"] == "lsc54":
        return LSC54_MVP.get(_key(str(row["label"])), row["label"])
    return row["label"]


def group_of(row):
    ds, lab = row["dataset"], row["label"]
    if ds == "lsc70":
        if lab in STATIC:
            return "lsc70_static"
        if lab in MOVING:
            return "lsc70_moving"
        return "rest"
    if ds == "lsc54":
        return "lsc54_mvp" if lab in LSC54_MVP.values() else "rest"
    if ds == "lsc50":
        return "lsc50_mvp" if lab in LSC50_MVP else "lsc50_imp"
    return "rest"


def _hand_opts(is_image):
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    conf = 0.3 if is_image else 0.5
    return vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=f"{MODELS}/hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1 if is_image else 2,
        min_hand_detection_confidence=conf,
        min_tracking_confidence=0.5)


def _pose_opts():
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    return vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=f"{MODELS}/pose_landmarker_lite.task"),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)


def _frames(path, kind, sample_id):
    """Yield (frame_idx, t_ms, rgb). Images: the 6 frames of one sample are separate files."""
    if kind == "image":
        # sample_id = PerNN_<class>_<k>; the clip is k=0..5 in the same folder
        d = os.path.dirname(path)
        stem = re.sub(r"_\d+$", "", os.path.basename(path)[:-4])
        for k in range(6):
            p = os.path.join(d, f"{stem}_{k}.jpg")
            img = cv2.imread(p)
            if img is None:
                continue
            yield k, round(k * 1000 / 6), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield i, round(i * 1000 / fps), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()


def process(row):
    """Extract one clip. `row` is a dict from the manifest plus `pose` (bool)."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    out = os.path.join(OUT, row["dataset"], f"{row['sample_id']}.parquet")
    if os.path.exists(out):
        return "skip"
    is_image = row["kind"] == "image"
    hand = vision.HandLandmarker.create_from_options(_hand_opts(is_image))
    pose = vision.PoseLandmarker.create_from_options(_pose_opts()) if row["pose"] else None
    recs = []
    last_t = -1
    for fi, t, rgb in _frames(os.path.join(RAW, row["path"]), row["kind"], row["sample_id"]):
        t = max(t, last_t + 1)  # timestamps must strictly increase
        last_t = t
        h, w = rgb.shape[:2]
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        base = dict(dataset=row["dataset"], sample_id=row["sample_id"], signer=row["signer"],
                    label=row["label"], rep=row["rep"], frame=fi, t_ms=t, img_w=w, img_h=h)
        pv = {c: np.nan for c in POSE_COLS}
        if pose is not None:
            pr = pose.detect_for_video(mpi, t)
            if pr.pose_landmarks:
                lm = pr.pose_landmarks[0]
                pv = dict(nose_x=lm[0].x, nose_y=lm[0].y, lsh_x=lm[11].x, lsh_y=lm[11].y,
                          rsh_x=lm[12].x, rsh_y=lm[12].y)
        res = hand.detect_for_video(mpi, t)
        if not res.hand_landmarks:
            recs.append({**base, "hand_idx": -1, "handedness": None, "hand_score": np.nan, **pv})
            continue
        for hi, lms in enumerate(res.hand_landmarks):
            cat = res.handedness[hi][0]
            r = {**base, "hand_idx": hi, "handedness": cat.category_name,
                 "hand_score": cat.score, **pv}
            for i, p in enumerate(lms):
                r[f"x{i}"], r[f"y{i}"], r[f"z{i}"] = p.x, p.y, p.z
            recs.append(r)
    hand.close()
    if pose is not None:
        pose.close()
    cols = ["dataset", "sample_id", "signer", "label", "rep", "frame", "t_ms", "img_w", "img_h",
            "hand_idx", "handedness", "hand_score"] + HAND_COLS + POSE_COLS
    df = pd.DataFrame(recs).reindex(columns=cols)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out + ".part")
    os.replace(out + ".part", out)
    return "ok"


def load_rows(groups):
    m = pd.read_csv(MANIFEST)
    m["label"] = m.apply(canon_label, axis=1)
    m["group"] = m.apply(group_of, axis=1)
    # LSC70: one clip = 6 files; keep only k=0 as the clip entry point
    m = m[~((m.dataset == "lsc70") & ~m.sample_id.str.endswith("_0"))]
    m["pose"] = m.group.isin(["lsc54_mvp", "lsc50_mvp"])
    rows = []
    for g in groups:
        rows += m[m.group == g].to_dict("records")
    return rows


def main():
    groups = sys.argv[1:] or GROUPS
    for g in groups:
        rows = load_rows([g])
        print(f"{g}: {len(rows)} clips", flush=True)
        done = 0
        with Pool(2) as pool:
            for r in pool.imap_unordered(process, rows, chunksize=4):
                done += 1
                if done % 200 == 0:
                    print(f"  {g}: {done}/{len(rows)}", flush=True)
        print(f"{g}: done", flush=True)


if __name__ == "__main__":
    main()

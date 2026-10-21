"""Fetch datasets into RAW (default /content/raw) and build manifest.csv.

    python ml/fetch.py lsc50 lsc54 lsc70     # any subset
    python ml/fetch.py manifest              # (re)build manifest.csv from what is on disk

Manifest columns: dataset, sample_id, path, label, signer, rep, kind
`path` is relative to RAW. Signer ids are namespaced (lsc70:Per07, lsc50:p2, lsc54:23).
"""
import csv
import os
import re
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

from remote_zip import RemoteFile

RAW = os.environ.get("SOMAN_RAW", "/content/raw")
MANIFEST = os.environ.get("SOMAN_MANIFEST", "manifest.csv")
WORKERS = 8

LSC70_URL = ("https://data.mendeley.com/public-files/datasets/9ssyn8tff5/files/"
             "5a25e9cb-d67b-4590-bb90-e1784a90a578/file_downloaded")
LSC54_URL = ("https://download.scidb.cn/download?fileId=76fa12dbe08e6d7e08549d80447ab1bd"
             "&dataSetType=organization&fileName=Videos%20Courtesy.zip")
LSC50_URL = "https://ndownloader.figshare.com/files/50118645"  # VIDEOS.zip

# LSC50 word ids -> label. Everything else is an impostor (label "imp_<id>").
LSC50_MVP = {"0000": "GRACIAS", "0001": "BUENOS_DIAS", "0002": "BUENAS_TARDES",
             "0003": "BUENAS_NOCHES", "0024": "BIEN", "0030": "HOLA",
             "0031": "ADIOS", "0032": "POR_FAVOR"}

LSC50_RE = re.compile(r"VIDEOS/COLOR_BODY/(\d{4})_(\d{4})_(\d{4})\.avi$")


def _lsc50_wanted(names):
    """MVP words: all reps. Impostors: rep 0000 of each person."""
    out = []
    for n in names:
        m = LSC50_RE.search(n)
        if not m:
            continue
        word, person, rep = m.groups()
        if word in LSC50_MVP or rep == "0000":
            out.append((n, word, person, rep))
    return out


def fetch_lsc50():
    z = zipfile.ZipFile(RemoteFile(LSC50_URL))
    todo = _lsc50_wanted(z.namelist())
    dest = os.path.join(RAW, "lsc50")
    os.makedirs(dest, exist_ok=True)

    def one(item):
        name = item[0]
        out = os.path.join(dest, os.path.basename(name))
        if os.path.exists(out):
            return
        # fresh zip handle per file: threads must not share a seek position
        zz = zipfile.ZipFile(RemoteFile(LSC50_URL))
        with open(out + ".part", "wb") as f:
            f.write(zz.read(name))
        os.replace(out + ".part", out)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))
    print("lsc50:", len(todo), "videos")


def _curl(url, out):
    if os.path.exists(out):
        return
    subprocess.run(["curl", "-L", "-sS", "-A", "Mozilla/5.0", "-o", out + ".part", url], check=True)
    os.replace(out + ".part", out)


def fetch_lsc54():
    d = os.path.join(RAW, "lsc54")
    os.makedirs(d, exist_ok=True)
    zp = os.path.join(d, "Videos Courtesy.zip")
    _curl(LSC54_URL, zp)
    if not os.path.isdir(os.path.join(d, "Videos Courtesy")):
        zipfile.ZipFile(zp).extractall(d)
    print("lsc54 ready")


def fetch_lsc70():
    d = os.path.join(RAW, "lsc70")
    os.makedirs(d, exist_ok=True)
    zp = os.path.join(d, "LSC70.zip")
    _curl(LSC70_URL, zp)
    if not os.path.isdir(os.path.join(d, "LSC70")):
        zipfile.ZipFile(zp).extractall(d)
    print("lsc70 ready")


def _rows_lsc50():
    for fn in sorted(os.listdir(os.path.join(RAW, "lsc50"))):
        m = re.match(r"(\d{4})_(\d{4})_(\d{4})\.avi$", fn)
        if not m:
            continue
        word, person, rep = m.groups()
        label = LSC50_MVP.get(word, f"imp_{word}")
        yield ["lsc50", fn[:-4], f"lsc50/{fn}", label, f"lsc50:p{int(person)}", int(rep), "video"]


def _rows_lsc54():
    base = os.path.join(RAW, "lsc54", "Videos Courtesy")
    for signer in sorted(os.listdir(base)):
        for cat in sorted(os.listdir(os.path.join(base, signer))):
            for word in sorted(os.listdir(os.path.join(base, signer, cat))):
                for fn in sorted(os.listdir(os.path.join(base, signer, cat, word))):
                    if not fn.endswith(".avi"):
                        continue
                    rep = int(os.path.splitext(fn)[0])
                    rel = os.path.relpath(os.path.join(base, signer, cat, word, fn), RAW).replace("\\", "/")
                    yield ["lsc54", f"{signer}_{word}_{rep}", rel, word, f"lsc54:{signer}", rep, "video"]


def _rows_lsc70():
    # File-name pattern is confirmed after the first download; see ml/NOTES.md.
    pat = re.compile(r"^(?P<signer>Per\d+)[_-](?P<label>[^_\-]+)[_-](?P<rep>\d+)", re.I)
    base = os.path.join(RAW, "lsc70")
    for root, _, files in os.walk(base):
        if "LSC70AN" not in root.replace("\\", "/").split("/") and "LSC70AN" not in root:
            continue
        if "LSC70ANH" in root:
            continue
        for fn in sorted(files):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = pat.match(fn)
            if not m:
                continue
            rel = os.path.relpath(os.path.join(root, fn), RAW).replace("\\", "/")
            yield ["lsc70", os.path.splitext(fn)[0], rel, m["label"].upper(),
                   f"lsc70:{m['signer']}", int(m["rep"]), "image"]


def build_manifest():
    rows = []
    for ds, gen in (("lsc70", _rows_lsc70), ("lsc50", _rows_lsc50), ("lsc54", _rows_lsc54)):
        if os.path.isdir(os.path.join(RAW, ds)):
            rows += list(gen())
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sample_id", "path", "label", "signer", "rep", "kind"])
        w.writerows(rows)
    print("manifest:", len(rows), "rows ->", MANIFEST)


if __name__ == "__main__":
    for a in sys.argv[1:] or ["lsc50", "lsc54", "lsc70", "manifest"]:
        {"lsc50": fetch_lsc50, "lsc54": fetch_lsc54, "lsc70": fetch_lsc70,
         "manifest": build_manifest}[a]()

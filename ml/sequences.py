"""Clip -> fixed-length feature sequence (Stage F). Features come from normalize.py.

clip_to_sequence(g, T, ...) takes one clip's rows from the frames table (all hands, all frames).
Returns dict(X [T,42], X2 [T,42], locB/locN/locB2/locN2 [T,2], dom_path, second_path) or None if unusable.
"""
import numpy as np

import normalize as N

HAND_COLS = [f"{a}{i}" for i in range(21) for a in "xyz"]
MAX_GAP = 4
SMOOTH = 3
MATCH_DIST = 0.25  # normalized image units for wrist tracking


def _tracks(g):
    """Group hands into <= 2 tracks by nearest wrist (not by hand_idx).
    Returns list of dict(frames->(lm[21,3], is_left)), over the clip's frame numbers."""
    tracks = []
    for fr, rows in g[g.hand_idx >= 0].groupby("frame", sort=True):
        free = list(range(len(rows)))
        vals = rows[HAND_COLS].to_numpy(float).reshape(-1, 21, 3)
        left = (rows.handedness == "Left").to_numpy()
        # greedy: closest (track, hand) pairs first
        pairs = sorted(
            ((np.hypot(*(vals[h, 0, :2] - t["last"])), ti, h) for ti, t in enumerate(tracks) for h in free),
            key=lambda p: p[0])
        used_t = set()
        for d, ti, h in pairs:
            if d > MATCH_DIST or ti in used_t or h not in free:
                continue
            tracks[ti]["f"][fr] = (vals[h], left[h])
            tracks[ti]["last"] = vals[h, 0, :2]
            used_t.add(ti)
            free.remove(h)
        for h in free:
            tracks.append(dict(f={fr: (vals[h], left[h])}, last=vals[h, 0, :2]))
    return tracks


def _fill(frames, t_all, lm_by_f):
    """Dense [n,21,3] over frames (NaN where missing); interpolate gaps <= MAX_GAP,
    keep the longest valid run otherwise. Returns (idx_range, arr) or None."""
    n = len(frames)
    arr = np.full((n, 21, 3), np.nan)
    for i, fr in enumerate(frames):
        if fr in lm_by_f:
            arr[i] = lm_by_f[fr]
    ok = ~np.isnan(arr[:, 0, 0])
    if ok.sum() < 2:
        return None
    # interpolate short gaps
    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        if i > 0 and j < n and j - i <= MAX_GAP:
            for k in range(i, j):
                w = (k - i + 1) / (j - i + 1)
                arr[k] = (1 - w) * arr[i - 1] + w * arr[j]
            ok[i:j] = True
        i = j
    # longest valid run
    best, cur_s, cur_l, bs, bl = 0, 0, 0, 0, 0
    for k in range(n + 1):
        if k < n and ok[k]:
            if cur_l == 0:
                cur_s = k
            cur_l += 1
        else:
            if cur_l > bl:
                bs, bl = cur_s, cur_l
            cur_l = 0
    if bl < 2:
        return None
    return slice(bs, bs + bl), arr[bs:bs + bl]


def _smooth(a, k=SMOOTH):
    """Moving average over k frames, edge frames use the available neighbours."""
    if len(a) < k:
        return a
    out = np.empty_like(a)
    h = k // 2
    for i in range(len(a)):
        out[i] = a[max(0, i - h):i + h + 1].mean(0)
    return out


def _trim_idle(wrist, scale, t_ms, thr):
    """Keep frames from the first to the last with wrist speed > thr (palm units / s)."""
    if thr is None or len(wrist) < 3:
        return slice(0, len(wrist))
    dt = np.maximum(np.diff(t_ms) / 1000.0, 1e-3)
    sp = np.hypot(*np.diff(wrist, axis=0).T) / scale / dt
    sp = np.r_[sp[0], sp]
    mv = np.where(sp > thr)[0]
    if len(mv) == 0:
        return slice(0, len(wrist))
    return slice(max(mv[0] - 1, 0), min(mv[-1] + 2, len(wrist)))


def _resample(a, t, T):
    tt = np.linspace(t[0], t[-1], T)
    flat = a.reshape(len(a), -1)
    out = np.stack([np.interp(tt, t, flat[:, d]) for d in range(flat.shape[1])], 1)
    return out.reshape((T,) + a.shape[1:])


def _hand_features(arr, t, w, h, is_left, nose_sh, T, idle_thr):
    """arr [n,21,3] (already filled) -> dict with X [T,42], locB [T,2], path (palm units)."""
    arr = _smooth(arr)
    wrist, scale = N.wrist_px(arr, w, h, is_left)
    sl = _trim_idle(wrist, scale, t, idle_thr)
    arr, t, wrist = arr[sl], t[sl], wrist[sl]
    if len(arr) < 2:
        return None
    shape = N.handshape_seq(arr, w, h, is_left, scale)
    locA = N.location_a(wrist, scale)
    if nose_sh is not None:
        nose, lsh, rsh = nose_sh
        if is_left:
            nose, lsh, rsh = nose * [-1, 1], lsh * [-1, 1], rsh * [-1, 1]
        locB = N.location_b(wrist, scale, None, lsh, rsh)
        width = np.sqrt(((lsh - rsh) ** 2).sum())
        locN = (wrist - nose) / max(width, 1e-9)
    else:
        locB = locN = np.full((len(arr), 2), np.nan)
    path = float(np.hypot(*np.diff(wrist, axis=0).T).sum() / scale)
    return dict(X=_resample(np.hstack([shape, locA]), t, T), locB=_resample(locB, t, T),
                locN=_resample(locN, t, T), path=path)


# ---- moving letters (feature version m1) ---------------------------------------------------------
# Shape from the hand crop (like static letters), wrist path from the full frame. No smoothing, no idle trim.
MOVING_MIN_FRAMES = 4


def _fill_nan(a, t):
    """a [n,...] with NaN rows (no hand) -> filled by linear interpolation in time, edges held.
    None if fewer than MOVING_MIN_FRAMES rows have a hand."""
    flat = np.asarray(a, np.float64).reshape(len(a), -1)
    ok = ~np.isnan(flat).any(1)
    if ok.sum() < MOVING_MIN_FRAMES:
        return None
    out = np.stack([np.interp(t, t[ok], flat[ok, d]) for d in range(flat.shape[1])], 1)
    return out.reshape(np.shape(a))


def moving_shape(lm, img_w, img_h, is_left, t_ms, T=12):
    """lm [n,21,2+] from the hand crop (NaN rows = no hand) -> [T,40] handshape over time, or None."""
    t = np.asarray(t_ms, np.float64)
    a = _fill_nan(np.asarray(lm, np.float64)[..., :2], t)
    if a is None:
        return None
    return _resample(N.handshape_seq(a, img_w, img_h, is_left), t, T)


def moving_path(lm, img_w, img_h, is_left, t_ms, T=12):
    """lm [n,21,2+] from the full frame (NaN rows = no hand) -> [T,2] wrist path (variant A), or None."""
    t = np.asarray(t_ms, np.float64)
    a = _fill_nan(np.asarray(lm, np.float64)[..., :2], t)
    if a is None:
        return None
    wrist, scale = N.wrist_px(a, img_w, img_h, is_left)
    return _resample(N.location_a(wrist, scale), t, T)


def clip_to_sequence(g, T, idle_thr=None):
    g = g.sort_values("frame")
    frames = np.array(sorted(g.frame.unique()))
    t_ms = g.groupby("frame").t_ms.first().reindex(frames).to_numpy(float)
    w, h = float(g.img_w.iloc[0]), float(g.img_h.iloc[0])
    tr = _tracks(g)
    feats = []
    for tk in tr:
        filled = _fill(frames, t_ms, {fr: v[0] for fr, v in tk["f"].items()})
        if filled is None:
            continue
        sl, arr = filled
        lefts = [tk["f"][fr][1] for fr in tk["f"]]
        is_left = np.mean(lefts) > 0.5  # majority label over the clip
        sh = None
        p = g[["nose_x", "nose_y", "lsh_x", "lsh_y", "rsh_x", "rsh_y"]].median()
        if not p.isna().any():
            sh = (np.array([p.nose_x * w, p.nose_y * h]), np.array([p.lsh_x * w, p.lsh_y * h]),
                  np.array([p.rsh_x * w, p.rsh_y * h]))
        f = _hand_features(arr, t_ms[sl], w, h, is_left, sh, T, idle_thr)
        if f is not None:
            feats.append(f)
    if not feats:
        return None
    feats.sort(key=lambda f: -f["path"])
    d = feats[0]
    s = feats[1] if len(feats) > 1 else None
    z2 = np.zeros_like(d["locB"])
    return dict(X=d["X"], locB=d["locB"], locN=d["locN"], dom_path=d["path"],
                X2=s["X"] if s else np.zeros_like(d["X"]), locB2=s["locB"] if s else z2,
                locN2=s["locN"] if s else z2, second_path=s["path"] if s else 0.0)

"""Feature contract: the single source of truth. The Dart port must match golden vectors to 1e-4.

Steps (per hand): pixel units -> mirror to right-hand frame -> wrist origin -> scale by |lm0 - lm9|.
No rotation. Handshape = x,y of landmarks 1..20 (40 dims). Location adds 2 dims (variant A or B).
Plain arithmetic only, so the Dart port stays simple.
"""
import numpy as np

HANDSHAPE_DIMS = 40


def to_pixels(lm, img_w, img_h):
    """lm [...,21,2+] normalized image coords -> pixel x,y [...,21,2]."""
    lm = np.asarray(lm, dtype=np.float64)
    out = lm[..., :2].copy()
    out[..., 0] *= img_w
    out[..., 1] *= img_h
    return out


def mirror_if_left(xy, is_left):
    """Mirror x so a left hand looks like a right hand. xy [...,21,2]."""
    xy = np.array(xy, dtype=np.float64)
    if is_left:
        xy[..., 0] = -xy[..., 0]
    return xy


def palm_size(xy):
    """|lm0 - lm9| for xy [...,21,2] (one scalar per frame)."""
    d = xy[..., 0, :] - xy[..., 9, :]
    return np.sqrt((d ** 2).sum(-1))


def handshape(lm, img_w, img_h, is_left):
    """One frame: lm [21,2+] -> 40 dims (x,y of landmarks 1..20, wrist origin, palm units)."""
    xy = mirror_if_left(to_pixels(lm, img_w, img_h), is_left)
    xy = xy - xy[0]
    s = palm_size(xy)
    return (xy[1:] / s).reshape(-1)


def handshape_seq(lms, img_w, img_h, is_left, scale=None):
    """A clip: lms [T,21,2+] -> [T,40]. `scale` defaults to the clip's mean palm size."""
    xy = mirror_if_left(to_pixels(lms, img_w, img_h), is_left)
    xy = xy - xy[:, :1, :]
    s = palm_size(xy).mean() if scale is None else scale
    return (xy[:, 1:] / s).reshape(len(xy), -1)


def wrist_px(lms, img_w, img_h, is_left):
    """Wrist path in mirrored pixel units [T,2] and the clip's mean palm size."""
    xy = mirror_if_left(to_pixels(lms, img_w, img_h), is_left)
    return xy[:, 0, :], palm_size(xy - xy[:, :1, :]).mean()


def location_a(wrist, scale):
    """Variant A: wrist relative to the first kept frame, in palm units. wrist [T,2] (mirrored px)."""
    return (wrist - wrist[0]) / scale


def location_b(wrist, scale, nose, lsh, rsh):
    """Variant B: wrist relative to shoulder midpoint over shoulder width.

    wrist is mirrored pixels, so it is un-mirrored by the caller only if needed; here the shoulders
    are given in the SAME (mirrored) pixel frame. wrist [T,2]; nose/lsh/rsh [2] or [T,2] pixels.
    """
    mid = (np.asarray(lsh) + np.asarray(rsh)) / 2.0
    width = np.sqrt(((np.asarray(lsh) - np.asarray(rsh)) ** 2).sum(-1))
    return (wrist - mid) / np.maximum(width, 1e-9)[..., None]

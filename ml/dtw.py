"""Banded DTW, the reference the Dart port must match (golden/words.json).

dtw(a, b, band): a [n,D], b [m,D]. Local cost = Euclidean distance between frames. Steps: diag, up (i-1,j),
left (i,j-1); ties prefer diag, then up. Sakoe-Chiba band |i-j| <= band (n == m == T in the app).
Result = accumulated cost / number of cells on the chosen path.
"""
import numpy as np

try:
    from numba import njit
except ImportError:  # slow but identical
    def njit(f=None, **_):
        return f if f is not None else (lambda g: g)

INF = 1e30


@njit(cache=True)
def dtw(a, b, band):
    n, m = a.shape[0], b.shape[0]
    D = np.full((n, m), INF)
    L = np.zeros((n, m))
    for i in range(n):
        for j in range(max(0, i - band), min(m, i + band + 1)):
            c = 0.0
            for k in range(a.shape[1]):
                c += (a[i, k] - b[j, k]) ** 2
            c = np.sqrt(c)
            if i == 0 and j == 0:
                D[i, j], L[i, j] = c, 1.0
                continue
            best, bl = INF, 0.0
            if i > 0 and j > 0 and D[i - 1, j - 1] < best:
                best, bl = D[i - 1, j - 1], L[i - 1, j - 1]
            if i > 0 and D[i - 1, j] < best:
                best, bl = D[i - 1, j], L[i - 1, j]
            if j > 0 and D[i, j - 1] < best:
                best, bl = D[i, j - 1], L[i, j - 1]
            D[i, j], L[i, j] = best + c, bl + 1.0
    return D[n - 1, m - 1] / L[n - 1, m - 1]


@njit(cache=True)
def dist_matrix(Q, Tm, band):
    """Q [nq,T,D], Tm [nt,T,D] -> [nq,nt]."""
    out = np.zeros((Q.shape[0], Tm.shape[0]))
    for i in range(Q.shape[0]):
        for j in range(Tm.shape[0]):
            out[i, j] = dtw(Q[i], Tm[j], band)
    return out


def class_scores(dm, tlabels, classes, k=5):
    """dm [nq,nt] -> [nq,len(classes)]: mean of the k smallest distances to each class's templates."""
    out = np.full((dm.shape[0], len(classes)), np.inf)
    for ci, c in enumerate(classes):
        d = np.sort(dm[:, tlabels == c], axis=1)[:, :k]
        if d.shape[1]:
            out[:, ci] = d.mean(1)
    return out

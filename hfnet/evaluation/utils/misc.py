import numpy as np


def to_homogeneous(points):
    return np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=-1)


def from_homogeneous(points):
    return points[:, :-1] / points[:, -1:]


def angle_error(R1, R2):
    cos = (np.trace(np.dot(np.linalg.inv(R1), R2)) - 1) / 2
    return np.rad2deg(np.abs(np.arccos(cos)))


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if np.isscalar(c):
            c = c if np.isfinite(c) else (1 if a == 0 else 0)
        else:
            idx = ~np.isfinite(c)
            c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c

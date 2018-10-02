import numpy as np
import cv2


def sample_bilinear(data, points):
    # Pad the input data with zeros
    data = np.lib.pad(
        data, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=0)
    points = np.asarray(points) + 1

    x, y = points.T
    x0, y0 = points.T.astype(int)
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, data.shape[1]-1)
    x1 = np.clip(x1, 0, data.shape[1]-1)
    y0 = np.clip(y0, 0, data.shape[0]-1)
    y1 = np.clip(y1, 0, data.shape[0]-1)

    Ia = data[y0, x0]
    Ib = data[y1, x0]
    Ic = data[y0, x1]
    Id = data[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


def sample_descriptors(descriptor_map, keypoints, image_size):
    factor = np.array(descriptor_map.shape[:-1]) / np.array(image_size)
    desc = sample_bilinear(descriptor_map, keypoints*factor[::-1])
    desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)
    assert np.all(np.isfinite(desc))
    return desc


def matching(desc1, desc2, do_ratio_test=False):
    desc1, desc2 = np.float32(desc1), np.float32(desc2)
    if do_ratio_test:
        matches = []
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        for m, n in matcher.knnMatch(desc1, desc2, k=2):
            m.distance = m.distance / n.distance
            matches.append(m)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.match(desc1, desc2)
    return matches_cv2np(matches)


def keypoints_cv2np(kpts_cv):
    kpts_np = np.array([k.pt for k in kpts_cv])
    scores = np.array([k.response for k in kpts_cv])
    return kpts_np, scores


def matches_cv2np(matches_cv):
    matches_np = np.int32([[m.queryIdx, m.trainIdx] for m in matches_cv])
    distances = np.float32([m.distance for m in matches_cv])
    return matches_np.reshape(-1, 2), distances


def to_homogeneous(points):
    return np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=-1)


def from_homogeneous(points):
    return points[:, :2] / points[:, 2:]


def keypoints_warp_2D(kpts, H, shape):
    kpts_w = from_homogeneous(np.dot(to_homogeneous(kpts), np.transpose(H)))
    visibility = np.all(
        (kpts_w >= 0) & (kpts_w <= (np.array(shape)[::-1]-1)), axis=-1)
    return kpts_w, visibility


def keypoints_filter_borders(kpts, shape, border):
    good = np.all(np.logical_and(
        kpts < (np.asarray(shape[::-1]) - border),
        kpts >= border), -1)
    return kpts[good]


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c

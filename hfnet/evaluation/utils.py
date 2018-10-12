import numpy as np
import cv2


def nms_fast(kpts, scores, shape, dist_thresh=4):
    grid = np.zeros(shape, dtype=int)
    inds = np.zeros(shape, dtype=int)

    inds1 = np.argsort(-scores)  # Sort by confidence
    kpts_sorted = kpts[inds1]
    kpts_sorted = kpts_sorted.round().astype(int)  # Rounded corners.

    # Check for edge case of 0 or 1 corners.
    if kpts_sorted.shape[0] == 0:
        return np.zeros(0, dtype=int)
    if kpts_sorted.shape[0] == 1:
        return np.zeros((1), dtype=int)

    grid[kpts_sorted[:, 1], kpts_sorted[:, 0]] = 1
    inds[kpts_sorted[:, 1], kpts_sorted[:, 0]] = np.arange(len(kpts_sorted))
    pad = dist_thresh
    grid = np.pad(grid, [[pad]*2]*2, mode='constant')

    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, k in enumerate(kpts_sorted):
        pt = (k[0]+pad, k[1]+pad)
        if grid[pt[1], pt[0]] == 1:
            grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1

    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    scores_keep = scores[inds1][inds_keep]
    inds2 = np.argsort(-scores_keep)
    out_inds = inds1[inds_keep[inds2]]
    return out_inds


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
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
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
    return points[:, :-1] / points[:, -1:]


def keypoints_warp_2D(kpts, H, shape):
    kpts_w = from_homogeneous(np.dot(to_homogeneous(kpts), np.transpose(H)))
    visibility = np.all(
        (kpts_w >= 0) & (kpts_w <= (np.array(shape)[::-1]-1)), axis=-1)
    return kpts_w, visibility


def keypoints_warp_3D(kpts1, depth1, K1, K2, T_1to2, shape2):
    kpts1_int = np.round(kpts1).astype(int)
    depth1_kpts = depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
    kpts1_3d_1 = np.dot(to_homogeneous(kpts1), np.linalg.inv(K1).T)
    kpts1_3d_1 = depth1_kpts[:, np.newaxis]*kpts1_3d_1
    kpts1_3d_2 = from_homogeneous(np.dot(to_homogeneous(kpts1_3d_1), T_1to2.T))
    kpts1_w = from_homogeneous(np.dot(kpts1_3d_2, K2.T))

    visibility = np.all(
        (kpts1_w >= 0) & (kpts1_w <= (np.array(shape2)[::-1]-1)), axis=-1)
    visibility = visibility & (depth1_kpts > 0)
    return kpts1_w, visibility


def keypoints_filter_borders(kpts, shape, border):
    good = np.all(np.logical_and(
        kpts < (np.asarray(shape[::-1]) - border),
        kpts >= border), -1)
    return good


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if np.isscalar(c):
            c = c if np.isfinite(c) else (1 if a == 0 else 0)
        else:
            idx = ~np.isfinite(c)
            c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c

import numpy as np

from .misc import from_homogeneous, to_homogeneous


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
    for i, k in enumerate(kpts_sorted):
        pt = (k[0]+pad, k[1]+pad)
        if grid[pt[1], pt[0]] == 1:
            grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
            grid[pt[1], pt[0]] = -1

    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    scores_keep = scores[inds1][inds_keep]

    inds2 = np.argsort(-scores_keep)
    out_inds = inds1[inds_keep[inds2]]
    return out_inds


def keypoints_cv2np(kpts_cv):
    kpts_np = np.array([k.pt for k in kpts_cv])
    scores = np.array([k.response for k in kpts_cv])
    return kpts_np, scores


def keypoints_filter_borders(kpts, shape, border):
    good = np.all(np.logical_and(
        kpts < (np.asarray(shape[::-1]) - border),
        kpts >= border), -1)
    return good


def keypoints_warp_2D(kpts, H, shape):
    kpts_w = from_homogeneous(np.dot(to_homogeneous(kpts), np.transpose(H)))
    vis = np.all((kpts_w >= 0) & (kpts_w <= (np.array(shape)-1)), axis=-1)
    return kpts_w, vis


def keypoints_warp_3D(kpts1, depth1, K1, K2, T_1to2, shape2,
                      consistency_check=False, depth2=None, thresh=0.1):
    kpts1_int = np.round(kpts1).astype(int)
    depth1_kpts = depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
    kpts1_3d_1 = np.dot(to_homogeneous(kpts1), np.linalg.inv(K1).T)
    kpts1_3d_1 = depth1_kpts[:, np.newaxis]*kpts1_3d_1
    kpts1_3d_2 = from_homogeneous(np.dot(to_homogeneous(kpts1_3d_1), T_1to2.T))
    kpts1_w = from_homogeneous(np.dot(kpts1_3d_2, K2.T))

    vis = np.all((kpts1_w >= 0) & (kpts1_w <= (np.array(shape2)-1)), axis=-1)
    vis &= (depth1_kpts > 0)  # visible in SfM
    vis &= (kpts1_3d_2[:, -1] > 0)  # point in front of the camera

    if consistency_check:
        assert depth2 is not None
        kpts1_w_int = np.round(kpts1_w[vis]).astype(int)
        depth2_kpts = depth2[kpts1_w_int[:, 1], kpts1_w_int[:, 0]]
        kpt1_w_z = kpts1_3d_2[vis, -1]
        # Consistency of the two depth values for each point
        error = np.abs(kpt1_w_z - depth2_kpts) / np.maximum(depth2_kpts, 1e-4)
        vis[vis] &= (error < thresh) & (depth2_kpts > 0)

    return kpts1_w, vis, kpts1_3d_1

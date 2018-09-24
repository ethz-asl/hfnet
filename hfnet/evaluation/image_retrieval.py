from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import numpy as np


def normalize(l, axis=-1):
    return np.array(l) / np.linalg.norm(l, axis=axis, keepdims=True)


def is_gt_match_3D(query_poses, ref_poses, distance_thresh, angle_thresh):
    distances = np.linalg.norm(np.expand_dims(query_poses['pos'], axis=1)
                               - np.expand_dims(ref_poses['pos'], axis=0), axis=-1)
    angle_errors = np.arccos(
        (np.trace(
            np.matmul(np.expand_dims(np.linalg.inv(query_poses['rot']), axis=1),
                      np.expand_dims(ref_poses['rot'], axis=0)),
            axis1=2, axis2=3) - 1)/2)
    return np.logical_and(distances < distance_thresh, angle_errors < angle_thresh)


def is_gt_match_2D(query_poses, ref_poses, distance_thresh, angle_thresh):
    distances = np.linalg.norm(
            np.expand_dims([query_poses['x'], query_poses['y']], axis=2)
            - np.expand_dims([ref_poses['x'], ref_poses['y']], axis=1), axis=0)
    angle_errors = np.abs(np.mod(np.expand_dims(query_poses['angle'], axis=1)
                                 - np.expand_dims(ref_poses['angle'], axis=0) + np.pi,
                                 2*np.pi) - np.pi)  # bring it in [-pi,+pi]
    return np.logical_and(distances < distance_thresh, angle_errors < angle_thresh)


def retrieval(ref_descriptors, query_descriptors, max_num_nn, pca_dim=0):
    if pca_dim != 0:
        pca = PCA(n_components=pca_dim)
        ref_descriptors = normalize(pca.fit_transform(normalize(ref_descriptors)))
        query_descriptors = normalize(pca.transform(normalize(query_descriptors)))

    ref_tree = cKDTree(ref_descriptors)
    _, indices = ref_tree.query(query_descriptors, k=max_num_nn)
    return indices


def compute_tp_fp(ref_descriptors, query_descriptors,
                  gt_matches, *arg, **kwarg):
    indices = retrieval(ref_descriptors, query_descriptors, *arg, **kwarg)
    tp = gt_matches[np.expand_dims(np.arange(len(indices)), axis=1), indices]
    fp = np.logical_not(tp)
    tp_cum = np.cumsum(tp, axis=1)
    fp_cum = np.cumsum(fp, axis=1)
    valid = np.any(gt_matches, axis=1)
    return tp_cum, fp_cum, valid


def compute_recall(*arg, **kwarg):
    tp, fp, valid = compute_tp_fp(*arg, **kwarg)
    return np.mean(tp[valid] > 0, axis=0)

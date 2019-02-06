import numpy as np
import cv2
from sklearn.decomposition import PCA
from collections import namedtuple

from .descriptors import (
    normalize, root_descriptors, fast_matching, matches_cv2np)
from .db_management import LocalDbItem
from hfnet.utils.tools import Timer


LocResult = namedtuple(
    'LocResult', ['success', 'num_inliers', 'inlier_ratio', 'T'])
loc_failure = LocResult(False, 0, 0, None)


def preprocess_globaldb(global_descriptors, config):
    global_descriptors = normalize(global_descriptors)
    transf = [lambda x: normalize(x)]  # noqa: E731
    if config.get('pca_dim', 0) > 0:
        pca = PCA(n_components=config['pca_dim'], svd_solver='full')
        global_descriptors = normalize(pca.fit_transform(global_descriptors))
        transf.append(lambda x: normalize(pca.transform(x)))  # noqa: E731

    def f(x):
        for t in transf:
            x = t(x)
        return x

    return global_descriptors, f


def preprocess_localdb(local_db, config):
    if config.get('root', False):
        for frame_id in local_db:
            item = local_db[frame_id]
            desc = root_descriptors(item.descriptors)
            local_db[frame_id] = LocalDbItem(
                item.landmark_ids, desc, item.keypoints)
        transf = root_descriptors
    else:
        transf = lambda x: x  # noqa: E731
    return local_db, transf


def covis_clustering(frame_ids, local_db, points):
    components = dict()
    visited = set()
    count_components = 0

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        current_component = count_components
        components[current_component] = []
        count_components += 1
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited |= {exploration_frame}
            components[current_component].append(exploration_frame)

            landmarks = local_db[exploration_frame].landmark_ids
            connected_frames = set(i for lm in landmarks
                                   for i in points[lm].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clustered_frames = sorted(components.values(), key=len, reverse=True)
    return clustered_frames


def match_against_place(frame_ids, local_db, query_desc, ratio_thresh,
                        do_fast_matching=True, debug_dict=None):
    place_db = [local_db[frame_id] for frame_id in frame_ids]
    place_lms = np.concatenate([db.landmark_ids for db in place_db])
    place_desc = np.concatenate([db.descriptors for db in place_db])

    duration = 0
    if len(query_desc) > 0 and len(place_desc) > 1:
        query_desc = query_desc.astype(np.float32, copy=False)
        place_desc = place_desc.astype(np.float32, copy=False)
        with Timer() as t:
            if do_fast_matching:
                matches = fast_matching(
                    query_desc, place_desc, ratio_thresh, labels=place_lms)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_L2)
                matches = matcher.knnMatch(query_desc, place_desc, k=2)
                matches1, matches2 = list(zip(*matches))
                (matches1, dist1) = matches_cv2np(matches1)
                (matches2, dist2) = matches_cv2np(matches2)
                good = (place_lms[matches1[:, 1]] == place_lms[matches2[:, 1]])
                good = good | (dist1/dist2 < ratio_thresh)
                matches = matches1[good]
        duration = t.duration
    else:
        matches = np.empty((0, 2), np.int32)

    if debug_dict is not None and len(matches) > 0:
        lm_frames = [frame_id for frame_id, db in zip(frame_ids, place_db)
                     for _ in db.landmark_ids]
        lm_indices = np.concatenate([np.arange(len(db.keypoints))
                                     for db in place_db])
        sorted_frames, counts = np.unique(
            [lm_frames[m2] for m1, m2 in matches], return_counts=True)
        best_frame_id = sorted_frames[np.argmax(counts)]
        best_matches = [(m1, m2) for m1, m2 in matches
                        if lm_frames[m2] == best_frame_id]
        best_matches = np.array(best_matches)
        best_matches = np.stack([best_matches[:, 0],
                                 lm_indices[best_matches[:, 1]]], -1)
        debug_dict['best_id'] = best_frame_id
        debug_dict['best_matches'] = best_matches
        debug_dict['lm_frames'] = lm_frames
        debug_dict['lm_indices'] = lm_indices

    return matches, place_lms, duration


def do_pnp(kpts, lms, query_info, config):
    kpts = kpts.astype(np.float32).reshape((-1, 1, 2))
    lms = lms.astype(np.float32).reshape((-1, 1, 3))

    success, R_vec, t, inliers = cv2.solvePnPRansac(
        lms, kpts, query_info.K, np.array([query_info.dist, 0, 0, 0]),
        iterationsCount=5000, reprojectionError=config['reproj_error'],
        flags=cv2.SOLVEPNP_P3P)

    if success:
        inliers = inliers[:, 0]
        num_inliers = len(inliers)
        inlier_ratio = len(inliers) / len(kpts)
        success &= num_inliers >= config['min_inliers']

        ret, R_vec, t = cv2.solvePnP(
                lms[inliers], kpts[inliers], query_info.K,
                np.array([query_info.dist, 0, 0, 0]), rvec=R_vec, tvec=t,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ret

        query_T_w = np.eye(4)
        query_T_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        query_T_w[:3, 3] = t[:, 0]
        w_T_query = np.linalg.inv(query_T_w)

        ret = LocResult(success, num_inliers, inlier_ratio, w_T_query)
    else:
        inliers = np.empty((0,), np.int32)
        ret = loc_failure

    return ret, inliers

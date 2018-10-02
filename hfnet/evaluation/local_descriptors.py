import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path

from .utils import matching, keypoints_filter_borders, sample_descriptors
from .utils import keypoints_warp_2D, keypoints_cv2np, div0, to_homogeneous
from hfnet.settings import EXPER_PATH


def sift_loader(image, name, **config):
    num_features = config.get('num_features', 0)

    sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=num_features, contrastThreshold=1e-5)
    kpts, desc = sift.detectAndCompute(image.astype(np.uint8), None)
    kpts, scores = keypoints_cv2np(kpts)
    return {'keypoints': kpts, 'descriptors': desc}


def export_loader(image, name, experiment, **config):
    num_features = config.get('num_features', 0)
    remove_borders = config.get('remove_borders', 0)
    keypoint_predictor = config.get('keypoint_predictor', None)

    path = Path(EXPER_PATH, 'exports', experiment, name.decode('utf-8')+'.npz')
    with np.load(path) as p:
        pred = {k: v.copy() for k, v in p.items()}
    if keypoint_predictor:
        keypoint_config = config.get('keypoint_config', config)
        keypoint_config['keypoint_predictor'] = None
        pred_keypoints = keypoint_predictor(
            image, name, **{'experiment': experiment, **keypoint_config})
        pred['keypoints'] = pred_keypoints['keypoints']
    else:
        assert 'keypoints' in pred
        pred['keypoints'] = pred['keypoints'][:, ::-1]
    if remove_borders:
        pred['keypoints'] = keypoints_filter_borders(
            pred['keypoints'], image.shape[:2], remove_borders)
    if num_features:
        pred['keypoints'] = pred['keypoints'][:num_features]
    if 'descriptors' not in pred:
        pred['descriptors'] = sample_descriptors(
            pred['local_descriptor_map'], pred['keypoints'], image.shape[:2])
    return pred


def compute_homography_correctness(kpts1, kpts2, matches, shape2, H_gt,
                                   dist_thresh=3):
    if matches.shape[0] == 0:
        return False, None
    kpts1 = kpts1[matches[:, 0]]
    kpts2 = kpts2[matches[:, 1]]
    H, _ = cv2.findHomography(kpts2, kpts1, cv2.RANSAC, 5.0)
    if H is None:
        return False, None

    h, w = shape2
    corners2 = to_homogeneous(
        np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]))
    corners1_gt = np.dot(corners2, np.transpose(H_gt))
    corners1_gt = corners1_gt[:, :2] / corners1_gt[:, 2:]
    corners1 = np.dot(corners2, np.transpose(H))
    corners1 = corners1[:, :2] / corners1[:, 2:]
    mean_dist = np.mean(np.linalg.norm(corners1 - corners1_gt, axis=1))
    correct = (mean_dist <= dist_thresh)
    return correct, mean_dist


def compute_matching_score(kpts1, kpts2, matches, shape1, shape2, H_gt,
                           dist_thresh=5):
    kpts1_w, visibility1 = keypoints_warp_2D(
        kpts1, np.linalg.inv(H_gt), shape2)
    kpts2_w, visibility2 = keypoints_warp_2D(
        kpts2, H_gt, shape1)

    def compute_matching_score_single(kpts, kpts_w, matches, visibility):
        visibility_matched = visibility[matches[:, 0]]
        match_dist = np.linalg.norm(kpts[matches[:, 0]]
                                    - kpts_w[matches[:, 1]], axis=-1)
        correct_matches = ((match_dist < dist_thresh)*visibility_matched).sum()
        match_score = correct_matches / np.maximum(np.sum(visibility), 1.0)
        inlier_ratio = (correct_matches
                        / np.maximum(np.sum(visibility_matched), 1.0))
        return match_score, inlier_ratio

    score1, ratio1 = compute_matching_score_single(
        kpts1, kpts2_w, matches, visibility1)
    score2, ratio2 = compute_matching_score_single(
        kpts2, kpts1_w, matches[:, ::-1], visibility2)
    score = (score1 + score2) / 2
    ratio = (ratio1 + ratio2) / 2
    return score, ratio


def compute_tp_fp(kpts1, kpts2, matches, distances, shape1, H_gt,
                  dist_thresh=3):
    kpts2_w, _ = keypoints_warp_2D(kpts2, H_gt, shape1)
    all_2D_dist = np.linalg.norm(kpts1[:, np.newaxis]
                                 - kpts2_w[np.newaxis], axis=-1)
    num_gt = (all_2D_dist.min(1) < dist_thresh).sum()
    tp = np.linalg.norm(kpts1[matches[:, 0]]
                        - kpts2_w[matches[:, 1]], axis=-1) < dist_thresh
    fp = np.logical_not(tp)
    return num_gt, tp, fp, distances


def compute_pr(tp, distances, num_gt):
    sort_idx = np.argsort(distances)
    tp = tp[sort_idx]
    distances = distances[sort_idx]
    fp = np.logical_not(tp)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = div0(tp_cum, num_gt)
    precision = div0(tp_cum, tp_cum + fp_cum)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, distances


def compute_average_precision(precision, recall):
    return np.sum(precision[1:] * (recall[1:] - recall[:-1]))


def evaluate(data_iter, config):
    hcorrectness = []
    matching_scores = []
    inlier_ratios = []
    all_tp = []
    all_num_gt = 0
    all_distances = []

    for data in tqdm(data_iter):
        shape1, shape2 = data['image'].shape[:2], data['image_ref'].shape[:2]
        pred1 = config['predictor'](
            data['image'], data['name'], **config)
        pred2 = config['predictor'](
            data['image_ref'], data['name_ref'], **config)
        all_matches, matches_dist = matching(
            pred1['descriptors'], pred2['descriptors'],
            do_ratio_test=config['do_ratio_test'])
        matches = all_matches[matches_dist < config['match_thresh']]

        hcorrect, dist = compute_homography_correctness(
            pred1['keypoints'], pred2['keypoints'], matches, shape2,
            data['homography'], config['correct_match_thresh'])
        hcorrectness.append(hcorrect)

        matching_score, inlier_ratio = compute_matching_score(
            pred1['keypoints'], pred2['keypoints'], matches, shape1, shape2,
            data['homography'], config['correct_match_thresh'])
        matching_scores.append(matching_score)
        inlier_ratios.append(inlier_ratio)

        num_gt, tp, _, distances = compute_tp_fp(
            pred1['keypoints'], pred2['keypoints'], all_matches, matches_dist,
            shape1, data['homography'], config['correct_match_thresh'])
        all_tp.append(tp)
        all_num_gt += num_gt
        all_distances.append(distances)

    precision, recall, distances = compute_pr(
        np.concatenate(all_tp, 0), np.concatenate(all_distances, 0),
        all_num_gt)
    mAP = compute_average_precision(precision, recall)

    metrics = {
        'homography_correctness': np.mean(hcorrectness),
        'matching_score': np.mean(matching_scores),
        'inlier_ratio': np.mean(inlier_ratios),
        'mAP': mAP,
    }
    return metrics, precision, recall, distances

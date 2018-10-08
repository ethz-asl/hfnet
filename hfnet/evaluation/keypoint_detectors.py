import numpy as np
from tqdm import tqdm

from .utils import keypoints_warp_2D, div0
from .shared import compute_pr, compute_average_precision


def compute_correctness(kpts1, kpts2, shape1, shape2, H_gt, thresh):
    kpts1_w, visibility1 = keypoints_warp_2D(
        kpts1, np.linalg.inv(H_gt), shape2)
    kpts2_w, visibility2 = keypoints_warp_2D(
        kpts2, H_gt, shape1)

    def compute_correctness_single(kpts, kpts_w):
        dist = np.linalg.norm(kpts_w[:, np.newaxis]
                              - kpts[np.newaxis], axis=-1)
        dist = np.min(dist, axis=1)
        correct = dist <= thresh
        return dist, correct

    dist1, correct1 = compute_correctness_single(kpts2, kpts1_w)
    dist2, correct2 = compute_correctness_single(kpts1, kpts2_w)
    return correct1, correct2, dist1, dist2, visibility1, visibility2


def evaluate(data_iter, config):
    num_kpts = []
    loc_error = []
    repeatability = []
    all_tp = []
    all_num_gt = 0
    all_scores = []

    for data in tqdm(data_iter):
        shape1, shape2 = data['image'].shape[:2], data['image2'].shape[:2]
        pred1 = config['predictor'](
            data['image'], data['name'], **config)
        pred2 = config['predictor'](
            data['image2'], data['name2'], **config)
        num_kpts.extend([len(pred1['keypoints']), len(pred2['keypoints'])])

        if len(pred1['keypoints']) == 0 or len(pred2['keypoints']) == 0:
            repeatability.append(0)
            continue

        correct1, correct2, dist1, dist2, vis1, vis2 = compute_correctness(
            pred1['keypoints'], pred2['keypoints'], shape1, shape2,
            data['homography'], config['correct_match_thresh'])

        error1 = dist1[vis1 & correct1]
        if len(error1) > 0:
            loc_error.append(error1.mean())
        error2 = dist2[vis2 & correct2]
        if len(error2) > 0:
            loc_error.append(error2.mean())

        repeat = div0(correct1[vis1].sum() + correct2[vis2].sum(),
                      vis1.sum() + vis2.sum())
        repeatability.append(repeat)

        all_tp.extend([correct1[vis1], correct2[vis2]])
        all_scores.extend([pred1['scores'][vis1], pred2['scores'][vis2]])
        all_num_gt += vis2.sum() + vis1.sum()

    precision, recall, scores = compute_pr(
        np.concatenate(all_tp, 0), np.concatenate(all_scores, 0), all_num_gt,
        reverse=True)  # confidence is in decreasing order
    mAP = compute_average_precision(precision, recall)

    metrics = {
        'average_num_keypoints': np.mean(num_kpts),
        'localization_error': np.mean(loc_error),
        'repeatability': np.mean(repeatability),
        'mAP': mAP,
    }
    return metrics, precision, recall, scores

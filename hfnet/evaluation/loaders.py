import cv2
import numpy as np
from pathlib import Path

from .utils import keypoints_filter_borders, sample_descriptors, nms_fast
from .utils import keypoints_cv2np
from hfnet.settings import EXPER_PATH


def sift_loader(image, name, **config):
    num_features = config.get('num_features', 0)
    do_nms = config.get('do_nms', False)
    nms_thresh = config.get('nms_thresh', 4)
    do_root = config.get('root', False)

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=1e-5)
    kpts, desc = sift.detectAndCompute(image.astype(np.uint8), None)
    kpts, scores = keypoints_cv2np(kpts)
    if do_nms:
        keep = nms_fast(kpts, scores, image.shape[:2], nms_thresh)
        kpts, scores, desc = kpts[keep], scores[keep], desc[keep]
    if num_features:
        keep_indices = np.argsort(scores)[::-1][:num_features]
        kpts, desc, scores = [i[keep_indices] for i in [kpts, desc, scores]]
    if do_root:
        desc /= np.sum(desc, axis=-1, keepdims=True)
        desc = np.sqrt(desc)
    return {'keypoints': kpts, 'descriptors': desc, 'scores': scores}


def fast_loader(image, name, **config):
    num_features = config.get('num_features', 0)
    do_nms = config.get('do_nms', False)
    nms_thresh = config.get('nms_thresh', 4)

    fast = cv2.FastFeatureDetector_create()
    kpts = fast.detect(image.astype(np.uint8), None)
    kpts, scores = keypoints_cv2np(kpts)
    if do_nms:
        keep = nms_fast(kpts, scores, image.shape[:2], nms_thresh)
        kpts, scores = kpts[keep], scores[keep]
    if num_features:
        keep_indices = np.argsort(scores)[::-1][:num_features]
        kpts, scores = [i[keep_indices] for i in [kpts, scores]]
    return {'keypoints': kpts, 'scores': scores}


def harris_loader(image, name, **config):
    num_features = config.get('num_features', 0)
    do_nms = config.get('do_nms', False)
    nms_thresh = config.get('nms_thresh', 4)

    detect_map = cv2.cornerHarris(image.astype(np.uint8), 4, 3, 0.04)
    kpts = np.where(detect_map > 1e-6)
    scores = detect_map[kpts]
    kpts = np.stack([kpts[1], kpts[0]], axis=-1)
    if do_nms:
        keep = nms_fast(kpts, scores, image.shape[:2], nms_thresh)
        kpts, scores = kpts[keep], scores[keep]
    if num_features:
        keep_indices = np.argsort(scores)[::-1][:num_features]
        kpts, scores = [i[keep_indices] for i in [kpts, scores]]
    return {'keypoints': kpts, 'scores': scores}


def export_loader(image, name, experiment, **config):
    num_features = config.get('num_features', 0)
    remove_borders = config.get('remove_borders', 0)
    keypoint_predictor = config.get('keypoint_predictor', None)
    do_nms = config.get('do_nms', False)
    nms_thresh = config.get('nms_thresh', 4)
    keypoint_refinement = config.get('keypoint_refinement', False)
    binarize = config.get('binarize', False)
    entries = ['keypoints', 'scores', 'descriptors']

    path = Path(EXPER_PATH, 'exports', experiment, name.decode('utf-8')+'.npz')
    with np.load(path) as p:
        pred = {k: v.copy() for k, v in p.items()}
    if keypoint_predictor:
        keypoint_config = config.get('keypoint_config', config)
        keypoint_config['keypoint_predictor'] = None
        pred_detector = keypoint_predictor(
            image, name, **{'experiment': experiment, **keypoint_config})
        pred['keypoints'] = pred_detector['keypoints']
        pred['scores'] = pred_detector['scores']
    else:
        assert 'keypoints' in pred
        if remove_borders:
            mask = keypoints_filter_borders(
                pred['keypoints'], image.shape[:2], remove_borders)
            pred = {**pred,
                    **{k: v[mask] for k, v in pred.items() if k in entries}}
        if do_nms:
            keep = nms_fast(
                pred['keypoints'], pred['scores'], image.shape[:2], nms_thresh)
            pred = {**pred,
                    **{k: v[keep] for k, v in pred.items() if k in entries}}
        if keypoint_refinement:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            pred['keypoints'] = cv2.cornerSubPix(
                image, np.float32(pred['keypoints']),
                (3, 3), (-1, -1), criteria)
    if num_features:
        keep = np.argsort(pred['scores'])[::-1][:num_features]
        pred = {**pred,
                **{k: v[keep] for k, v in pred.items() if k in entries}}
    if 'descriptors' not in pred:
        pred['descriptors'] = sample_descriptors(
            pred['local_descriptor_map'], pred['keypoints'], image.shape[:2])
    if binarize:
        pred['descriptors'] = pred['descriptors'] > 0
    return pred

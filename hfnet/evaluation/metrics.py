import numpy as np

from .utils.misc import div0


def compute_pr(tp, distances, num_gt, reverse=False):
    sort_idx = np.argsort(distances)
    if reverse:
        sort_idx = sort_idx[::-1]
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

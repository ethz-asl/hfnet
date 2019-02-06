import numpy as np
import cv2


def normalize(l, axis=-1):
    return np.array(l) / np.linalg.norm(l, axis=axis, keepdims=True)


def root_descriptors(d, axis=-1):
    return np.sqrt(d / np.sum(d, axis=axis, keepdims=True))


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


def sample_descriptors(descriptor_map, keypoints, image_shape,
                       input_shape=None, do_round=True):
    '''In some cases, the deep network computing the dense descriptors requires
       the input to be divisible by the downsampling rate, and crops the
       remaining pixels (PyTorch) or pad the input (Tensorflow). We assume the
       PyTorch behavior and round the factor between the network input
       (input shape) and the output (desc_shape). The keypoints are assumed to
       be at the scale of image_shape.
    '''
    fix = np.round if do_round else lambda x: x
    image_shape = np.array(image_shape)
    desc_shape = np.array(descriptor_map.shape[:-1])

    if input_shape is not None:
        input_shape = np.array(input_shape)
        factor = image_shape / input_shape
        effective_input_shape = desc_shape * fix(input_shape / desc_shape)
        factor = factor * (effective_input_shape - 1) / (desc_shape - 1)
    else:
        factor = (image_shape - 1) / (desc_shape - 1)

    desc = sample_bilinear(descriptor_map, keypoints/factor[::-1])
    desc = normalize(desc, axis=1)
    assert np.all(np.isfinite(desc))
    return desc


def matching(desc1, desc2, do_ratio_test=False, cross_check=True):
    if desc1.dtype == np.bool and desc2.dtype == np.bool:
        desc1, desc2 = np.packbits(desc1, axis=1), np.packbits(desc2, axis=1)
        norm = cv2.NORM_HAMMING
    else:
        desc1, desc2 = np.float32(desc1), np.float32(desc2)
        norm = cv2.NORM_L2

    if do_ratio_test:
        matches = []
        matcher = cv2.BFMatcher(norm)
        for m, n in matcher.knnMatch(desc1, desc2, k=2):
            m.distance = 1.0 if (n.distance == 0) else m.distance / n.distance
            matches.append(m)
    else:
        matcher = cv2.BFMatcher(norm, crossCheck=cross_check)
        matches = matcher.match(desc1, desc2)
    return matches_cv2np(matches)


def fast_matching(desc1, desc2, ratio_thresh, labels=None):
    '''A fast matching method that matches multiple descriptors simultaneously.
       Assumes that descriptors are normalized and can run on GPU if available.
       Performs the landmark-aware ratio test if labels are provided.
    '''
    import torch
    cuda = torch.cuda.is_available()

    desc1, desc2 = torch.from_numpy(desc1), torch.from_numpy(desc2)
    if cuda:
        desc1, desc2 = desc1.cuda(), desc2.cuda()

    with torch.no_grad():
        dist = 2*(1 - desc1 @ desc2.t())
        dist_nn, ind = dist.topk(2, dim=-1, largest=False)
        match_ok = (dist_nn[:, 0] <= (ratio_thresh**2)*dist_nn[:, 1])

        if labels is not None:
            labels = torch.from_numpy(labels)
            if cuda:
                labels = labels.cuda()
            labels_nn = labels[ind]
            match_ok |= (labels_nn[:, 0] == labels_nn[:, 1])

        matches = torch.stack(
            [torch.nonzero(match_ok)[:, 0], ind[match_ok][:, 0]], dim=-1)

    return matches.cpu().numpy()


def matches_cv2np(matches_cv):
    matches_np = np.int32([[m.queryIdx, m.trainIdx] for m in matches_cv])
    distances = np.float32([m.distance for m in matches_cv])
    return matches_np.reshape(-1, 2), distances

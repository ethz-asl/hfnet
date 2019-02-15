import cv2
import numpy as np
from matplotlib import pyplot as plt


def baseline_sift_matching(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)

    good = [[m] for m, n in matches if m.distance < 0.7*n.distance]
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                              matchColor=(0, 255, 0), matchesMask=None,
                              singlePointColor=(255, 0, 0), flags=0)
    return img3


def debug_matching(frame1, frame2, path_image1, path_image2, matches,
                   matches_mask, num_points, use_ratio_test):
    img1 = cv2.imread(path_image1, 0)
    img2 = cv2.imread(path_image2, 0)

    kp1 = get_ocv_kpts_from_np(frame1['keypoints'][:num_points, :])
    kp2 = get_ocv_kpts_from_np(frame2['keypoints'][:num_points, :])

    if use_ratio_test:
        img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                 matchColor=(0, 255, 0),
                                 matchesMask=matches_mask,
                                 singlePointColor=(255, 0, 0), flags=0)
    else:
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(255, 0, 0), flags=0)

    img_sift = baseline_sift_matching(img1, img2)

    fig = plt.figure(figsize=(2, 1))
    fig.add_subplot(2, 1, 1)
    plt.imshow(img)
    plt.title('Custom features')
    fig.add_subplot(2, 1, 2)
    plt.imshow(img_sift)
    plt.title('SIFT')
    plt.show()


def get_ocv_kpts_from_np(keypoints_np):
    return [cv2.KeyPoint(x=x, y=y, _size=1) for x, y in keypoints_np]


def match_frames(path_npz1, path_npz2, path_image1, path_image2, num_points,
                 use_ratio_test, ratio_test_values, debug):
    frame1 = np.load(path_npz1)
    frame2 = np.load(path_npz2)

    # Assert the keypoints are sorted according to the score.
    assert np.all(np.sort(frame1['scores'])[::-1] == frame1['scores'])

    # WARNING: scores are not taken into account as of now.
    des1 = frame1['descriptors'].astype('float32')[:num_points]
    des2 = frame2['descriptors'].astype('float32')[:num_points]

    if use_ratio_test:
        keypoint_matches = [[] for _ in ratio_test_values]
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)

        smallest_distances = [dict() for _ in ratio_test_values]

        matches_mask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            for ratio_idx, ratio in enumerate(ratio_test_values):
                if m.distance < ratio * n.distance:
                    if m.trainIdx not in smallest_distances[ratio_idx]:
                        smallest_distances[ratio_idx][m.trainIdx] = (
                            m.distance, m.queryIdx)
                        matches_mask[i] = [1, 0]
                        keypoint_matches[ratio_idx].append(
                            (m.queryIdx, m.trainIdx))
                    else:
                        old_dist, old_queryIdx = smallest_distances[
                            ratio_idx][m.trainIdx]
                        if m.distance < old_dist:
                            old_distance, old_queryIdx = smallest_distances[
                                ratio_idx][m.trainIdx]
                            smallest_distances[ratio_idx][m.trainIdx] = (
                                m.distance, m.queryIdx)
                            matches_mask[i] = [1, 0]
                            keypoint_matches[ratio_idx].remove(
                                (old_queryIdx, m.trainIdx))
                            keypoint_matches[ratio_idx].append(
                                (m.queryIdx, m.trainIdx))
    else:
        keypoint_matches = [[]]
        matches_mask = []
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)

        # Matches are already cross-checked.
        for match in matches:
            # match.trainIdx belongs to des2.
            keypoint_matches[0].append((match.queryIdx, match.trainIdx))

    if debug:
        debug_matching(frame1, frame2, path_image1, path_image2, matches,
                       matches_mask, num_points, use_ratio_test)

    return keypoint_matches

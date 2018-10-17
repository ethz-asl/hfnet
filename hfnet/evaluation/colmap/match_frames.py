import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_ocv_kpts_from_np(numpy_keypoints):
    ocv_keypoints = []
    for keypoint in numpy_keypoints:
      ocv_keypoints.append(cv2.KeyPoint(x=keypoint[0], y=keypoint[1], _size=1))

    return ocv_keypoints

def baseline_sift_matching(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = None, flags = 0)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None,**draw_params)
    plt.imshow(img3)
    plt.show()


def match_frames(path_frame1, path_frame2, path_image1, path_image2, num_points, debug=False):
    frame1 = np.load(path_frame1)
    frame2 = np.load(path_frame2)

    # Assert the keypoints are sorted according to the score.
    assert(np.sort(frame1['scores']).all() == frame1['scores'].all())

    # WARNING: scores are not taken into account as of now.
    des1 = frame1['descriptors'].astype('float32')[:num_points,:]
    des2 = frame2['descriptors'].astype('float32')[:num_points,:]

    RATIO_TEST = True
    if RATIO_TEST:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)

        # Ratio test as per Lowe's paper.
        matchesMask = [[0,0] for i in xrange(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.80*n.distance:
                matchesMask[i]=[1,0]
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)

    keypoint_matches = []
    for match in matches:
        # match.trainIdx belongs to des2.
        keypoint_matches.append((match[0].queryIdx, match[0].trainIdx))

    if debug:
        img1 = cv2.imread(path_image1, 0)
        img2 = cv2.imread(path_image2, 0)

        current_width = frame1['image_size'][0]
        current_height = frame1['image_size'][1]

        if (current_width > current_height):
            original_width = 1600
        else:
            original_width = 1063

        scaling = float(original_width) / current_width

        kp1 = frame1['keypoints'][:num_points,:] * scaling
        kp2 = frame2['keypoints'][:num_points,:] * scaling

        cvkp1 = get_ocv_kpts_from_np(kp1)
        cvkp2 = get_ocv_kpts_from_np(kp2)

        if RATIO_TEST:
            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask, flags = 0)
            img = cv2.drawMatchesKnn(img1, cvkp1, img2, cvkp2, matches, None,
                                     **draw_params)
        else:
            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0), flags = 0)
            img = cv2.drawMatches(img1, cvkp1, img2, cvkp2, matches, None, **draw_params)
        plt.imshow(img)
        plt.show()

        # Switch on to compare to SIFT.
        # baseline_sift_matching(img1, img2)

    return keypoint_matches

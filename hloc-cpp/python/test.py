import numpy as np

import _hloc_cpp


def main():
    local_descriptor_size = 256
    global_descriptor_size = 1024

    print('test hloc')
    localdescr = np.random.rand(local_descriptor_size, 4).astype(np.float32)
    keypoints = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]]).astype(np.float32) # Why other input and transpose doesn't work?

    globaldescr = np.random.rand(1, global_descriptor_size).astype(np.float32)

    hloc = _hloc_cpp.HLoc()
    image_idx = hloc.addImage(globaldescr, keypoints, localdescr)
    print('Added first image', image_idx)

    for i in range(0, 2):
        globaldescr2 = np.random.rand(1, global_descriptor_size).astype(np.float32)
        localdescr2 = np.random.rand(local_descriptor_size, 3).astype(np.float32)
        keypoints2 = np.random.rand(2, 3).astype(np.float32)
        image_idx2 = hloc.addImage(globaldescr2, keypoints2, localdescr2)
        print('Added image', image_idx2)

    # Add obs: image 0, kpt 2; image 1, kpt 5 etc.
    observing_keyframes = np.array([[0, 0]]).astype(np.int32)
    hloc.add3dPoint(np.array((-1,  0, 1)).astype(np.float32), observing_keyframes)
    observing_keyframes = np.array([[0, 1]]).astype(np.int32)
    hloc.add3dPoint(np.array((1,  0, 1)).astype(np.float32), observing_keyframes)
    observing_keyframes = np.array([[0, 2]]).astype(np.int32)
    hloc.add3dPoint(np.array((0, -1, 1)).astype(np.float32), observing_keyframes)
    observing_keyframes = np.array([[0, 3]]).astype(np.int32)
    hloc.add3dPoint(np.array((0,  1, 1)).astype(np.float32), observing_keyframes)

    point_xyz = np.array((1, 2, 3)).astype(np.float32)

    observing_keyframes = np.array([[1, 2], [1, 0], [2, 1]]).astype(np.int32)
    hloc.add3dPoint(point_xyz, observing_keyframes)
    observing_keyframes = np.array([[2, 2], [1, 1], [2, 0]]).astype(np.int32)
    hloc.add3dPoint(point_xyz, observing_keyframes)

    hloc.buildIndex()

    # Check retrieval
    ret = hloc.localize(globaldescr, keypoints, localdescr)
    (success, num_components_tested, num_inliers,
     num_iters, global_ms, covis_ms, local_ms, pnp_ms) = ret
    print(success, num_components_tested, num_inliers, num_iters)
    print('Timing: ', global_ms, covis_ms, local_ms, pnp_ms)


if __name__ == "__main__":
    main()

import numpy as np


# Code from http://nipy.org/nibabel/reference/nibabel.quaternions.html
def quat2mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-10:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def convert_nvm_pose_to_colmap_p(q_nvm, p_nvm):
    R = quat2mat(q_nvm)
    p_colmap = R.dot(-p_nvm)
    return p_colmap


def test():
    # Colmap output
    # 3157 0.921457 -0.0279726 0.372923 -0.105178 -3.43436 -1.03257 -1.86432 3157 db/384.jpg
    q_colmap = np.array([0.921457, -0.0279726, 0.372923, -0.105178])
    p_colmap = np.array([-3.43436, -1.03257, -1.86432])

    # NVM
    # db/384.jpg 3186.77 0.921457 -0.0279726 0.372923 -0.105178 0.911128 1.3598 3.6956 -0.0538065 0
    q_nvm = np.array([0.921457, -0.0279726, 0.372923, -0.105178])
    p_nvm = np.array([0.911128, 1.3598, 3.6956])

    np.testing.assert_equal(q_colmap, q_nvm)

    p_colmap_computed = convert_nvm_pose_to_colmap_p(q_nvm, p_nvm)

    np.testing.assert_almost_equal(p_colmap_computed, p_colmap, decimal=4)


if __name__ == "__main__":
    test()

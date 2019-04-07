import matplotlib.pyplot as plt
import numpy as np


def plot_images(imgs, titles=None, cmap='brg', ylabel='', normalize=True,
                r=(0, 1), dpi=100, titlefont=None, labelfont=None, title=None,
                keypoints=None, keypoint_colors=(0, 1, 0), keypoints_size=3):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if keypoints is not None:
        assert len(keypoints) == n
        if not isinstance(keypoint_colors[0], (tuple, list)):
            keypoint_colors = [keypoint_colors]*n
    fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
    if n == 1:
        ax = [ax]
    if title is not None:
        fig.suptitle(title)
    for i in range(n):
        if len(imgs[i].shape) == 3:
            if imgs[i].shape[-1] == 3:
                imgs[i] = imgs[i]
            elif imgs[i].shape[-1] == 1:
                imgs[i] = imgs[i][..., 0]
        if len(imgs[i].shape) == 2 and cmap[i] == 'brg':
            cmap[i] = 'gray'
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else r[0],
                     vmax=None if normalize else r[1])
        if keypoints is not None:
            ax[i].scatter(keypoints[i][:, 0], keypoints[i][:, 1],
                          s=keypoints_size, c=keypoint_colors[i])
        if titles:
            ax[i].set_title(titles[i], fontsize=titlefont)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    if ylabel:
        ax[0].set_ylabel(ylabel, fontsize=labelfont)
    plt.tight_layout()


def plot_matches(img1, kp1, img2, kp2, matches, color=None, kp_size=4,
                 thickness=2.0, margin=20, cmap='brg', ylabel='',
                 normalize=True, r=(0, 1), dpi=100, title=None):
    # Create frame
    if len(img1.shape) == 2:
        img1 = img1[..., np.newaxis]
    if len(img2.shape) == 2:
        img2 = img2[..., np.newaxis]
    if img1.shape[-1] == 1:
        img1 = np.repeat(img1, 3, -1)
    if img2.shape[-1] == 1:
        img2 = np.repeat(img2, 3, -1)
    tile_shape = (max(img1.shape[0], img2.shape[0]),
                  img1.shape[1]+img2.shape[1]+margin,
                  img1.shape[2])
    tile = np.ones(tile_shape, type(img1.flat[0]))
    if np.max(img1) > 1 or np.max(img2) > 1:
        tile *= 255

    # Place original images
    tile[0:img1.shape[0], 0:img1.shape[1]] = img1
    tile[0:img2.shape[0],
         img1.shape[1]+margin:img1.shape[1]+img2.shape[1]+margin] = img2

#    fig, ax = plt.subplots(1, 1, dpi=dpi, frameon=False, figsize=(12, 12))
    fig = plt.figure(frameon=False, dpi=dpi)
    w, h, _ = tile.shape
    fig_size = 12
    fig.set_size_inches(fig_size, fig_size/h*w)
    ax = plt.Axes(fig, [0., 0., 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(tile, cmap=plt.get_cmap(cmap),
              vmin=None if normalize else r[0],
              vmax=None if normalize else r[1], aspect='auto')

    kp1 = kp1[matches[:, 0]]
    kp2 = kp2[matches[:, 1]]
    kp2[:, 0] += img1.shape[1]+margin
    xs = np.stack([kp1[:, 0], kp2[:, 0]], 0)
    ys = np.stack([kp1[:, 1], kp2[:, 1]], 0)

    if isinstance(color, list) and len(color) == len(matches):
        for i, c in enumerate(color):
            ax.plot(
                xs[:, i], ys[:, i], linestyle='-', linewidth=thickness,
                aa=True, marker='.', markersize=kp_size, color=c,
            )
    else:
        ax.plot(
            xs, ys, linestyle='-', linewidth=thickness,
            aa=True, marker='.', markersize=kp_size, color=color,
        )

    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for spine in ax.spines.values():  # remove frame
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=None)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=None)


def add_frame(img, c, b=40):
    """Add a colored frame around an image with color c and thickness b
    """
    img = img.copy()
    img[:, :b] = img[:b, :] = img[:, -b:] = img[-b:, :] = c
    return img

import cv2 as cv
import numpy as np
import tensorflow as tf


augmentations = [
        'additive_gaussian_noise',
        'additive_speckle_noise',
        'random_brightness',
        'random_contrast',
        'additive_shade',
        'motion_blur'
]


def additive_gaussian_noise(image, stddev_range=[5, 95]):
    stddev = tf.random_uniform((), *stddev_range)
    noise = tf.random_normal(tf.shape(image), stddev=stddev)
    noisy_image = tf.clip_by_value(image + noise, 0, 255)
    return noisy_image


def additive_speckle_noise(image, prob_range=[0.0, 0.005]):
    prob = tf.random_uniform((), *prob_range)
    sample = tf.random_uniform(tf.shape(image))
    noisy_image = tf.where(sample <= prob, tf.zeros_like(image), image)
    noisy_image = tf.where(sample >= (1. - prob), 255.*tf.ones_like(image), noisy_image)
    return noisy_image


def random_brightness(image, delta_range=[-50, 50]):
    delta = tf.random_uniform((), *delta_range)
    return tf.clip_by_value(tf.image.adjust_brightness(image, delta), 0, 255)


def random_contrast(image, strength_range=[0.5, 1.5]):
    return tf.clip_by_value(tf.image.random_contrast(image, *strength_range), 0, 255)


def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350], prob=0.5):

    def _py_additive_shade(img):
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], np.uint8)
        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*transparency_range)
        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis]/255.)
        return np.clip(shaded, 0, 255)

    f = lambda: tf.reshape(
        tf.py_func(_py_additive_shade, [image], tf.float32), tf.shape(image))
    rand = tf.random_uniform(())
    return tf.cond(rand < prob, f, lambda: tf.identity(image))


def motion_blur(image, kernel_size=[0, 10], prob=0.5):

    def _py_motion_blur(img):
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        # ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
        ksize = np.random.randint(kernel_size[0]//2, (kernel_size[1]+1)//2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv.filter2D(img, -1, kernel)
        return img

    f = lambda: tf.reshape(
        tf.py_func(_py_motion_blur, [image], tf.float32), tf.shape(image))
    rand = tf.random_uniform(())
    return tf.cond(rand < prob, f, lambda: tf.identity(image))

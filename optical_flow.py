"""
This file adapted from Sean Batzel's submission to Homework 2.
"""

import argparse
import logging

import imageio
import numpy as np
from scipy.ndimage.filters import convolve


def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0] - 1, image.shape[1] - 1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0] - 2, image.shape[1] - 2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl + 1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1 - a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1 - a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1 - b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize // 2), ksize // 2,
                                 ksize) ** 2 / 2) / np.sqrt(2 * np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def apply_mask(array, mask):
    """
    There's definitely a better way to do this. I'm still learning how to use
    numpy without completely embarrassing myself.

    :param array: The array to apply mask to
    :param mask: A boolean mask to filter array by
    :returns: A filtered array created by array -> mask
    """
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            if not mask[x, y, 0]:
                array[x, y, 0] = 0
                array[x, y, 1] = 0
                array[x, y, 2] = 0
    return array


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]

    kernel_x = np.array([[1., 0., -1.],
                         [2., 0., -2.],
                         [1., 0., -1.]]) / 8.

    kernel_y = np.array([[1., 2., 1.],
                         [0., 0., 0.],
                         [-1., -2., -1.]]) / 8.

    I_x = convolve_img(I, kernel_x)
    I_y = convolve_img(I, kernel_y)
    I_t = I - H

    Ixx = (I_x * I_x) * mask
    Ixy = (I_x * I_y) * mask
    Iyy = (I_y * I_y) * mask
    Ixt = (I_x * I_t) * mask
    Iyt = (I_y * I_t) * mask

    AtA = np.array([[Ixx.sum(), Ixy.sum()],
                    [Ixy.sum(), Iyy.sum()]])
    Atb = -np.array([Ixt.sum(), Iyt.sum()])

    displacement = np.linalg.solve(AtA, Atb)

    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        tranlated_H = translate(H, disp)

        disp += lucas_kanade(tranlated_H, I)[0]

    return disp


def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Retuns:
        An array of images where each image is a blurred and shruken version of the first.
    """
    kernel = gaussian_kernel()

    pyr = [image]
    for level in range(1, int(levels)):
        convolved = convolve_img(pyr[level - 1], kernel)

        decimated = convolved[::2, ::2]

        pyr.append(decimated)

    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""
    initial_d = np.asarray(initial_d, dtype=np.float32)

    pyramid_H = gaussian_pyramid(H, levels)
    pyramid_I = gaussian_pyramid(I, levels)

    disp = initial_d / 2. ** levels
    for level in range(int(levels)):
        disp *= 2

        level_H = pyramid_H[-(1 + level)]
        level_I = pyramid_I[-(1 + level)]
        
        level_I_displaced = translate(level_I, -disp)
        disp += iterative_lucas_kanade(level_H, level_I_displaced, steps)

    return disp


def track_object(frame1, frame2, boundingBox, steps):
    """
    Attempts to track the object defined by window from frame one to
    frame two.

    args:
        frame1 - the first frame in the sequence
        frame2 - the second frame in the sequence
        boundingBox - A bounding box (x, y, w, h) around the object in the first frame
    """
    x, y, w, h = boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]

    H = frame1[y:y+h, x:x+w]
    I = frame2[y:y+h, x:x+w]

    levels = np.floor(np.log(w if w < h else h))

    initial_displacement = np.array([0, 0])

    flow = pyramid_lucas_kanade(H, I, initial_displacement, levels, steps)

    final_flow = np.array([0, 0, 0, 0, 0, 0])

    if flow[0] < 0:
        final_flow[0] = abs(flow[0])
    elif flow[0] > 0:
        final_flow[1] = abs(flow[0])
    
    if flow[1] < 0:
        final_flow[2] = abs(flow[1])
    elif flow[1] > 0:
        final_flow[3] = abs(flow[1])

    return final_flow

"""
This file adapted from Sean Batzel's submission to Lucas-Kanade implementation.

Sections indicated to be adapted or unmodified from skeleton code are from assignment
code from Dr. Lou Kratz (CS 583, Drexel University).

Portions adapted from "The Implementation of Optical Flow in Neural Networks",
Nicole Ku'ulei-lani Flett http://nrs.harvard.edu/urn-3:HUL.InstRepos:39011510
"""

import imageio
import numpy as np
from scipy.ndimage.filters import convolve
from numpy.linalg.linalg import LinAlgError


def bilinear_interp(image, points):
    # Lucas-Kanade implementation - Unmodified from skeleton code
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

    top = (1 - a) * image[tl[..., 0], tl[..., 1]] + a * image[tr[..., 0], tr[..., 1]]
    bot = (1 - a) * image[bl[..., 0], bl[..., 1]] + a * image[br[..., 0], br[..., 1]]
    return ((1 - b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    # Lucas-Kanade implementation - Unmodified from skeleton code
    pts = np.mgrid[:image.shape[0], :image.shape[1]].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    # Lucas-Kanade implementation - Unmodified from skeleton code
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    # Lucas-Kanade implementation - Unmodified from skeleton code
    kernel = np.exp(-np.linspace(-(ksize // 2), ksize // 2, ksize) ** 2 / 2) / np.sqrt(2 * np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(h, i):
    # Lucas-Kanade implementation - SB
    mask = (h.mean(-1) > 0.25) * (i.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]

    kernel_x = np.array([[1., 0., -1.],
                         [2., 0., -2.],
                         [1., 0., -1.]]) / 8.

    kernel_y = np.array([[1., 2., 1.],
                         [0., 0., 0.],
                         [-1., -2., -1.]]) / 8.

    i_x = convolve_img(i, kernel_x)
    i_y = convolve_img(i, kernel_y)
    i_t = i - h

    ixx = (i_x * i_x) * mask
    ixy = (i_x * i_y) * mask
    iyy = (i_y * i_y) * mask
    ixt = (i_x * i_t) * mask
    iyt = (i_y * i_t) * mask

    at_a = np.array([[ixx.sum(), ixy.sum()],
                    [ixy.sum(), iyy.sum()]])

    eig_vals, eig_vecs = np.linalg.eig(at_a)

    ata = np.array([[eig_vals[0], 0],
                    [0, eig_vals[1]]])

    atb = -np.array([ixt.sum(), iyt.sum()])

    displacement = np.linalg.solve(ata, atb)

    return displacement, ata, atb


def iterative_lucas_kanade(h, i, steps):
    # Adapted from L-K submission.
    disp = np.zeros((2,), np.float32)
    for x in range(steps):
        tranlated_h = translate(h, disp)

        disp += lucas_kanade(tranlated_h, i)[0]

    return disp


def gaussian_pyramid(image, levels):
    # Adapted from L-K submission.
    kernel = gaussian_kernel()

    pyr = [image]
    for level in range(1, int(levels)):
        convolved = convolve_img(pyr[level - 1], kernel)

        decimated = convolved[::2, ::2]

        pyr.append(decimated)

    return pyr


def pyramid_lucas_kanade(h, i, initial_d, levels, steps):
    # Adapted from L-K submission.
    initial_d = np.asarray(initial_d, dtype=np.float32)

    pyramid_h = gaussian_pyramid(h, levels)
    pyramid_i = gaussian_pyramid(i, levels)

    disp = initial_d / 2. ** levels
    for level in range(int(levels)):
        disp *= 2

        level_h = pyramid_h[-(1 + level)]
        level_i = pyramid_i[-(1 + level)]

        level_i_displaced = translate(level_i, -disp)
        disp += iterative_lucas_kanade(level_h, level_i_displaced, steps)

    return disp


def track_object(frame1, frame2, x, y, w, h, steps):
    # Adapted from L-K submission skeleton code.
    h_image = frame1[y:y + h, x:x + w]
    i_image = frame2[y:y + h, x:x + w]

    levels = np.floor(np.log(w if w < h else h))

    initial_displacement = np.array([0, 0])

    flow = pyramid_lucas_kanade(h_image, i_image, initial_displacement, levels, steps)

    final_flow = [0, 0, 0, 0, 0, 0]

    if flow[0] < 0:
        final_flow[0] = abs(flow[0])
    elif flow[0] > 0:
        final_flow[1] = abs(flow[0])

    if flow[1] < 0:
        final_flow[2] = abs(flow[1])
    elif flow[1] > 0:
        final_flow[3] = abs(flow[1])

    return final_flow


def run_lk(first_frame, second_frame, x, y, w, h, steps):
    # Adapted from L-K submission.
    first = imageio.imread(first_frame)[:, :, :3].astype(np.float32) / 255.0
    second = imageio.imread(second_frame)[:, :, :3].astype(np.float32) / 255.0
    return track_object(first, second, int(x), int(y), int(w), int(h), steps)


def prepare_dataset(files_list, result_file, x, y, w, h, steps=5):
    # Adapted from the CSV writer in the original Matlab code in the reference material.
    import csv
    flow_vector = [0, 0, 0, 0, 0, 0]
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(files_list) - 1):
            x_coord = x - flow_vector[0] + flow_vector[1]
            y_coord = y - flow_vector[2] + flow_vector[3]
            try:
                flow_vector = run_lk(
                    files_list[i], files_list[i + 1], x_coord, y_coord, w, h, steps)
                writer.writerow([files_list[i]])
                writer.writerow([files_list[i + 1]])
                writer.writerow(flow_vector)
            except LinAlgError:
                # In the case that the matrix is singular, just ignore.
                pass

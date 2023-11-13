from Helpers import get_height, get_width, clamp_pixel
import numpy as np

def label_cluster(image_dat, labels, label_num, check_pixel, x0, y0, background):
    height = get_height(image_dat)
    width = get_width(image_dat)
    to_visit = []
    to_visit.append([x0, y0])
    while len(to_visit) != 0:
        indx = to_visit.pop()
        x0 = indx[0]
        y0 = indx[1]
        if np.array_equal(image_dat[y0, x0], check_pixel):
            labels[y0, x0] = label_num
        yi = -1
        while yi <= 1:
            xi = -1
            while xi <= 1:
                if y0+yi < 0 or y0+yi > height-1 or x0+xi < 0 or x0+xi > width-1:
                    xi += 1
                    continue
                if labels[y0+yi, x0+xi] != 0:
                    xi += 1
                    continue
                curr_pix = image_dat[y0+yi, x0+xi]
                if np.array_equal(curr_pix, background):
                    xi += 1
                    continue
                if np.array_equal(curr_pix, check_pixel):
                    labels[y0+yi, x0+xi] = label_num
                    to_visit.append([x0+xi, y0+yi])
                xi += 1
            yi += 1
    return labels

def better_ccl(image_dat, background=None):
    if background is None:
        background = np.asarray([0,0,0])
    height = get_height(image_dat)
    width = get_width(image_dat)
    labels = np.zeros([height, width], dtype=np.uint8)

    curr_label = 1
    y = 0
    while y < height:
        x = 0
        while x < width:
            if labels[y, x] != 0:
                x += 1
                continue
            curr_pix = image_dat[y, x]
            if np.array_equal(curr_pix, background):
                xi += 1
                continue
            labels = label_cluster(image_dat, labels, curr_label, curr_pix, x, y, background)
            curr_label += 1
            x += 1
        y += 1
    return labels

def labels_to_image(labels, max_value):
    width = get_width(labels)
    height = get_height(labels)
    new_image = np.zeros([height, width, 3], dtype=np.uint8)

    x = 0
    while x < width:
        y = 0
        while y < height:
            new_image[y, x, 0] = labels[y, x]
            new_image[y, x, 1] = labels[y, x]
            new_image[y, x, 2] = labels[y, x]
            new_image[y, x] = clamp_pixel(new_image[y, x], 0, max_value)
            y += 1
        x += 1
    return new_image
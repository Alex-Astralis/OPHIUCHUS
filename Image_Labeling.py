from Helpers import get_height, get_width, clamp_pixel
import numpy as np

def label_cluster(image, labels, label_num, check_pixel, x0, y0, background):
    height = get_height(image)
    width = get_width(image)
    to_visit = []
    to_visit.append([x0, y0])
    while len(to_visit) != 0:
        indx = to_visit.pop()
        x0 = indx[0]
        y0 = indx[1]
        if np.array_equal(image[y0, x0], check_pixel):
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
                curr_pix = image[y0+yi, x0+xi]
                if np.array_equal(curr_pix, background):
                    xi += 1
                    continue
                if np.array_equal(curr_pix, check_pixel):
                    labels[y0+yi, x0+xi] = label_num
                    to_visit.append([x0+xi, y0+yi])
                xi += 1
            yi += 1
    return labels

def better_ccl(image, background=None):
    if background is None:
        background = np.asarray([0,0,0])
    height = get_height(image)
    width = get_width(image)
    labels = np.zeros([height, width], dtype=np.uint8)

    curr_label = 1
    y = 0
    while y < height:
        x = 0
        while x < width:
            if labels[y, x] != 0:
                x += 1
                continue
            curr_pix = image[y, x]
            if np.array_equal(curr_pix, background):
                xi += 1
                continue
            labels = label_cluster(image, labels, curr_label, curr_pix, x, y, background)
            curr_label += 1
            x += 1
        y += 1
    return labels

def labels_to_image(labels, max_bit):
    width = get_width(labels)
    height = get_height(labels)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)

    x = 0
    while x < width:
        y = 0
        while y < height:
            new_img[y, x, 0] = labels[y, x]
            new_img[y, x, 1] = labels[y, x]
            new_img[y, x, 2] = labels[y, x]
            new_img[y, x] = clamp_pixel(new_img[y, x], 0, max_bit)
            y += 1
        x += 1
    return new_img
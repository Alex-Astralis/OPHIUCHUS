import numpy as np
import cv2
from Helpers import get_height, get_width, clamp_pixel
from Image_Resize import match_size
import math
import sys

# everything here assumes that passed in values match [0, max_value]

def im_add(image_1, image_2, resize_priority=None, max_value=255):
    re_image_1, re_image_2 = match_size(image_1, image_2, resize_priority)

    width = get_width(re_image_1)
    height = get_height(re_image_1)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)
    x = 0
    while x < width:
        y = 0
        while y < height:
            # add
            pixel = np.copy(re_image_1[y, x] + re_image_2[y, x])

            # clamp and save
            new_img[y, x] = clamp_pixel(pixel, 0, max_value)
            y += 1
        x += 1
    return new_img

def im_sub(image_1, image_2, resize_priority=None, max_value=255):
    re_image_1, re_image_2 = match_size(image_1, image_2, resize_priority)

    width = get_width(re_image_1)
    height = get_height(re_image_1)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)
    x = 0
    while x < width:
        y = 0
        while y < height:
            # subtract
            pixel = np.copy(re_image_1[y, x] - re_image_2[y, x])

            # clamp and save
            new_img[y, x] = clamp_pixel(pixel, 0, max_value)
            y += 1
        x += 1
    return new_img

def im_mult(image_1, image_2, resize_priority=None, max_value=255):
    re_image_1, re_image_2 = match_size(image_1, image_2, resize_priority)

    width = get_width(re_image_1)
    height = get_height(re_image_1)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)
    x = 0
    while x < width:
        y = 0
        while y < height:
            # multiply
            pixel = np.copy(re_image_1[y, x] * re_image_2[y, x])

            # clamp and save
            new_img[y, x] = clamp_pixel(pixel, 0, max_value)
            y += 1
        x += 1
    return new_img

def im_negative(image, max_value=255):
    width = get_width(image)
    height = get_height(image)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)

    x = 0
    while x < width:
        y = 0
        while y < height:
            new_img[y, x] = (max_value - 1) - image[y, x]
            y += 1
        x += 1
    return new_img

def im_log(image, constant, base, max_value=255):
    width = get_width(image)
    height = get_height(image)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)

    x = 0
    while x < width:
        y = 0
        while y < height:
            pixel = np.copy(image[y, x])
            pixel[0] = int(constant * math.log(float(1 + pixel[0]), base))
            pixel[1] = int(constant * math.log(float(1 + pixel[1]), base))
            pixel[2] = int(constant * math.log(float(1 + pixel[2]), base))

            new_img[y, x] = clamp_pixel(pixel, 0, max_value)
            y += 1
        x += 1
    return new_img

def im_gamma(image, constant, exponent, max_value=255):
    width = get_width(image)
    height = get_height(image)
    new_img = np.zeros([height, width, 3], dtype=np.uint8)

    x = 0
    while x < width:
        y = 0
        while y < height:
            pixel = np.copy(image[y, x])
            
            # adjust zero values
            offset = [0.0, 0.0, 0.0]
            if pixel[0] == 0:
                offset[0] = sys.float_info.epsilon
            if pixel[1] == 0:
                offset[1] = sys.float_info.epsilon
            if pixel[2] == 0:
                offset[2] = sys.float_info.epsilon

            pixel[0] = int(constant * math.pow(float(pixel[0]) + offset[0], exponent))
            pixel[1] = int(constant * math.pow(float(pixel[1]) + offset[1], exponent))
            pixel[2] = int(constant * math.pow(float(pixel[2]) + offset[2], exponent))

            new_img[y, x] = clamp_pixel(pixel, 0, max_value)
            y += 1
        x += 1
    return new_img
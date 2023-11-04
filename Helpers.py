import numpy as np

def get_height(img_dat):
    return img_dat.shape[0]

def get_width(img_dat):
    return img_dat.shape[1]

def is_outside_bit_range(img_dat, min_bit, max_bit):
    return np.max(img_dat) >= max_bit or np.min(img_dat) < min_bit

def clamp_pixel(pixel, min_bit, max_bit):
    # clamp to max - 1
    if pixel[0] >= max_bit:
        pixel[0] = max_bit - 1
    if pixel[1] >= max_bit:
        pixel[1] = max_bit - 1
    if pixel[2] >= max_bit:
        pixel[2] = max_bit - 1
    
    # clamp to min
    if pixel[0] < min_bit:
        pixel[0] = min_bit
    if pixel[1] < min_bit:
        pixel[1] = min_bit
    if pixel[2] < min_bit:
        pixel[2] = min_bit
    return pixel
import numpy as np

def get_height(image_dat):
    return image_dat.shape[0]

def get_width(image_dat):
    return image_dat.shape[1]


# assumes a range of [min_value, max_value]
def is_outside_bit_range(image_dat, min_value, max_value):
    return np.max(image_dat) > max_value or np.min(image_dat) < min_value

# assumes a range of [min_value, max_value]
def clamp_pixel(pixel, min_value, max_value):
    # clamp to max
    if pixel[0] > max_value:
        pixel[0] = max_value
    if pixel[1] > max_value:
        pixel[1] = max_value
    if pixel[2] > max_value:
        pixel[2] = max_value
    
    # clamp to min
    if pixel[0] < min_value:
        pixel[0] = min_value
    if pixel[1] < min_value:
        pixel[1] = min_value
    if pixel[2] < min_value:
        pixel[2] = min_value
    return pixel
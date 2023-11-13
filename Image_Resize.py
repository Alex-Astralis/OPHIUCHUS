import numpy as np
from Helpers import get_height, get_width
import math


def nearest_neighbor_im(image, width, height):
    scale_x = width / get_width(image)
    scale_y = height / get_height(image)

    resized_image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Calculate the corresponding position in the original image
            src_x = x / scale_x
            src_y = y / scale_y

            # Find the nearest pixels in the original image
            x1 = round(src_x)
            y1 = round(src_y)

            # Ensure the points are within the bounds of the original image
            x1 = min(max(x1, 0), get_width(image) - 1)
            y1 = min(max(y1, 0), get_height(image) - 1)

            # Perform nearest neighbor interpolation
            interpolated_value = image[y1, x1]

            # Set the pixel value in the resized image
            resized_image[y, x] = interpolated_value
    return resized_image

def bilinear_im(image, width, height):
    scale_x = width / get_width(image)
    scale_y = height / get_height(image)

    resized_image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Calculate the corresponding position in the original image
            src_x = x / scale_x
            src_y = y / scale_y

            # Find the four nearest pixels in the original image
            x1 = int(src_x)
            x2 = x1 + 1
            y1 = int(src_y)
            y2 = y1 + 1

            # Ensure the points are within the bounds of the original image
            x1 = min(max(x1, 0), get_width(image) - 1)
            x2 = min(max(x2, 0), get_width(image) - 1)
            y1 = min(max(y1, 0), get_height(image) - 1)
            y2 = min(max(y2, 0), get_height(image) - 1)

            # Calculate the interpolation weights
            weight_x = src_x - x1
            weight_y = src_y - y1

            # Perform bilinear interpolation
            interpolated_value = (
                    (1 - weight_x) * (1 - weight_y) * image[y1, x1] +
                    weight_x * (1 - weight_y) * image[y1, x2] +
                    (1 - weight_x) * weight_y * image[y2, x1] +
                    weight_x * weight_y * image[y2, x2]
            )

            # Set the pixel value in the resized image
            resized_image[y, x] = interpolated_value
    return resized_image

# resize one or both images to match a common size
# priority:
# - None: resize both to the max dimentions from both images
# - 1   : keep image 1 the same, resize image 2 to the same dimentions as image 1
# - 2   : keep image 2 the same, resize image 1 to the same dimentions as image 2
def match_size(image_1, image_2, priority=None):
    height_1 = get_height(image_1)
    width_1 = get_width(image_1)
    height_2 = get_height(image_2)
    width_2 = get_width(image_2)

    if priority == 1:           # keep image 1 the same
        width_3 = width_1
        height_3 = height_1
    elif priority == 2:         # keep image 2 the same
        width_3 = width_2
        height_3 = height_2
    else:                       # resize to max of width & height
        width_3 = max(width_1, width_2)
        height_3 = max(height_1, height_2)

    # resize image 1 if needed, otherwise return image 1
    if width_1 != width_3 or height_1 != height_3:
        resized_1 = bilinear_im(image_1, width_3, height_3)
    else:
        resized_1 = image_1

    # resize image 2 if needed, otherwise return image 2
    if width_2 != width_3 or height_2 != height_3:
        resized_2 = bilinear_im(image_2, width_3, height_3)
    else:
        resized_2 = image_2

    return resized_1, resized_2
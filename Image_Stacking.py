import numpy as np
import cv2

# ave = ave + (sample - ave) / (sample count)
def ave_stack(paths):
    reference_data = cv2.imread(paths[0])
    ref_width = get_width(reference_data)
    ref_height = get_height(reference_data)
    new_image = np.zeros([ref_height, ref_width, 3], dtype=np.float32) # allow floating point values untill after processing
    processed_images = 0
    for path in paths:
        curr_image_data = cv2.imread(path)
        curr_width = get_width(curr_image_data)
        curr_height = get_height(curr_image_data)
        
        y = 0
        while y < ref_height:
            curr_y = y
            if curr_y >= curr_height:
                curr_y = curr_height - 1
            x = 0
            while x < ref_width:
                curr_x = x
                if curr_x >= curr_width:
                    curr_x = curr_width - 1
                
                new_image[y, x] = new_image[y, x] + (curr_image_data[curr_y, curr_x] - new_image[y, x]) / (processed_images + 1)
                x += 1
            y += 1
        processed_images += 1

    new_image.astype(np.uint8)
    return new_image
import os
from astropy.io import fits
import glob
import cv2

def median_stack(glob_pattern: str, folder_path: str) -> np.ndarray:
    """
    Performs median stacking on a list of .fits images matched by a glob pattern,
    opening only one image at a time and using a running median.

    Parameters:
    glob_pattern (str): Glob pattern to match .fits image file names to be stacked.
    folder_path (str): Path to the folder containing the images.

    Returns:
    np.ndarray: The median stacked image.
    """
    image_filenames = glob.glob(os.path.join(folder_path, glob_pattern))

    # Placeholder for the running median
    running_median = None
    image_count = 0

    for filename in image_filenames:
        with fits.open(filename) as hdul:
            image_data = hdul[0].data

            if image_data is not None:
                # Normalize the image data to range between 0 and 255 and convert to uint8
                norm_image = cv2.normalize(image_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                norm_image = norm_image.astype(np.uint8)

                # Convert the normalized data to a 2D numpy array with channels (assuming grayscale to RGB)
                image = np.stack([norm_image] * 3, axis=-1)

                if running_median is None:
                    # Initialize the running median with the first image
                    running_median = image.astype(np.float64)
                else:
                    # Update the running median. This is a simplified and approximate way of calculating the median.
                    delta = image - running_median
                    sign = np.sign(delta)
                    running_median += sign * (1.0 / (image_count + 1))

                image_count += 1

    if running_median is not None:
        return running_median.astype(np.uint8)
    else:
        raise ValueError("No valid images found using the provided glob pattern.")


def sigma_stacking(glob_pattern, folder_path: str, sigma_threshold=2) -> np.ndarray:
    """
    Performs Sigma Stacking on a list of .fits files matching a glob pattern.

    :param folder_path:
    :param glob_pattern: Glob pattern to match .fits files.
    :param sigma_threshold: Number of standard deviations for sigma clipping.
    :return: A numpy array representing the stacked image.
    """
    # Generate list of .fits files
    image_filenames = glob.glob(os.path.join(folder_path, glob_pattern))

    # Initialize variables for mean, standard deviation, and count
    mean_image = None
    std_image = None
    count = 0

    for filename in image_filenames:
        # Load .fits file and convert to numpy array
        with fits.open(filename) as hdul:
            img_data = hdul[0].data
            if img_data.ndim == 2:  # Single channel (grayscale)
                img_data = np.stack((img_data,)*3, axis=-1)  # Duplicate to 3 channels
            img_array = img_data.astype(np.float64)

            # Initialize mean and std images if it's the first image
            if mean_image is None:
                mean_image = np.zeros_like(img_array, dtype=np.float64)
                std_image = np.zeros_like(img_array, dtype=np.float64)

            # Update mean and standard deviation
            mean_image = (mean_image * count + img_array) / (count + 1)
            std_image = np.sqrt(((std_image**2 * count) + (img_array - mean_image)**2) / (count + 1))

            count += 1

    # Sigma clipping and final mean calculation
    final_image = np.zeros_like(mean_image, dtype=np.float64)
    count = 0

    for filename in image_filenames:
        # Load .fits file again
        with fits.open(filename) as hdul:
            img_data = hdul[0].data
            if img_data.ndim == 2:  # Single channel (grayscale)
                img_data = np.stack((img_data,)*3, axis=-1)  # Duplicate to 3 channels
            img_array = img_data.astype(np.float64)

            # Apply sigma clipping
            mask = np.abs(img_array - mean_image) < (sigma_threshold * std_image)
            final_image = (final_image * count + img_array * mask) / (count + 1)
            count += 1

    # Return the final stacked image
    return final_image.astype(np.uint8)
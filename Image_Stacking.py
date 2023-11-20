import numpy as np
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

# Example usage:
# glob_pattern = "*.fits"
# folder_path = "/path/to/images"
# result = median_stack_fits_glob_single_load(glob_pattern, folder_path)
# Display or save the result as needed

from Helpers import get_height, get_width, clamp_pixel
import numpy as np
import os
from astropy.io import fits
import glob


def median_stack_fits_glob(glob_pattern: str, folder_path: str) -> np.ndarray:
    """
    Performs median stacking on a list of .fits images matched by a glob pattern,
    without holding all images in memory at once.

    Parameters:
    glob_pattern (str): Glob pattern to match .fits image file names to be stacked.
    folder_path (str): Path to the folder containing the images.

    Returns:
    np.ndarray: The median stacked image.
    """
    # Generate the list of filenames from the glob pattern
    image_filenames = glob.glob(os.path.join(folder_path, glob_pattern))

    # Initialize a variable to store the running median
    running_median = None
    image_count = 0

    for filename in image_filenames:
        # Read the .fits file
        with fits.open(filename) as hdul:
            image_data = hdul[0].data

            if image_data is not None:
                # Convert the .fits data to a 2D numpy array with channels (assuming grayscale to RGB)
                image = np.stack([image_data] * 3, axis=-1)

                if running_median is None:
                    # Initialize the running median with the first image
                    running_median = image.astype(np.float64)
                else:
                    # Incrementally update the median
                    delta = np.sign(image - running_median)
                    running_median += delta / (image_count + 1)

                image_count += 1

    if running_median is not None and image_count > 0:
        # Convert the running median back to uint8
        running_median_image = np.clip(running_median, 0, 255).astype(np.uint8)
        return running_median_image
    else:
        raise ValueError("No valid images found using the provided glob pattern.")
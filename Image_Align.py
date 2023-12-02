import cv2
import numpy as np
import glob
import os
from astropy.io import fits
import gc  # Garbage Collector interface

def align_images(input_directory):
    output_directory = f"{input_directory}_ALIGNED"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_list = glob.glob(f"{input_directory}/*.fits")

    if not file_list:
        print("No FITS files found in the directory.")
        return

    reference_image_data = fits.getdata(file_list[0])
    reference_image = np.array(reference_image_data, dtype=np.float32)

    for file in file_list:
        try:
            image_data = fits.getdata(file)
            image = np.array(image_data, dtype=np.float32)

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
            _, warp_matrix = cv2.findTransformECC(reference_image, image, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

            aligned_image = cv2.warpAffine(image, warp_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            output_file_path = os.path.join(output_directory, os.path.basename(file))
            fits.writeto(output_file_path, aligned_image, overwrite=True)
        except Exception as e:
            print(f"Error processing {file}: {e}")
        finally:
            # Free memory
            del image_data, image, aligned_image
            gc.collect()

    print(f"Aligned images saved in {output_directory}")

# align_images('path_to_your_input_directory')

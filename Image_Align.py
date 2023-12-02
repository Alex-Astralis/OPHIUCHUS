import cv2
import numpy as np
import glob
import os
from astropy.io import fits
import tkinter as tk
from tkinter import messagebox

def align_images(input_directory):
    # Set up the Tkinter root window and hide it
    print("Test LINE 1")
    root = tk.Tk()
    root.withdraw()

    # Create the output directory name by appending '_ALIGNED' to the input directory name
    output_directory = f"{input_directory}_ALIGNED"

    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all .fits files in the input directory
    file_list = glob.glob(f"{input_directory}/*.fits")

    # Ensure there are files to process
    if not file_list:
        messagebox.showerror("Error", "No FITS files found in the directory.")
        return

    # Read the first image to serve as a reference
    reference_image_data = fits.getdata(file_list[0])
    reference_image = np.array(reference_image_data, dtype=np.float32)

    # Iterate over files to align
    for file in file_list:
        # Read FITS file
        image_data = fits.getdata(file)
        image = np.array(image_data, dtype=np.float32)

        # Align images
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        _, warp_matrix = cv2.findTransformECC(reference_image, image, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

        aligned_image = cv2.warpAffine(image, warp_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Save the aligned image
        output_file_path = os.path.join(output_directory, os.path.basename(file))
        fits.writeto(output_file_path, aligned_image, overwrite=True)

    # Display completion message
    messagebox.showinfo("Completion", f"Aligned images saved in {output_directory}")

    # Destroy the root window
    root.destroy()

# Example usage
# align_images('path_to_your_input_directory')

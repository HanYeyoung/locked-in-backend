import time

from skimage import io
import keras_ocr
import math
import cv2
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, clear_border
import pytesseract
from scipy.ndimage import maximum_filter
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image


def remove_gaps(image, peak_multiplier=0.15, min_size_ratio=0.03, search_ratio=0.05):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded. Check the file path.")

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist_transform = distance_transform_edt(binary_image)

    # Calculate thresholds and peaks
    local_max_large = maximum_filter(dist_transform, size=75)  # Adjust size parameter as needed
    local_max_small = maximum_filter(dist_transform, size=10)
    dist_max = dist_transform.max()
    #peaks_global = dist_transform > dist_max * peak_multiplier
    peaks = (dist_transform == local_max_large) & (dist_transform == local_max_small) | (dist_transform > peak_multiplier * dist_max)
    #peaks = peaks_local | peaks_global
    # Create visualization with threshold line
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    dist_rgb = cv2.cvtColor(dist_normalized, cv2.COLOR_GRAY2RGB)
    dist_rgb[peaks] = [0, 0, 255]
    markers = cv2.connectedComponents(np.uint8(peaks))[1]
    inverted_dist_transform = -dist_transform

    labels = watershed(inverted_dist_transform, markers, mask=binary_image)
    cleared_labels = clear_border(labels)

    # Remove small components
    min_size = int((image.shape[0] * image.shape[1]) * (min_size_ratio ** 2))
    final_labels = remove_small_objects(cleared_labels, min_size=min_size)

    # Create a color image to draw the contours
    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Get unique labels excluding the background (0)
    unique_labels = np.unique(final_labels)
    unique_labels = unique_labels[unique_labels > 0]

    # Generate distinct colors for each label
    colors = np.random.randint(50, 255, size=(len(unique_labels), 3))

    # Draw contours for each unique label
    for i, label in enumerate(unique_labels):
        # Find contours for the current label
        contours, _ = cv2.findContours((final_labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # merge the contours of rooms into one contour, this results in overlapping of contour
        # merge_distance of 8 is good for memory union floor plan
        merged_contours = merge_close_contours(contours, image.shape[0], image.shape[1], merge_distance= 8 )

        # Draw contours with the corresponding color
        for contour in  merged_contours:
            cv2.drawContours(contour_image, [contour], -1, colors[i].tolist(), 2)

    return contour_image






# Function to merge contours
def merge_close_contours(contours, height, width, merge_distance= 1):
    merged_contours = []
    contour_mask = np.zeros((height, width), dtype=np.uint8)  # Single-channel mask

    for contour in contours:
        # Draw each contour as a filled shape on the single-channel mask
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Dilate the contours to connect nearby contours
    kernel = np.ones((merge_distance, merge_distance), np.uint8)
    dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)

    # Find new contours on the dilated mask
    merged_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return merged_contours

# dilate the image to turn the overlapping contours into one thick line
# then apply skeletonization to ensure all lines have the same thickness
def merge_and_clean_lines(image, kernel_size=5, merge_distance=5):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    # Dilate to merge close contour
    kernel = np.ones((merge_distance, merge_distance), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # cv2.imshow("Dilated Image", dilated_image)
    # cv2.waitKey(0)

    _, binary_img = cv2.threshold(dilated_image, 127, 255, cv2.THRESH_BINARY)
    # Skeletonize the binary image to reduce all lines to single-pixel width
    skeleton = skeletonize(binary_img // 255)
    skeleton = skeleton.astype(np.uint8) * 255

    return skeleton

def post_processing(image_path):
    # cropped_img = extract_floor_plan(image_path)
    # cv2.imshow("room", cropped_img)
    # cv2.waitKey(0)
    img = cv2.imread(image_path)
    rooms = remove_gaps(img)
    refined_rooms = merge_and_clean_lines(rooms)
    directory, filename = os.path.split("opt1.jpg")
    output_path = os.path.join(directory, f"removed_text_{filename}")
    cv2.imwrite(output_path, refined_rooms)


if __name__ == "__main__":

    post_processing("opt1.jpg")




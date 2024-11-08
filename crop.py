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
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def remove_text(image, conf_threshold=60, min_area=100, max_area=10000):
    print("removing text...")

    #get image data
    ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    #add rectangles
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(len(ocr_results['text'])):
        conf = int(ocr_results['conf'][i])
        if conf > conf_threshold:
            x, y, w, h = ocr_results['left'][i], ocr_results['top'][i], ocr_results['width'][i], ocr_results['height'][
                i]
            area = w * h
            if min_area < area < max_area:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("blocked out text", mask)
    cv2.waitKey(0)
    # invert mask to create inpaint area
    img_for_inpaint = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    result = cv2.inpaint(img_for_inpaint, mask, 2, cv2.INPAINT_TELEA)

    return result

def extract_floor_plan(image_path, mp = 0.1):
    print("cropping...")
    image = cv2.imread(image_path)
    # cv2.imshow("image before text preprocessing", image)
    cv2.waitKey(0)

    #convert to grayscale
    image = remove_text(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("image after text preprocessing", gray)
    cv2.waitKey(0)

    #binary thresholding, 70-255 seems to work well
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Perform morphological closing to fill small gaps in the outer contour
    height, width = gray.shape
    kernel = np.ones((int(height*mp), int(width*mp)), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed binary image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour - usually the floor plan especially after morphology
    main_contour = max(contours, key=cv2.contourArea)

    #fill in the main contour as a mask
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)

    # Apply the mask to the original floor plan image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Get the bounding rectangle of the main contour and crop
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped = result[y:y + h, x:x + w]

    # cv2.imshow("cropped image", cropped)
    cv2.waitKey(0)
    return cropped


def remove_gaps(image_path, peak_multiplier=0.15, min_size_ratio=0.03, search_ratio=0.05):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded. Check the file path.")

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist_transform = distance_transform_edt(binary_image)

    # Calculate thresholds and peaks

    local_max_large = maximum_filter(dist_transform, size=100)  # Adjust size parameter as needed
    local_max_small = maximum_filter(dist_transform, size=20)
    dist_max = dist_transform.max()
    #peaks_global = dist_transform > dist_max * peak_multiplier
    peaks = (dist_transform == local_max_large) & (dist_transform == local_max_small) | (dist_transform > peak_multiplier * dist_max)
    #peaks = peaks_local | peaks_global
    # Create visualization with threshold line
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    dist_rgb = cv2.cvtColor(dist_normalized, cv2.COLOR_GRAY2RGB)

    # cv2.imshow('Distance Transform', dist_rgb)
    cv2.waitKey(0)

    dist_rgb[peaks] = [0, 0, 255]
    # cv2.imshow('Distance Transform with Peaks Identified', dist_rgb)
    cv2.waitKey(0)

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

    cv2.imshow("Dilated Image", dilated_image)
    cv2.waitKey(0)

    _, binary_img = cv2.threshold(dilated_image, 127, 255, cv2.THRESH_BINARY)
    # Skeletonize the binary image to reduce all lines to single-pixel width
    skeleton = skeletonize(binary_img // 255)
    skeleton = skeleton.astype(np.uint8) * 255

    return skeleton

if __name__ == "__main__":

    rooms = remove_gaps("cropped_floor_plan.jpg")
    cv2.imshow("identified rooms", rooms)
    cv2.waitKey(0)
    refined_rooms = merge_and_clean_lines(rooms)

    cv2.imshow("room", refined_rooms)
    cv2.waitKey(0)


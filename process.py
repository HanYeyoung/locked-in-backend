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
import matplotlib.pyplot as plt
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def remove_text(self, image, conf_threshold=40, min_area=20, max_area=50000):
    """
    Enhanced text removal function using multiple detection methods
    """
    # Create a copy and get grayscale
    working_image = image.copy()
    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

    # Create mask for text regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Method 1: Basic Thresholding for Text Detection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all connected components (potential text regions)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)

    # Filter components by size
    for i in range(1, num_labels):  # Skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Filter by aspect ratio (text usually has specific aspect ratios)
            aspect_ratio = w / h if h != 0 else 0
            if 0.2 < aspect_ratio < 15:  # Typical text aspect ratio range
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)

    # Method 2: MSER for text region detection
    mser = cv2.MSER_create(
        _min_area=100,
        _max_area=5000,
        _delta=5,
        _max_variation=0.5
    )

    # Detect regions
    regions, _ = mser.detectRegions(gray)

    # Convert regions to rectangles and add to mask
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        area = w * h
        if min_area < area < max_area:
            aspect_ratio = w / h if h != 0 else 0
            if 0.2 < aspect_ratio < 15:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)

    # Method 3: Modified Tesseract approach with different configs
    custom_config = r'--oem 3 --psm 6'  # Assume uniform text on uniform background
    ocr_results = pytesseract.image_to_data(working_image, output_type=pytesseract.Output.DICT, config=custom_config)

    for i in range(len(ocr_results['text'])):
        conf = int(ocr_results['conf'][i])
        text = ocr_results['text'][i].strip()

        if conf > conf_threshold and text:
            x, y, w, h = (ocr_results['left'][i], ocr_results['top'][i],
                          ocr_results['width'][i], ocr_results['height'][i])
            area = w * h
            if min_area < area < max_area:
                padding = int(min(w, h) * 0.2)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)

    # Enhance the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Debug: Save the mask (optional)
    # cv2.imwrite("text_mask.jpg", mask)

    # Apply inpainting
    img_for_inpaint = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    result = cv2.inpaint(img_for_inpaint, mask, 3, cv2.INPAINT_TELEA)

    return result  # Return both result and mask for debugging

def extract_floor_plan(image_path, mp = 0.1):
    print("cropping...")
    image = cv2.imread(image_path)
    cv2.imshow("image before text preprocessing", image)
    cv2.waitKey(0)
    #convert to grayscale
    image = remove_text(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("image after text preprocessing", gray)
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

    cv2.imshow("cropped image", cropped)
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

    cv2.imshow('Distance Transform', dist_rgb)
    cv2.waitKey(0)

    dist_rgb[peaks] = [0, 0, 255]
    cv2.imshow('Distance Transform with Peaks Identified', dist_rgb)
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

        # Draw contours with the corresponding color
        for contour in contours:
            cv2.drawContours(contour_image, [contour], -1, colors[i].tolist(), 2)

    return contour_image

if __name__ == "__main__":
    path = "MU_2.jpg"
#    image_path = os.path.join(os.path.join("floorplans", "raw"), path)
#    cropped = extract_floor_plan(image_path)
#    output_path_cropped = os.path.join(os.path.join("floorplans", "cropped"), path)
#    cv2.imwrite(output_path_cropped, cropped)
    #  for path in os.listdir(os.path.join(os.path.join("floorplans", "raw"))):
    #for path in os.path.join(os.path.join(os.path.join("floorplans", "raw")), "MU_1"):
    image_path = os.path.join(os.path.join("../floorplans", "raw"), path)
    cropped = extract_floor_plan(image_path)
    print("cropped: " + path)
    output_path_cropped = os.path.join(os.path.join("../floorplans", "cropped"), path)

    cv2.imwrite(output_path_cropped, cropped)
    while not (os.path.exists(output_path_cropped)):
        time.sleep(1)
    rooms = remove_gaps(output_path_cropped)
    print("indentified: " + path)
    output_path_rooms = os.path.join(os.path.join("../floorplans", "rooms"), path)
    cv2.imwrite(output_path_rooms, rooms)
    cv2.imshow("identified rooms", rooms)
    cv2.waitKey(0)

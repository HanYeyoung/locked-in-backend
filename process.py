import time
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.segmentation import watershed, clear_border
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt
import keras_ocr
import random


# Utility Functions
def generate_color_palette(num_colors):
    """Generate a consistent colour palette."""
    np.random.seed(42)  # For reproducibility
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_colors)]


def midpoint(x1, y1, x2, y2):
    """Calculate the midpoint of two points."""
    return (x1 + x2) // 2, (y1 + y2) // 2


# Text Removal
def remove_text(image, pipeline, conf_threshold=60):
    """Remove text from the image using OCR pipeline."""
    print("Removing text...")
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect text with OCR pipeline
    prediction_groups = pipeline.recognize([img_rgb])
    mask = np.zeros(image.shape[:2], dtype="uint8")

    # Draw mask over detected text
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        thickness = int(np.hypot(x2 - x1, y2 - y1))
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)

    # Inpaint image to remove text
    result = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    return result


# Floor Plan Extraction
def extract_floor_plan(image_path, pipeline, mp=0.1):
    """Extract the main floor plan by removing text and background."""
    print("Cropping...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not loaded. Check the file path.")

    # Remove text from the image
    img = remove_text(img, pipeline)

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Perform morphological closing to fill gaps
    height, width = gray.shape
    kernel = np.ones((int(height * mp), int(width * mp)), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    # Mask and crop the main floor plan
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped = result[y:y + h, x:x + w]

    return cropped


# Gap Removal
def remove_gaps(image, min_size=500):
    """Remove small gaps and extract clean contours."""
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_transform = distance_transform_edt(binary_image)
    peaks = dist_transform > (0.1 * dist_transform.max())
    markers = cv2.connectedComponents(np.uint8(peaks))[1]
    inverted_dist_transform = -dist_transform

    # Watershed segmentation
    labels = watershed(inverted_dist_transform, markers, mask=binary_image)
    cleared_labels = clear_border(labels)
    final_labels = remove_small_objects(cleared_labels, min_size=min_size)

    # Generate contour image
    contour_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(final_labels[final_labels > 0])
    colors = generate_color_palette(len(unique_labels))

    # Draw contours
    for i, label in enumerate(unique_labels):
        contours, _ = cv2.findContours((final_labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(contour_image, [contour], -1, colors[i], 2)

    return contour_image


# Processing Pipeline
def process_image(image_path, output_dir, pipeline):
    """Process an image through the full pipeline."""
    print(f"Processing {image_path}...")
    cropped = extract_floor_plan(image_path, pipeline)
    gaps_removed = remove_gaps(cropped)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_cropped.png"), cropped)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rooms.png"), gaps_removed)


if __name__ == "__main__":
    input_dir = "floorplans/raw"
    output_dir = "floorplans/output"
    image_paths = [os.path.join(input_dir, path) for path in os.listdir(input_dir) if path.endswith(('.png', '.jpg'))]

    # Initialise OCR pipeline outside the loop for efficiency
    pipeline = keras_ocr.pipeline.Pipeline()

    print("Starting batch processing...")
    for image_path in tqdm(image_paths, desc="Processing Images"):
        process_image(image_path, output_dir, pipeline)

    print("Processing completed.")

import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, clear_border
import keras_ocr
import random


# Helper Functions
def generate_color_palette(num_colors):
    """Generate a consistent colour palette."""
    np.random.seed(42)  # For reproducibility
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_colors)]


def midpoint(x1, y1, x2, y2):
    """Calculate the midpoint of two points."""
    return (x1 + x2) // 2, (y1 + y2) // 2


# Core Image Processing Functions
def remove_gaps(image, min_size=500):
    """Remove small gaps and extract clean contours."""
    # Threshold and distance transform
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


def extract_floor_plan(image_path, pipeline):
    """Extract the main floor plan by removing text and background."""
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not loaded. Check the file path.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect text
    prediction_groups = pipeline.recognize([img_rgb])
    mask = np.zeros(img.shape[:2], dtype="uint8")

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
    img_inpainted = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    # Extract main contour
    gray = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    # Mask and crop the main floor plan
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    result = cv2.bitwise_and(img_inpainted, img_inpainted, mask=mask)
    x, y, w, h = cv2.boundingRect(main_contour)
    return result[y:y + h, x:x + w]


def identify_rooms(image, min_room_area=350):
    """Identify and label rooms based on contours."""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Extract room contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    room_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_room_area]

    # Draw and label rooms
    result = image.copy()
    colors = generate_color_palette(len(room_contours))
    for i, cnt in enumerate(room_contours):
        cv2.drawContours(result, [cnt], 0, colors[i], 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result, f"{i + 1}", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    return result


# Unified Processing Pipeline
def process_image(image_path, output_dir, pipeline):
    """Process an image through the full pipeline."""
    cropped = extract_floor_plan(image_path, pipeline)
    gaps_removed = remove_gaps(cropped)
    room_labels = identify_rooms(gaps_removed)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_cropped.png"), cropped)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gaps_removed.png"), gaps_removed)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rooms.png"), room_labels)


# Batch Processing
if __name__ == "__main__":
    input_dir = "floorplans/raw"
    output_dir = "floorplans/output"
    image_paths = [os.path.join(input_dir, path) for path in os.listdir(input_dir) if path.endswith(('.png', '.jpg'))]

    # Initialise OCR pipeline outside the loop for efficiency
    pipeline = keras_ocr.pipeline.Pipeline()

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda p: process_image(p, output_dir, pipeline), image_paths), total=len(image_paths)))
    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")

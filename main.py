import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, clear_border
import keras_ocr

def midpoint(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

# Helper function for generating colours from a fixed palette
def generate_color_palette(num_colors):
    np.random.seed(42)  # For reproducibility
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_colors)]

# Optimised remove_gaps function
def remove_gaps(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not loaded. Check the file path.")

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_transform = distance_transform_edt(binary_image)

    peaks = dist_transform > (0.1 * dist_transform.max())
    markers = cv2.connectedComponents(np.uint8(peaks))[1]
    inverted_dist_transform = -dist_transform
    labels = watershed(inverted_dist_transform, markers, mask=binary_image)
    cleared_labels = clear_border(labels)

    min_size = 500
    final_labels = remove_small_objects(cleared_labels, min_size=min_size)

    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(final_labels[final_labels > 0])
    colors = generate_color_palette(len(unique_labels))

    for i, label in enumerate(unique_labels):
        contours, _ = cv2.findContours((final_labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(contour_image, [contour], -1, colors[i], 2)
    
    return contour_image

# Optimised extract_floor_plan function
def extract_floor_plan(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prediction_groups = pipeline.recognize([img_rgb])
    
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
        thickness = int(np.hypot(x2 - x1, y2 - y1))
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
    
    img_inpainted = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    gray = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    result = cv2.bitwise_and(img_inpainted, img_inpainted, mask=mask)
    
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped = result[y:y + h, x:x + w]
    
    return cropped

# Optimised identify_rooms function
def identify_rooms(image_path):
    start_time = time.time()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_room_area = 350
    room_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_room_area]
    result = img.copy()
    colors = generate_color_palette(len(room_contours))

    for i, cnt in enumerate(room_contours):
        cv2.drawContours(result, [cnt], 0, colors[i], 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result, f"{i + 1}", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    end_time = time.time()
    print(f"Processed {image_path} in {end_time - start_time:.2f} seconds")

    return result

# Batch processing with parallelism
def process_image(image_path):
    cropped = extract_floor_plan(image_path)
    output_path_cropped = os.path.join("floorplans", "cropped", os.path.basename(image_path))
    cv2.imwrite(output_path_cropped, cropped)
    
    rooms = remove_gaps(output_path_cropped)
    output_path_rooms = os.path.join("floorplans", "rooms", os.path.basename(image_path))
    cv2.imwrite(output_path_rooms, rooms)

if __name__ == "__main__":
    image_paths = [os.path.join("floorplans", "raw", path) for path in os.listdir("floorplans/raw")]
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_image, image_paths)

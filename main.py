import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, clear_border
import keras_ocr
import math


def remove_gaps(image_path):
    image = cv2.imread('empty.jpg', cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded. Check the file path.")

    # Apply binary thresholding to create a binary image (threshold value may need adjustment)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Compute the distance transform
    dist_transform = distance_transform_edt(binary_image)

    # Identify the room centers by finding peaks in the distance transform
    peaks = dist_transform > (0.2 * dist_transform.max())  # Adjust multiplier if needed

    # Use peaks as markers for the watershed algorithm
    markers = cv2.connectedComponents(np.uint8(peaks))[1]

    # Invert the distance transform for the watershed transform
    inverted_dist_transform = -dist_transform

    # Apply watershed segmentation
    labels = watershed(inverted_dist_transform, markers, mask=binary_image)

    # Remove components touching the border
    cleared_labels = clear_border(labels)

    # Remove small components (e.g., noise) with a size threshold (e.g., 1000)
    min_size = 500  # Adjust this value based on your image's scale
    final_labels = remove_small_objects(cleared_labels, min_size=min_size)

    # Create a color image to draw the contours
    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Get unique labels excluding the background (0)
    unique_labels = np.unique(final_labels)
    unique_labels = unique_labels[unique_labels > 0]

    # Generate distinct colors for each label
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))

    # Draw contours for each unique label
    for i, label in enumerate(unique_labels):
        # Find contours for the current label
        contours, _ = cv2.findContours((final_labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours with the corresponding color
        for contour in contours:
            cv2.drawContours(contour_image, [contour], -1, colors[i].tolist(), 2)  # Draw with unique color

    return contour_image
def extract_floor_plan(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    main_contour = max(contours, key=cv2.contourArea)

    # Create a mask from the main contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Get the bounding rectangle of the main contour
    x, y, w, h = cv2.boundingRect(main_contour)

    # Crop the image to the bounding rectangle
    cropped = result[y:y + h, x:x + w]

    return cropped

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
def identify_rooms(image_path):
    start_time = time.time()

    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    print(f"Image shape: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Thresholding image")
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed

    print("Finding contours")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours")
    min_room_area = 350  # Adjust this value based on the scale of your floor plan
    room_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_room_area]
    print(f"Filtered to {len(room_contours)} room contours")

    print("Drawing contours and numbering rooms")
    result = img.copy()
    for i, cnt in enumerate(room_contours):
        color = generate_random_color()
        cv2.drawContours(result, [cnt], 0, color, 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result, f"{i + 1}", (cX - 20, cY),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color, 2)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    return result

def inpaint_text(img_path, pipeline):
    # read the image
    img = keras_ocr.tools.read(img_path)

    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([img])

    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        #For the line thickness, we will calculate the length of the line between
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                 thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return(inpainted_img)

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

# Usage
if __name__ == "__main__":
    pipeline = keras_ocr.pipeline.Pipeline()
    img_text_removed = inpaint_text('floorplans/test/hard.jpg', pipeline)
    plt.imshow(img_text_removed)
    cv2.imwrite('text_removed_image.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))

    image = remove_gaps("floorplans/test/empty.jpg")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('Segmented Rooms Contours in Different Colors')
    plt.axis('off')
    plt.show()

    # for path in os.listdir(os.path.join(os.path.join("floorplans", "raw"))):
    #     image_path = os.path.join(os.path.join("floorplans", "raw"), path)
    #     cropped = extract_floor_plan(image_path)
    #     output_path_cropped = os.path.join(os.path.join("floorplans", "cropped"), path)
    #     cv2.imwrite(output_path_cropped, cropped)
    #     #cv2.imshow('Cropped Floor Plan', cropped)
    #     #cv2.waitKey(0)
    #     #cv2.destroyAllWindows()
    #     count = 0
    #     while not (os.path.exists(output_path_cropped) or count > 15):
    #         count += 1
    #         time.sleep(1)
    #     count = 0
    #     rooms = identify_rooms(output_path_cropped)
    #     output_path_rooms = os.path.join(os.path.join("floorplans", "rooms"), path)
    #     cv2.imwrite(output_path_rooms, rooms)
    #     cv2.imshow('Room Identification', rooms)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
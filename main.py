import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
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

# Usage
if __name__ == "__main__":
    for path in os.listdir(os.path.join(os.path.join("floorplans", "raw"))):
        image_path = os.path.join(os.path.join("floorplans", "raw"), path)
        cropped = extract_floor_plan(image_path)
        output_path_cropped = os.path.join(os.path.join("floorplans", "cropped"), path)
        cv2.imwrite(output_path_cropped, cropped)
        #cv2.imshow('Cropped Floor Plan', cropped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        count = 0
        while not (os.path.exists(output_path_cropped) or count > 15):
            count += 1
            time.sleep(1)
        count = 0
        rooms = identify_rooms(output_path_cropped)
        output_path_rooms = os.path.join(os.path.join("floorplans", "rooms"), path)
        cv2.imwrite(output_path_rooms, rooms)
        cv2.imshow('Room Identification', rooms)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
import cv2
import numpy as np
import svgwrite
import random
import os
import time

def generate_random_color():
    """Generates a random color in hex format."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def extract_floor_plan(image_path):
    """Extracts the main floor plan contour from the image by removing the background."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
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

    return cropped, (x, y)

def identify_rooms_to_svg(image_path, output_svg_path):
    """
    Detects the rooms (contours) from the floor plan image and saves them as an SVG file.
    Rooms are drawn as polygons and labeled with their room number.
    """
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter based on a minimum room area
    min_room_area = 350
    room_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_room_area]

    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny')

    # Loop through the contours and write them as paths in the SVG
    for i, cnt in enumerate(room_contours):
        path_data = []
        for point in cnt:
            # Extract (x, y) and ensure they are integers
            x, y = map(int, point[0])  # Ensuring x, y are integers
            path_data.append((x, y))  # Append as a tuple (x, y)

        # Debug: Print the points to make sure they are valid
        print(f"Room {i + 1} path data: {path_data}")

        # Ensure path_data contains valid coordinates before adding to SVG
        if path_data:
            # Create a polygon for each room
            dwg.add(dwg.polygon(points=path_data, fill=generate_random_color(), stroke='black', stroke_width=1))

            # Compute centroid for room number placement
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Add text label to the SVG
                dwg.add(dwg.text(f'Room {i + 1}', insert=(cX, cY), fill='black'))

    # Save the SVG file
    dwg.save()
    print(f"Saved SVG to {output_svg_path}")

# Main usage
if __name__ == "__main__":
    input_folder = os.path.join("floorplans", "raw")
    output_svg_folder = os.path.join("floorplans", "svg")

    # Create output folder if not exists
    if not os.path.exists(output_svg_folder):
        os.makedirs(output_svg_folder)

    for path in os.listdir(input_folder):
        image_path = os.path.join(input_folder, path)
        output_svg_path = os.path.join(output_svg_folder, path.replace(".jpg", ".svg"))
        
        cropped, _ = extract_floor_plan(image_path)
        temp_cropped_path = "temp_cropped.jpg"
        cv2.imwrite(temp_cropped_path, cropped)
        
        identify_rooms_to_svg(temp_cropped_path, output_svg_path)
        os.remove(temp_cropped_path)  # Clean up the temporary cropped image

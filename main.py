import cv2
import numpy as np
import json
import os
import random

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

def pixel_to_geo(x_pixel, y_pixel, origin, lon_per_pixel, lat_per_pixel):
    """Converts pixel coordinates to geographic (longitude, latitude) coordinates."""
    lon_top_left, lat_top_left = origin

    # Calculate geographic coordinates
    lon = lon_top_left + x_pixel * lon_per_pixel
    lat = lat_top_left - y_pixel * lat_per_pixel  # Subtract, because latitude decreases as y increases

    return lon, lat

def contours_to_geojson(contours, origin, lon_per_pixel, lat_per_pixel):
    """Convert contours to GeoJSON format."""
    features = []
    for i, cnt in enumerate(contours):
        coordinates = []
        for point in cnt:
            x_pixel, y_pixel = point[0]
            lon, lat = pixel_to_geo(x_pixel, y_pixel, origin, lon_per_pixel, lat_per_pixel)
            coordinates.append([lon, lat])

        # Add each room as a GeoJSON feature
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            },
            "properties": {
                "room_number": i + 1
            }
        })

    # Create a GeoJSON feature collection
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson_data

def save_geojson(geojson_data, output_path):
    """Saves the GeoJSON data to a file."""
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=4)

def identify_rooms_to_geojson(image_path, output_geojson_path, origin, lon_per_pixel, lat_per_pixel):
    """
    Detects the rooms (contours) from the floor plan image and saves them as a GeoJSON file.
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

    # Convert contours to GeoJSON
    geojson_data = contours_to_geojson(room_contours, origin, lon_per_pixel, lat_per_pixel)

    # Save the GeoJSON file
    save_geojson(geojson_data, output_geojson_path)
    print(f"Saved GeoJSON to {output_geojson_path}")

# Main usage
if __name__ == "__main__":
    input_folder = os.path.join("floorplans", "raw")
    output_geojson_folder = os.path.join("floorplans", "geojson")

    # Create output folder if not exists
    if not os.path.exists(output_geojson_folder):
        os.makedirs(output_geojson_folder)

    # Define the geographic bounds for the image
    origin = (-90.0, 45.0)  # Top-left corner coordinates (longitude, latitude)
    lon_bottom_right, lat_bottom_right = -89.9, 44.9  # Bottom-right corner coordinates

    # Scaling factors based on the image dimensions and geographic distance
    image_width = 1000  # Replace with actual image width
    image_height = 800  # Replace with actual image height

    # Calculate scaling factors (degrees per pixel)
    geo_width = lon_bottom_right - origin[0]
    geo_height = origin[1] - lat_bottom_right
    lon_per_pixel = geo_width / image_width
    lat_per_pixel = geo_height / image_height

    # Process each image in the input folder
    for path in os.listdir(input_folder):
        image_path = os.path.join(input_folder, path)
        output_geojson_path = os.path.join(output_geojson_folder, path.replace(".jpg", ".geojson"))

        cropped, _ = extract_floor_plan(image_path)
        temp_cropped_path = "temp_cropped.jpg"
        cv2.imwrite(temp_cropped_path, cropped)
        
        identify_rooms_to_geojson(temp_cropped_path, output_geojson_path, origin, lon_per_pixel, lat_per_pixel)
        os.remove(temp_cropped_path)  # Clean up the temporary cropped image

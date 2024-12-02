import os
import cv2
import numpy as np
from processors import FloorPlanProcessor


def test_processor():
    # Initialize processor
    processor = FloorPlanProcessor()

    # Test coordinates
    test_coordinates = {
        'min_lat': 40.7128,
        'max_lat': 40.7138,
        'min_long': -74.0060,
        'max_long': -74.0050
    }

    # Test paths
    input_path = os.path.join("..", "floorplans", "raw", "MU_5.jpg")
    output_dir = os.path.join("..", "floorplans", "test_output")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the test image
    with open(input_path, 'rb') as f:
        image_data = f.read()

    # Process the image
    result = processor.process_image(image_data, test_coordinates)

    if result['status'] == 'success':
        # Save processed images
        processed_path = os.path.join(output_dir, "processed.jpg")
        rooms_path = os.path.join(output_dir, "rooms.jpg")
        geojson_path = os.path.join(output_dir, "rooms.geojson")

        # Save processed image
        with open(processed_path, 'wb') as f:
            f.write(result['processed_image'])

        # Save room image
        with open(rooms_path, 'wb') as f:
            f.write(result['room_image'])

        # Save GeoJSON
        with open(geojson_path, 'w') as f:
            import json
            json.dump(result['geojson'], f, indent=2)

        print(f"Test successful!")
        print(f"Processed image saved to: {processed_path}")
        print(f"Rooms image saved to: {rooms_path}")
        print(f"GeoJSON saved to: {geojson_path}")

    else:
        print(f"Processing failed: {result['error']}")



if __name__ == "__main__":
    test_processor()
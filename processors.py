# processors.py
import cv2
import numpy as np
from typing import Tuple, Dict, Any
import pytesseract
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, clear_border
from scipy.ndimage import distance_transform_edt, maximum_filter

class FloorPlanProcessor:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

    def process_image(self, image_data: bytes, coordinates: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing function that handles the complete pipeline"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise Exception("Failed to decode image")

            # Add debug print
            print("Image shape:", image.shape)

            # Process the image through your pipeline
            cropped_image = self.extract_floor_plan(image)
            processed_image, room_contours = self.remove_gaps(cropped_image)

            # Convert processed images back to bytes
            _, processed_buffer = cv2.imencode('.jpg', cropped_image)
            _, rooms_buffer = cv2.imencode('.jpg', processed_image)

            # Create GeoJSON from contours
            geojson = self.create_geojson(room_contours, coordinates)

            return {
                "status": "success",
                "geojson": geojson,
                "processed_image": processed_buffer.tobytes(),
                "room_image": rooms_buffer.tobytes()
            }

        except Exception as e:
            print(f"Processing error: {str(e)}")  # Add debug print
            return {
                "status": "error",
                "error": str(e)
            }

    def remove_text(self, image):
        """
        Text removal using EasyOCR with improved inpainting
        """
        import easyocr

        # Store original image
        original = image.copy()

        # Convert to grayscale for background analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])

        # Create mask for text regions
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Detect text regions
        results = reader.readtext(image)

        # Process each detected text region
        for box in results:
            points = box[0]
            pts = np.array(points, dtype=np.int32)

            # Increase padding for better coverage
            padding = 0
            min_x = max(0, np.min(pts[:, 0]) - padding)
            min_y = max(0, np.min(pts[:, 1]) - padding)
            max_x = min(image.shape[1], np.max(pts[:, 0]) + padding)
            max_y = min(image.shape[0], np.max(pts[:, 1]) + padding)

            # Draw filled rectangle
            cv2.rectangle(text_mask, (int(min_x), int(min_y)), (int(max_x), int(max_y)), 255, -1)

        # First pass: larger radius for overall structure
        result = cv2.inpaint(image, text_mask, 5, cv2.INPAINT_NS)

        # Second pass: smaller radius for details
        result = cv2.inpaint(result, text_mask, 3, cv2.INPAINT_TELEA)

        return result

    def extract_floor_plan(self, image):
        """Main function to extract and crop floor plan"""
        # Remove text first
        image = self.remove_text(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initial binary threshold
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        height, width = gray.shape

        # Initial small closing to connect very close components
        small_kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, small_kernel, iterations=2)

        # Find contours to analyze gaps
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find minimum distance from each contour to image border
        max_gap = 0
        for contour in contours:
            for point in contour[:, 0]:
                x, y = point
                # Distance to nearest border
                dist_to_border = min(x, y, width - x, height - y)
                max_gap = max(max_gap, dist_to_border)

        # Calculate adaptive mp based on largest gap
        mp = (max_gap / min(height, width)) * 1.2  # Add 20% margin
        mp = np.clip(mp, 0.01, 0.04)  # More conservative upper bound

        # Flood fill from borders to identify and remove background
        mask = np.zeros((height + 2, width + 2), np.uint8)
        flood_fill = closed.copy()
        cv2.floodFill(flood_fill, mask, (0, 0), 255)

        # Invert flood fill result to get the floor plan
        flood_fill_inv = cv2.bitwise_not(flood_fill)

        # Combine with original binary
        floor_plan = closed | flood_fill_inv

        # Do closing with adaptive mp value
        kernel = np.ones((int(height * mp), int(width * mp)), np.uint8)
        floor_plan = cv2.morphologyEx(floor_plan, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(floor_plan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise Exception("No contours found")

        # Filter by minimum area
        min_area = (height * width) * 0.01
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid_contours:
            raise Exception("No valid contours found after area filtering")

        main_contour = max(valid_contours, key=cv2.contourArea)

        # Create mask and apply it
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [main_contour], 0, 255, -1)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Crop to bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        cropped = result[y:y + h, x:x + w]

        return cropped

    def remove_gaps(self, image, peak_multiplier=0.15, min_size_ratio=0.03, search_ratio=0.05):
        """Your existing remove_gaps function modified for server use"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist_transform = distance_transform_edt(binary_image)

        local_max_large = maximum_filter(dist_transform, size=60)
        local_max_small = maximum_filter(dist_transform, size=20)
        dist_max = dist_transform.max()
        peaks = ((dist_transform == local_max_large) &
                 (dist_transform == local_max_small) |
                 (dist_transform > peak_multiplier * dist_max))

        markers = cv2.connectedComponents(np.uint8(peaks))[1]
        inverted_dist_transform = -dist_transform
        labels = watershed(inverted_dist_transform, markers, mask=binary_image)
        cleared_labels = clear_border(labels)

        min_border_size = image.shape[0] * image.shape[1] * 0.0001  # 1% of image size
        border_labels = np.unique(labels[0, :])  # Get labels touching top border
        border_labels = np.append(border_labels, np.unique(labels[-1, :]))  # bottom border
        border_labels = np.append(border_labels, np.unique(labels[:, 0]))  # left border
        border_labels = np.append(border_labels, np.unique(labels[:, -1]))

        cleared_labels = labels.copy()
        for label in np.unique(border_labels):
            if np.sum(labels == label) < min_border_size:
                cleared_labels[labels == label] = 0

        final_labels = remove_small_objects(cleared_labels, min_size=min_border_size)

        # Create contour image
        contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        unique_labels = np.unique(final_labels)
        unique_labels = unique_labels[unique_labels > 0]
        colors = np.random.randint(50, 255, size=(len(unique_labels), 3))

        room_contours = []
        for i, label in enumerate(unique_labels):
            binary = (final_labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                room_contours.append({
                    'contour': contour,
                    'color': colors[i].tolist()
                })
                cv2.drawContours(contour_image, [contour], -1, colors[i].tolist(), 2)

        return contour_image, room_contours

    def create_geojson(self, room_contours, coordinates) -> Dict:
        """Convert room contours to GeoJSON with proper coordinate scaling"""
        features = []

        # Get the coordinate bounds
        min_lat = coordinates['min_lat']
        max_lat = coordinates['max_lat']
        min_long = coordinates['min_long']
        max_long = coordinates['max_long']

        lat_span = max_lat - min_lat
        long_span = max_long - min_long
        # Calculate the span
        all_points = np.concatenate([cont['contour'] for cont in room_contours])
        max_x = np.max(all_points[:, :, 0])
        max_y = np.max(all_points[:, :, 1])

        for room in room_contours:
            contour = room['contour']
            color = room['color']

            # Reshape contour if needed
            if len(contour.shape) == 3 and contour.shape[1] == 1:
                contour = contour.reshape(-1, 2)

            # Convert contour points to geographic coordinates
            geo_coords = []
            for point in contour:
                x, y = point[0], point[1]

                # Scale x and y to 0-1 range based on max dimensions from contours
                x_scaled = x / max_x
                y_scaled = 1 - (y / max_y)  # Flip Y axis

                # Convert to geographic coordinates using spans
                longitude = min_long + (long_span * x_scaled)
                latitude = min_lat + (lat_span * y_scaled)

                geo_coords.append([longitude, latitude])

            # Close the polygon if needed
            if len(geo_coords) > 0 and geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                },
                "properties": {
                    "color": color,
                    "area": float(cv2.contourArea(contour))
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }
    @staticmethod
    def encode_image(image) -> bytes:
        """Convert OpenCV image to bytes"""
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()

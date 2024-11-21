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
        # Configure pytesseract path if needed
        pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Main processing function that handles the complete pipeline"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process the image through your pipeline
            cropped_image = self.extract_floor_plan(image)
            room_contours = self.remove_gaps(cropped_image)

            # Convert contours to GeoJSON
            geojson = self.create_geojson(room_contours)

            return {
                "status": "success",
                "geojson": geojson,
                "processed_image": self.encode_image(cropped_image),
                "room_image": self.encode_image(room_contours)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def remove_text(self, image, conf_threshold=60, min_area=100, max_area=10000):
        """Your existing remove_text function"""
        ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(len(ocr_results['text'])):
            conf = int(ocr_results['conf'][i])
            if conf > conf_threshold:
                x, y, w, h = (ocr_results['left'][i], ocr_results['top'][i],
                              ocr_results['width'][i], ocr_results['height'][i])
                area = w * h
                if min_area < area < max_area:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        img_for_inpaint = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        result = cv2.inpaint(img_for_inpaint, mask, 2, cv2.INPAINT_TELEA)

        return result

    def extract_floor_plan(self, image, mp=0.1):
        """Your existing extract_floor_plan function modified for server use"""
        # Remove text first
        image = self.remove_text(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binary thresholding
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations
        height, width = gray.shape
        kernel = np.ones((int(height * mp), int(width * mp)), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)

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

        local_max_large = maximum_filter(dist_transform, size=100)
        local_max_small = maximum_filter(dist_transform, size=20)
        dist_max = dist_transform.max()
        peaks = ((dist_transform == local_max_large) &
                 (dist_transform == local_max_small) |
                 (dist_transform > peak_multiplier * dist_max))

        markers = cv2.connectedComponents(np.uint8(peaks))[1]
        inverted_dist_transform = -dist_transform
        labels = watershed(inverted_dist_transform, markers, mask=binary_image)
        cleared_labels = clear_border(labels)

        min_size = int((image.shape[0] * image.shape[1]) * (min_size_ratio ** 2))
        final_labels = remove_small_objects(cleared_labels, min_size=min_size)

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

    def create_geojson(self, room_contours) -> Dict:
        """Convert room contours to GeoJSON format"""
        features = []
        for room in room_contours:
            contour = room['contour']
            color = room['color']

            # Convert contour to coordinates
            coordinates = contour.squeeze().tolist()
            if len(coordinates) < 3:  # Skip invalid polygons
                continue

            # Ensure the polygon is closed
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                },
                "properties": {
                    "color": color,
                    "area": cv2.contourArea(contour)
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

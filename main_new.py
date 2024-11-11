import cv2
import numpy as np
import pytesseract
import json
import os

def extract_floor_plan(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped = result[y:y + h, x:x + w]
    return cropped

def remove_boxes_and_fill_arrows(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([25, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    
    mask = cv2.bitwise_or(red_mask, orange_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    image[mask > 0] = (255, 255, 255)

    cv2.imshow("mask", mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 800:
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 3:
                cv2.drawContours(image, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)
    
    return image

def make_floor_plan_black_and_white(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return result

def extract_room_boundaries(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    ocr_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
    for i, text in enumerate(ocr_data['text']):
        if len(text.strip()) == 1:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            cv2.rectangle(binary, (x, y), (x + w, y + h), (255), -1)

    edges = cv2.Canny(binary, 50, 150)
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for i, contour in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        coordinates = [[int(point[0][0]), int(point[0][1])] for point in approx]

        feature = {
            "type": "Feature",
            "properties": {
                "room_id": f"Room_{i+1}"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            }
        }
        geojson_data["features"].append(feature)

        # Find the center of each contour for labeling
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, f"{i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    geojson_output = json.dumps(geojson_data, indent=4)
    print("GeoJSON Data:")
    print(geojson_output)

    output_geojson_path = os.path.join("output", "room_boundaries.geojson")
    os.makedirs("output", exist_ok=True)
    with open(output_geojson_path, "w") as f:
        f.write(geojson_output)
    
    result = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
    cv2.imshow("Room Boundaries", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return geojson_data

# Usage
if __name__ == "__main__":
    image_path = os.path.join("floorplans", "raw", "MU_1.jpg")  # Update with the correct path to your image
    cropped = extract_floor_plan(image_path)
    output_path_cropped = os.path.join("new", "cropped_1.jpg")
    #cv2.imwrite(output_path_cropped, cropped)

    cleaned_img = remove_boxes_and_fill_arrows(cropped)
    final_img = make_floor_plan_black_and_white(cleaned_img)
    output_path_cleaned = os.path.join("new", "cleaned_floor_plan_1.jpg")
    #cv2.imwrite(output_path_cleaned, final_img)

    geojson_data = extract_room_boundaries(output_path_cleaned)
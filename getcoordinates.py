import requests

def get_coordinates_mapbox(address, access_token):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {
        "access_token": access_token
    }

    response = requests.get(url, params=params)
    data = response.json()

    # print(data)

    if data["features"]:
        location = data["features"][0]["geometry"]["coordinates"]
        longitude, latitude = location  # Note Mapbox returns [lng, lat]
        return latitude, longitude
    else:
        print("No results found")
        return None


def get_building_bounds_mapbox(address, access_token):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {
        "access_token": access_token,
    }

    response = requests.get(url, params=params)
    data = response.json()

    # print(data)

    if data["features"]:
        bbox = data["features"][0].get("bbox")
        # print(bbox)
        if bbox:
            top_left = (bbox[1], bbox[2])  # [max_lat, min_lng]
            bottom_right = (bbox[0], bbox[3])  # [min_lat, max_lng]
            return top_left, bottom_right
        else:
            print("Bounding box not available for this location.")
            return None
    else:
        print("No results found.")
        return None







def get_place_bounds_google(place_id, api_key):
    url = f"https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": "geometry"  # Request geometry to get the viewport
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] == "OK":
        viewport = data["result"]["geometry"]["viewport"]
        top_left = (viewport["northeast"]["lat"], viewport["southwest"]["lng"])
        bottom_right = (viewport["southwest"]["lat"], viewport["northeast"]["lng"])
        return top_left, bottom_right
    else:
        print("Place details request failed:", data["status"])
        return None



def get_building_bounds_google(address, api_key):
    # Step 1: Get the place_id and central coordinates using Geocoding API
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {
        "address": address,
        "key": api_key
    }
    geocode_response = requests.get(geocode_url, params=geocode_params)
    geocode_data = geocode_response.json()

    if geocode_data["status"] == "OK":
        place_id = geocode_data["results"][0]["place_id"]

        # Step 2: Use the place_id to get the bounding box from the Places API
        bounds = get_place_bounds_google(place_id, api_key)

        if bounds:
            print(f"Top-left: {bounds[0]}, Bottom-right: {bounds[1]}")
            return bounds
        else:
            print("Could not retrieve bounds.")
            return None
    else:
        print("Geocoding request failed:", geocode_data["status"])
        return None



# api_key = "YOUR_GOOGLE_API_KEY"
# address = "800 Langdon Street, Madison, WI"
# bounds = get_building_bounds_google(address, api_key)


# Works
#fetches bounding box for a given address using the
def get_building_bounds_osm(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json"
    }

    headers = {
        "User-Agent": "getcoordinates (simcard.adis@gmail.com)"  # Replace with your app name and email
    }

    response = requests.get(url, params=params, headers=headers)

    # Check if the response is successful
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        print("Response text:", response.text)
        return None

    try:
        data = response.json()
    except requests.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        print("Response text:", response.text)
        return None

    for item in data:
        if item.get("osm_type") == "way" and item.get("class") == "building":
            bbox = item.get("boundingbox")
            if bbox:
                top_left = (float(bbox[1]), float(bbox[2]))  # [max_lat, min_lng]
                bottom_right = (float(bbox[0]), float(bbox[3]))  # [min_lat, max_lng]
                return top_left, bottom_right

    print("No results found.")
    return None


address = "800 Langdon Street, Madison, WI"
access_token = "pk.eyJ1Ijoic2ltYXJqaXQxMjMiLCJhIjoiY20xb2V1cjM2MTR5YjJpcHZwNGVxbG5jeiJ9.1ppiJSjLROk1SM71_zHm9Q"
coordinates = get_coordinates_mapbox(address, access_token)
# if coordinates:
#     print(f"Coordinates for '{address}': {coordinates}")


bounds = get_building_bounds_osm(address)
if bounds:
    print(f"Top-left: {bounds[0]}, Bottom-right: {bounds[1]}")



# bounds = get_building_bounds_mapbox(address, access_token)
# if bounds:
#     print(f"Top-left: {bounds[0]}, Bottom-right: {bounds[1]}")


# latitude, longitude = coordinates[0], coordinates[1]  # Replace with actual coordinates
# data = get_building_bounds_osm(latitude, longitude , radius=1000)

# print(data)
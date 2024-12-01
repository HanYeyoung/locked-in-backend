import requests

# Test data
floor_id = "674bf2350aedf51f6ed9c5d7"
coordinates = {
    "min_lat": 43.07613222169164,
    "max_lat": 43.07646261221137,
    "min_long": -89.40027950525828,
    "max_long": -89.39964155730027,
    "center": {
        "lat": 43.076297416951505,
        "long": -89.39996053127928
    }
}

# Make the PUT request
response = requests.put(
    f"http://localhost:8000/floors/{floor_id}/coordinates",
    json=coordinates
)

# Print results
print(f"Status Code: {response.status_code}")
try:
    print(f"Response: {response.json()}")
except:
    print(f"Raw Response: {response.text}")
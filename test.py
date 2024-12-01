import requests

def get_all_buildings():
    response = requests.get('http://localhost:8000/buildings')
    print(response.json())

if __name__ == "__main__":
    get_all_buildings()
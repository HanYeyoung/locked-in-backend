from dotenv import load_dotenv
import os
from cloudinary import config as cloudinary_config
from cloudinary import uploader, api  # Add this import
from pymongo import MongoClient
from bson import ObjectId
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client.floorplans

CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

cloudinary_config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

def validate_config():
    required_vars = ['MONGO_URI']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    try:
        print("Cloudinary connection successful")
    except Exception as e:
        print(f"Cloudinary connection failed: {str(e)}")
        raise

validate_config()
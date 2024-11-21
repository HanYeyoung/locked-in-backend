from dotenv import load_dotenv
import os
from cloudinary import config as cloudinary_config
from cloudinary import uploader, api  # Add this import
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client.floorplans

CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRECT = os.getenv('CLOUDINARY_API_SECRET')

cloudinary_config(
    cloud_name="drxspo9mq",
    api_key="967515716937372",
    api_secret="LMwMmJu00yJtIbe1REcVpE1qESY"
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

# You might want to call this when starting your app
validate_config()
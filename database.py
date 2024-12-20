# database.py
from bson import ObjectId
import cloudinary.uploader
from datetime import datetime
from config import db
import cv2
import numpy as np
from typing import Dict, Any, Optional
from processors import FloorPlanProcessor

class DatabaseHandler:
    def __init__(self):
        self.buildings = db.buildings
        self.floors = db.floors
        self.processor = FloorPlanProcessor()

    async def create_building(self, name: str, address: str, coordinates: Dict[str, float]) -> Dict[str, Any]:
        """Create a new building record"""
        building_data = {
            "name": name,
            "address": address,
            "coordinates": coordinates,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = self.buildings.insert_one(building_data)
        building_data["_id"] = str(result.inserted_id)
        return building_data

    async def create_floor(self, building_id: str, floor_number: str, name: str) -> Dict[str, Any]:
        """Create a new floor record"""
        floor_data = {
            "building_id": building_id,
            "floor_number": floor_number,
            "name": name,
            "status": "pending",  # Initial status
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = self.floors.insert_one(floor_data)
        floor_data["_id"] = str(result.inserted_id)
        return floor_data

    async def process_floor_plan(self, floor_id: str, image_file, filename: str) -> Dict[str, Any]:
        """Process floor plan image and store results"""
        try:
            # Read image file
            image_data = await image_file.read()

            # Update status to processing
            self.floors.update_one(
                {"_id": ObjectId(floor_id)},
                {
                    "$set": {
                        "status": "processing",
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            # Process image
            result = self.processor.process_image(image_data)

            if result["status"] != "success":
                raise Exception(result.get("error", "Processing failed"))

            # Upload original to Cloudinary
            upload_result = cloudinary.uploader.upload(
                image_data,
                folder=f"floor_plans/{floor_id}",
                public_id="original"
            )

            # Upload processed images
            processed_result = cloudinary.uploader.upload(
                result["processed_image"],
                folder=f"floor_plans/{floor_id}",
                public_id="processed"
            )

            rooms_result = cloudinary.uploader.upload(
                result["room_image"],
                folder=f"floor_plans/{floor_id}",
                public_id="rooms"
            )

            # Update floor record with results
            update_data = {
                "images": {
                    "original": upload_result['secure_url'],
                    "processed": processed_result['secure_url'],
                    "rooms": rooms_result['secure_url']
                },
                "geojson": result["geojson"],
                "status": "completed",
                "processed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            self.floors.update_one(
                {"_id": ObjectId(floor_id)},
                {"$set": update_data}
            )

            # Return the complete updated record
            floor = self.get_floor(floor_id)
            return floor

        except Exception as e:
            # Update floor record with error status
            error_data = {
                "status": "error",
                "error_message": str(e),
                "updated_at": datetime.utcnow()
            }
            self.floors.update_one(
                {"_id": ObjectId(floor_id)},
                {"$set": error_data}
            )
            raise e

    async def get_floor(self, floor_id: str) -> Optional[Dict[str, Any]]:
        """Get floor details"""
        floor = self.floors.find_one({"_id": ObjectId(floor_id)})
        if floor:
            floor["_id"] = str(floor["_id"])
            return floor
        return None

    async def get_building(self, building_id: str) -> Optional[Dict[str, Any]]:
        """Get building details"""
        building = self.buildings.find_one({"_id": ObjectId(building_id)})
        if building:
            building["_id"] = str(building["_id"])
            return building
        return None

    async def get_building_floors(self, building_id: str) -> list:
        """Get all floors for a building"""
        floors = list(self.floors.find({"building_id": building_id}))
        for floor in floors:
            floor["_id"] = str(floor["_id"])
        return floors
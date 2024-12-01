# database.py
from bson import ObjectId
import cloudinary.uploader
from datetime import datetime
from config import db
import cv2
import numpy as np
from typing import Dict, Any, Optional
from processors import FloorPlanProcessor
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict, Any
import requests
import io
class DatabaseHandler:
    def __init__(self):
        self.buildings = db.buildings
        self.floors = db.floors
        self.processor = FloorPlanProcessor()

    async def get_buildings(self) -> list:
        """Get all buildings"""
        cursor = self.buildings.find({})
        buildings = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            buildings.append(doc)
        return buildings

    async def get_building(self, building_id: str) -> Optional[Dict[str, Any]]:
        """Get building details"""
        building = self.buildings.find_one({"_id": ObjectId(building_id)})
        if building:
            building["_id"] = str(building["_id"])
            return building
        return None

    async def create_building(self, name: str, address: str, coordinates: Dict[str, Any]) -> Dict[str, Any]:
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
        building = self.buildings.find_one({"_id": ObjectId(building_id)})
        if not building:
            raise HTTPException(status_code=404, message="Building not found")

        floor_data = {
            "building_id": building_id,
            "floor_number": floor_number,
            "name": name,
            "coordinates": building.get("coordinates"),
            "status": "pending",
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

            floor = self.floors.find_one({"_id": ObjectId(floor_id)})
            # Process image
            result = self.processor.process_image(image_data,floor.get("coordinates"))

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
            floor = await self.get_floor(floor_id)  # Add await here
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

    async def update_floor_coordinates(self, floor_id: str, coordinates: Dict[str, Any]) -> Dict[str, Any]:
        """Update floor coordinates and reprocess the floor plan if it exists"""
        try:
            # Find the floor and store original coordinates
            floor = self.floors.find_one({"_id": ObjectId(floor_id)})
            if not floor:
                raise ValueError("Floor not found")

            original_coordinates = floor.get('coordinates', {})

            # Update floor coordinates
            floor_update = self.floors.update_one(
                {"_id": ObjectId(floor_id)},
                {"$set": {
                    "coordinates": coordinates,
                    "updated_at": datetime.utcnow()
                }}
            )

            # Check if this floor has images to reprocess
            if "images" in floor and "original" in floor["images"]:
                try:
                    # Download the original image from Cloudinary
                    original_url = floor["images"]["original"]
                    response = requests.get(original_url)
                    if response.status_code == 200:
                        # Create a file-like object from the image data
                        class UploadFile:
                            def __init__(self, file, filename):
                                self.file = file
                                self.filename = filename

                            async def read(self):
                                return self.file.read()

                        file = io.BytesIO(response.content)
                        upload_file = UploadFile(file, f"floor_{floor_id}.jpg")

                        # Process the floor plan
                        result = await self.process_floor_plan(
                            floor_id,
                            upload_file,
                            upload_file.filename
                        )

                        return {
                            "floor_id": floor_id,
                            "coordinates": coordinates,
                            "status": "reprocessed",
                            "result": result
                        }
                    else:
                        raise Exception("Failed to download original image")

                except Exception as e:
                    # Revert coordinates update if processing fails
                    self.floors.update_one(
                        {"_id": ObjectId(floor_id)},
                        {"$set": {
                            "coordinates": original_coordinates,
                            "status": "error",
                            "error_message": str(e)
                        }}
                    )
                    raise Exception(f"Failed to reprocess floor plan: {str(e)}")

            # Return updated data if no images needed processing
            return {
                "floor_id": floor_id,
                "coordinates": coordinates,
                "status": "updated",
                "message": "Coordinates updated successfully. No images to reprocess."
            }

        except Exception as e:
            raise e

    async def get_building_floors(self, building_id: str) -> list:
        """Get all floors for a building"""
        floors = list(self.floors.find({"building_id": building_id}))
        for floor in floors:
            floor["_id"] = str(floor["_id"])
        return floors
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import DatabaseHandler
from typing import Dict, Any
from pydantic import BaseModel
import uvicorn
from bson import ObjectId

app = FastAPI()
db = DatabaseHandler()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Coordinates(BaseModel):
    min_lat: float
    max_lat: float
    min_long: float
    max_long: float
    center: Dict[str, float]
class BuildingCreate(BaseModel):
    name: str
    address: str
    coordinates: Dict[str, Any]
class FloorCreate(BaseModel):
    building_id: str
    floor_number: str
    name: str

@app.post("/buildings")
async def create_building(building: BuildingCreate):
    """Create a new building"""
    try:
        return await db.create_building(
            name=building.name,
            address=building.address,
            coordinates=building.coordinates
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings")
async def get_buildings():
    """Get all buildings"""
    try:
        buildings = await db.get_buildings()
        return buildings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buildings/{building_id}/floors")
async def create_floor(floor: FloorCreate):
    """Create a new floor for a building"""
    try:
        return await db.create_floor(
            building_id=floor.building_id,
            floor_number=floor.floor_number,
            name=floor.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buildings/{building_id}/floors/{floor_id}/image")
async def upload_floor_plan(
    building_id: str,
    floor_id: str,
    file: UploadFile = File(...)
):
    """Upload and process a floor plan image"""
    print(f"Received upload request - Building: {building_id}, Floor: {floor_id}")
    print(f"File: {file.filename}")
    try:
        return await db.process_floor_plan(floor_id, file, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buildings/{building_id}")
async def get_building(building_id: str):
    """Get building details"""
    building = await db.get_building(building_id)
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")
    return building

@app.get("/buildings/{building_id}/floors")
async def get_building_floors(building_id: str):
    """Get all floors for a building"""
    return await db.get_building_floors(building_id)

@app.get("/floors/{floor_id}")
async def get_floor(floor_id: str):
    """Get floor details"""
    floor = await db.get_floor(floor_id)
    if not floor:
        raise HTTPException(status_code=404, detail="Floor not found")
    return floor

@app.put("/floors/{floor_id}/coordinates")
async def update_floor_coordinates(
    floor_id: str,
    coordinates: Coordinates
):
    """Update floor coordinates and reprocess floor plan if images exist"""
    try:
        return await db.update_floor_coordinates(floor_id, coordinates.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
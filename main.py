from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import DatabaseHandler
from typing import Dict, Any
import uvicorn

app = FastAPI()
db = DatabaseHandler()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/buildings")
async def create_building(name: str, address: str, coordinates: Dict[str, float]):
    """Create a new building"""
    try:
        return await db.create_building(name, address, coordinates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buildings/{building_id}/floors")
async def create_floor(building_id: str, floor_number: str, name: str):
    """Create a new floor for a building"""
    try:
        return await db.create_floor(building_id, floor_number, name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/buildings/{building_id}/floors/{floor_id}/image")
async def upload_floor_plan(
    building_id: str,
    floor_id: str,
    file: UploadFile = File(...)
):
    """Upload and process a floor plan image"""
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
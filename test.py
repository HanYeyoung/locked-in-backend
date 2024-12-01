import asyncio
from database import DatabaseHandler
import os
from bson import ObjectId
from datetime import datetime


async def test_database_operations():
    # Initialize database handler
    db = DatabaseHandler()

    # Use existing floor ID
    floor_id = "674bdf22f719dd199cfa2b7d"

    try:
        # Path to existing test image
        test_image_path = os.path.join("..", "floorplans", "raw", "MU_4.jpg")

        # Mock file class
        class MockFile:
            async def read(self):
                with open(test_image_path, 'rb') as f:
                    return f.read()

            @property
            def filename(self):
                return "MU_4.jpg"

        # Process floor plan
        result = await db.process_floor_plan(floor_id, MockFile(), "MU_4.jpg")

        # Print the result for debugging
        print("Processing Result:", result)

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(test_database_operations())
import React, { useState, useEffect, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import GeoJSONButtons from "./GeoJSONButtons";

const FloorInterface = () => {
    const buildingId = useParams().buildingId;
    const floorId = useParams().floorId;
    const [floor, setFloor] = useState({});
    const [dragActive, setDragActive] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [loading, setLoading] = useState(true);
    const [showUpdateButton, setShowUpdateButton] = useState(false);
    const [updating, setUpdating] = useState(false);
    const fileInputRef = useRef(null);
    const mapContainer = useRef(null);
    const map = useRef(null);
    const blMarker = useRef(null);
    const trMarker = useRef(null);

    useEffect(() => {
        getFloor();
    }, [floorId]);

    useEffect(() => {
        if (floor?.coordinates) {
            getMap();
        }
    }, [floor]);

    const resetCoordinates = () => {
        if (!floor.coordinates) return;
        if (blMarker.current && trMarker.current) {
            blMarker.current.setLngLat([
                floor.coordinates.min_long,
                floor.coordinates.min_lat,
            ]);
            trMarker.current.setLngLat([
                floor.coordinates.max_long,
                floor.coordinates.max_lat,
            ]);
        }
        getMap();
        setShowUpdateButton(false);
    };

    const updateFloorCoordinates = async () => {
        if (!blMarker.current || !trMarker.current) return;

        setUpdating(true);
        const blCoords = blMarker.current.getLngLat();
        const trCoords = trMarker.current.getLngLat();

        const coordinates = {
            min_lat: Math.min(blCoords.lat, trCoords.lat),
            max_lat: Math.max(blCoords.lat, trCoords.lat),
            min_long: Math.min(blCoords.lng, trCoords.lng),
            max_long: Math.max(blCoords.lng, trCoords.lng),
            center: {
                lat: (blCoords.lat + trCoords.lat) / 2,
                long: (blCoords.lng + trCoords.lng) / 2,
            },
        };
        console.log("Sending coordinates:", coordinates);
        console.log("Stringified:", JSON.stringify(coordinates));

        fetch(`http://localhost:8000/floors/${floorId}/coordinates`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(coordinates),
        })
            .then((res) => {
                console.log(res);
                if (!res.ok) {
                    throw new Error("Failed to update");
                }
                return res.json();
            })
            .then((data) => {
                console.log(data);
                getFloor();
            })
            .catch((error) => {
                console.error(error.message);
            })
            .finally(() => {
                setUpdating(false);
                setShowUpdateButton(false);
            });
    };

    const getMap = () => {
        setLoading(true);
        mapboxgl.accessToken =
            "pk.eyJ1IjoiYXJpYW5hYmJhc3phZGVoIiwiYSI6ImNtNDRzeDF3NDBwbWIya3ExbndpZWZiNXoifQ.yIar70M4eN2Q1KKX_iYgqA";

        var map = new mapboxgl.Map({
            container: mapContainer.current,
            style: "mapbox://styles/mapbox/dark-v11",
            center: [
                floor.coordinates.center.long,
                floor.coordinates.center.lat,
            ],
            zoom: 18,
        });

        map.on("load", () => {
            map.addControl(new mapboxgl.NavigationControl(), "top-right");

            const bl = new mapboxgl.Marker({
                draggable: true,
                color: "#1e293b",
            })
                .setLngLat([
                    floor.coordinates.min_long,
                    floor.coordinates.min_lat,
                ])
                .addTo(map);

            const tr = new mapboxgl.Marker({
                draggable: true,
                color: "#1e293b",
            })
                .setLngLat([
                    floor.coordinates.max_long,
                    floor.coordinates.max_lat,
                ])
                .addTo(map);

            blMarker.current = bl;
            trMarker.current = tr;

            if (floor.geojson) {
                if (map.getSource("floor")) {
                    map.removeLayer("floor-fill");
                    map.removeLayer("floor-border");
                    map.removeSource("floor");
                }

                map.addSource("floor", {
                    type: "geojson",
                    data: floor.geojson,
                });

                // Add floor fill layer
                map.addLayer({
                    id: "floor-fill",
                    type: "fill",
                    source: "floor",
                    paint: {
                        "fill-color": "#FFFFFF",
                        "fill-opacity": 0.5,
                    },
                });

                // Add floor border layer
                map.addLayer({
                    id: "floor-border",
                    type: "line",
                    source: "floor",
                    paint: {
                        "line-color": "#FFFFFF",
                        "line-width": 2,
                        "line-opacity": 1,
                    },
                });
            }
            const drawBox = () => {
                // Remove existing box if it exists
                if (map.getSource("box")) {
                    map.removeLayer("box-fill");
                    map.removeLayer("box-border");
                    map.removeSource("box");
                }

                const minCoords = bl.getLngLat();
                const maxCoords = tr.getLngLat();

                map.addSource("box", {
                    type: "geojson",
                    data: {
                        type: "Feature",
                        geometry: {
                            type: "Polygon",
                            coordinates: [
                                [
                                    [minCoords.lng, minCoords.lat],
                                    [maxCoords.lng, minCoords.lat],
                                    [maxCoords.lng, maxCoords.lat],
                                    [minCoords.lng, maxCoords.lat],
                                    [minCoords.lng, minCoords.lat],
                                ],
                            ],
                        },
                    },
                });

                map.addLayer({
                    id: "box-fill",
                    type: "fill",
                    source: "box",
                    paint: {
                        "fill-color": "#3b82f6",
                        "fill-opacity": 0.2,
                    },
                });

                map.addLayer({
                    id: "box-border",
                    type: "line",
                    source: "box",
                    paint: {
                        "line-color": "#3b82f6",
                        "line-width": 2,
                    },
                });
                setShowUpdateButton(true);
            };

            const updateBounds = () => {
                drawBox();
            };

            // Draw initial box
            drawBox();

            // Update on marker drag
            bl.on("dragend", updateBounds);
            tr.on("dragend", updateBounds);
            bl.on("drag", drawBox);
            tr.on("drag", drawBox);
            setShowUpdateButton(false);
        });
        setLoading(false);
        return () => map.remove();
    };

    const getFloor = () => {
        fetch(`http://localhost:8000/floors/${floorId}`)
            .then((res) => {
                if (!res.ok)
                    throw new Error(`Building fetch failed: ${res.status}`);
                return res.json();
            })
            .then((data) => {
                console.log(data);
                setFloor(data);
            })
            .then(() => {
                console.log("floor", JSON.stringify(floor));
                getMap();
            })
            .catch((err) => console.error("Error:", err));
    };

    const uploadImage = async (file) => {
        setUploading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(
                `http://localhost:8000/buildings/${buildingId}/floors/${floorId}/image`,
                {
                    method: "POST",
                    body: formData,
                }
            );

            if (!response.ok) throw new Error("Upload failed");

            const data = await response.json();
            getFloor();
        } catch (error) {
            console.error("Upload error:", error);
        } finally {
            setUploading(false);
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            await uploadImage(file);
        }
    };

    const handleFileSelect = async (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith("image/")) {
            await uploadImage(file);
        }
    };
    return (
        <div className="bg-black px-4 py-4 text-white">
            {floor && (
                <div className="flex flex-row items-center p-4 h-32">
                    <Link
                        to={`/buildings/${buildingId}/floors`}
                        className="text-blue-600 text-4xl hover:text-blue-400  rounded-full hover:bg-slate-800 p-2 px-3 duration-200 flex items-center"
                    >
                        ←
                    </Link>
                    <div className="text-5xl text-center font-bold p-4">
                        {floor.name}
                    </div>
                </div>
            )}
            <div className="grid grid-cols-5 gap-6 p-4">
                <div className="col-span-3">
                    <div className="rounded-3xl border border-white p-6">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-2xl font-bold ">Location</h3>
                            {showUpdateButton && (
                                <div className="justify-end ">
                                    <button
                                        onClick={() => updateFloorCoordinates()}
                                        disabled={updating}
                                        className="px-4 py-2 text-white rounded-full border border-white hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition duration-200"
                                    >
                                        {updating
                                            ? "Updating..."
                                            : "Update Coordinates"}
                                    </button>
                                    <button
                                        onClick={() => resetCoordinates()}
                                        className="px-4 py-2 mx-4 text-white rounded-full border border-white hover:bg-red-700 transition duration-200"
                                    >
                                        Reset
                                    </button>
                                </div>
                            )}
                        </div>
                        {floor.coordinates ? (
                            <div
                                id="map"
                                ref={mapContainer}
                                className="w-full h-[60vh] bg-slate-800 rounded-2xl border-slate-400 border-2"
                            />
                        ) : (
                            <div className="w-full h-[50vh] bg-slate-800 rounded-2xl flex items-center justify-center text-gray-400">
                                No location data available
                            </div>
                        )}
                    </div>
                </div>

                <div className="col-span-2 flex flex-col gap-6 h-full">
                    <div className="rounded-3xl border border-white p-6">
                        <div className="flex-row justify-start">
                            <h3 className="text-2xl font-bold mb-4">Image</h3>
                        </div>
                        {floor.images && !loading ? (
                            <img
                                src={floor.images.original}
                                alt="Floor Plan"
                                className="w-full h-56 rounded-2xl object-cover duration-100 hover:brightness-75"
                            />
                        ) : (
                            <div
                                className={`w-full h-56 bg-slate-800 rounded-2xl border-2 border-dashed 
								${dragActive ? "border-blue-500 bg-slate-700" : "border-gray-400"} 
								flex flex-col items-center justify-center relative`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                            >
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    onChange={handleFileSelect}
                                    accept="image/*"
                                    className="hidden"
                                />
                                <div className="text-gray-400 text-center">
                                    {uploading ? (
                                        <p>Uploading...</p>
                                    ) : (
                                        <>
                                            <p className="mb-2">
                                                Drag and drop your floor plan
                                                here
                                            </p>
                                            <p className="text-sm">or</p>
                                            <button
                                                className="mt-2 px-4 py-2 border border-white rounded-full hover:bg-blue-600 text-white hover:border-2 duration-200"
                                                onClick={() =>
                                                    fileInputRef.current?.click()
                                                }
                                            >
                                                Browse Files
                                            </button>
                                        </>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                    <div className="rounded-3xl border border-white p-6 flex-1 h-full">
                        <h3 className="text-2xl font-bold mb-4">GeoJSON</h3>
                        <GeoJSONButtons floor={floor} />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FloorInterface;

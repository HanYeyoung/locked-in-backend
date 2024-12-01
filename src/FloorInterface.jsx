import React, { useState, useEffect, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const FloorInterface = () => {
    const buildingId = useParams().buildingId;
    const floorId = useParams().floorId;
    const [floor, setFloor] = useState({});
    const [dragActive, setDragActive] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [loading, setLoading] = useState(true);
    const fileInputRef = useRef(null);
    const mapContainer = useRef(null);
    const map = useRef(null);

    useEffect(() => {
        getFloor();
    }, [floorId]);

    useEffect(() => {
        if (floor?.coordinates) {
            getMap();
        }
    }, [floor]);

    const getMap = () => {
        setLoading(true);
        mapboxgl.accessToken =
            "pk.eyJ1IjoiYXJpYW5hYmJhc3phZGVoIiwiYSI6ImNtNDRzeDF3NDBwbWIya3ExbndpZWZiNXoifQ.yIar70M4eN2Q1KKX_iYgqA";

        var map = new mapboxgl.Map({
            container: mapContainer.current,
            style: "mapbox://styles/mapbox/streets-v12",
            center: [
                floor.coordinates.center.long,
                floor.coordinates.center.lat,
            ],
            zoom: 17,
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
            };

            const updateBounds = () => {
                const minCoords = bl.getLngLat();
                const maxCoords = tr.getLngLat();

                drawBox();

                console.log("New bounds:", {
                    min_long: minCoords.lng,
                    min_lat: minCoords.lat,
                    max_long: maxCoords.lng,
                    max_lat: maxCoords.lat,
                });
            };

            // Draw initial box
            drawBox();

            // Update on marker drag
            bl.on("dragend", updateBounds);
            tr.on("dragend", updateBounds);
            bl.on("drag", drawBox);
            tr.on("drag", drawBox);
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
                        <h3 className="text-2xl font-bold mb-4">Location</h3>
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
                    <div className="rounded-3xl border border-white p-6 flex-1">
                        <h3 className="text-2xl font-bold mb-4">GeoJSON</h3>
                        <button className="w-full py-3 border border-white rounded-full hover:bg-blue-600 hover:border-2 duration-200">
                            Download GeoJSON
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FloorInterface;

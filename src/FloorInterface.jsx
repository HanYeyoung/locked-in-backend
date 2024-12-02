import React, { useState, useEffect, useRef, Suspense } from "react";
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
    const [selectedImage, setSelectedImage] = useState("rooms");
    const [showUploadScreen, setShowUploadScreen] = useState(false);
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
            zoom: 17,
        });

        map.on("load", () => {
            map.addControl(new mapboxgl.NavigationControl(), "top-right");

            const bl = new mapboxgl.Marker({
                draggable: true,
                color: "#64748b	",
            })
                .setLngLat([
                    floor.coordinates.min_long,
                    floor.coordinates.min_lat,
                ])
                .addTo(map);

            const tr = new mapboxgl.Marker({
                draggable: true,
                color: "#64748b",
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
                        "fill-color": "#ffffff",
                        "fill-opacity": 0.2,
                    },
                });

                // Add floor border layer
                map.addLayer({
                    id: "floor-border",
                    type: "line",
                    source: "floor",
                    paint: {
                        "line-color": "#ffffff",
                        "line-width": 2,
                        "line-opacity": 0.5,
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

        fetch(
            `http://localhost:8000/buildings/${buildingId}/floors/${floorId}/image`,
            {
                method: "POST",
                body: formData,
            }
        )
            .then((res) => {
                if (!res.ok) throw new Error("Upload failed");
                return res.json();
            })
            .then(() => {
                getFloor();
                handleUploadSuccess();
            })
            .catch((error) => {
                console.error("Upload error:", error);
            })
            .finally(() => {
                setUploading(false);
            });
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

    const getImageStyle = (type) => {
        if (type === "original") return "object-cover";
        return "object-contain";
    };

    const imageOptions = [
        { value: "rooms", label: "Rooms" },
        { value: "processed", label: "Cropped" },
        { value: "original", label: "Original" },
    ];

    const handleImageClick = () => {
        setShowUploadScreen(true);
    };

    const handleUploadSuccess = () => {
        setShowUploadScreen(false);
    };

    return (
        <div className="bg-black px-4 py-4 text-white">
            <Suspense>
                {floor && (
                    <div className="flex flex-row items-center p-4 h-32">
                        <Link
                            to={`/buildings/${buildingId}/floors`}
                            className="text-blue-600 text-4xl hover:text-blue-400 rounded-full hover:bg-slate-800 p-2 px-3 duration-200 flex items-center"
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
                                <h3 className="text-2xl font-bold">Location</h3>
                                {showUpdateButton && (
                                    <div className="justify-end">
                                        <button
                                            onClick={() =>
                                                updateFloorCoordinates()
                                            }
                                            disabled={updating}
                                            className={`px-4 py-2 text-white rounded-full border border-white disabled:cursor-not-allowed transition duration-200 
                                                ${
                                                    updating
                                                        ? "animate-[gradient_3s_ease-in-out_infinite] bg-[length:200%_200%] bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500 text-white font-medium border-none"
                                                        : "hover:bg-blue-700"
                                                }`}
                                            style={{
                                                backgroundPosition: updating
                                                    ? "right center"
                                                    : "left center",
                                            }}
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
                                    className="w-full h-[500px] bg-slate-800 rounded-2xl border-slate-400 border-2"
                                />
                            ) : (
                                <div className="w-full h-[500px] bg-slate-800 rounded-2xl flex items-center justify-center text-gray-400">
                                    No location data available
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="col-span-2 flex flex-col gap-6 h-full">
                        <div className="rounded-3xl border border-white p-6 flex-1 flex flex-col">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-2xl font-bold">Image</h3>
                                {floor.images && !showUploadScreen && (
                                    <select
                                        value={selectedImage}
                                        onChange={(e) =>
                                            setSelectedImage(e.target.value)
                                        }
                                        className="bg-black text-white border cursor-pointer border-white hover:bg-blue-600 duration-100 rounded-full px-2 py-2 focus:outline-none "
                                    >
                                        {imageOptions.map((option) => (
                                            <option
                                                key={option.value}
                                                value={option.value}
                                            >
                                                {option.label}
                                            </option>
                                        ))}
                                    </select>
                                )}
                            </div>
                            {floor.images && !loading && !showUploadScreen ? (
                                <div
                                    className="flex-1 flex items-center justify-center relative group cursor-pointer"
                                    onClick={handleImageClick}
                                >
                                    <img
                                        src={floor.images[selectedImage]}
                                        alt="Floor Plan"
                                        className={`w-full h-[300px] border-2 border-slate-400 rounded-3xl ${getImageStyle(
                                            selectedImage
                                        )} group-hover:brightness-50 duration-200`}
                                    />
                                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 duration-200">
                                        <span className=" px-6 py-3 rounded-full text-white font-medium">
                                            Upload New Image
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div
                                    className={`w-full h-full bg-slate-800 rounded-3xl border-2 border-dashed 
                                    ${
                                        dragActive
                                            ? "border-blue-500 bg-slate-700"
                                            : "border-gray-400"
                                    } 
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
                                                    Drag and drop your floor
                                                    plan here
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
                        <div className="rounded-3xl border border-white p-6 h-[150px]">
                            <h3 className="text-2xl font-bold">GeoJSON</h3>
                            <GeoJSONButtons floor={floor} />
                        </div>
                    </div>
                </div>
            </Suspense>
        </div>
    );
};

export default FloorInterface;

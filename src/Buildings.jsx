import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

const Buildings = () => {
    const [buildings, setBuildings] = useState([]);
    const [modal, setModal] = useState(false);
    const [bname, setBname] = useState("");
    const [address, setAddress] = useState("");
    const [coordinates, setCoordinates] = useState({});
    const [addError, setAddError] = useState("");

    useEffect(() => {
        getBuildings();
    }, []);

    useEffect(() => {
        setAddError("");
        if (!modal) {
            setBname("");
            setAddress("");
            setCoordinates({});
        }
    }, [modal]);

    const getBuildings = () => {
        fetch("http://localhost:8000/buildings")
            .then((res) => {
                if (res.ok) {
                    return res.json();
                }
                throw new Error("Network request Failed");
            })
            .then((data) => setBuildings(data))
            .catch((err) => console.error("Error:", err));
    };

    const createBuilding = async () => {
        if (!coordinates?.center) {
            setAddError("No coordinates available");
            return;
        }
        console.log(JSON.stringify(coordinates));
        console.log("truth", !coordinates?.center);

        fetch("http://localhost:8000/buildings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: bname,
                address: address,
                coordinates: coordinates,
            }),
        })
            .then((res) => {
                console.log(res);
                if (!res.ok) {
                    throw new Error(res.statusText);
                }
                return res.json();
            })
            .then((data) => {
                alert("Successfully created " + data.name);
                setModal(false);
                getBuildings();
            })
            .catch((err) => console.error("Error:", err));
    };

    const getCoordinates = async () => {
        if (!address?.trim()) {
            console.log("Invalid address provided");
            return null;
        }

        const url = new URL("https://nominatim.openstreetmap.org/search");

        url.search = new URLSearchParams({
            q: address,
            format: "json",
        }).toString();
        return fetch(url, {
            method: "GET",
            headers: {
                "User-Agent": "getcoordinates (simcard.adis@gmail.com)",
            },
        })
            .then((res) => {
                if (!res.ok) {
                    throw new Error(
                        `Request failed with status code ${res.status}\n `
                    );
                }
                return res.json();
            })
            .then((data) => {
                for (const item of data) {
                    if (item.osm_type === "way" && item.class === "building") {
                        const bbox = item.boundingbox;
                        if (bbox) {
                            return setCoordinates({
                                min_lat: parseFloat(bbox[0]),
                                max_lat: parseFloat(bbox[1]),
                                min_long: parseFloat(bbox[2]),
                                max_long: parseFloat(bbox[3]),
                                center: {
                                    lat:
                                        (parseFloat(bbox[0]) +
                                            parseFloat(bbox[1])) /
                                        2,
                                    long:
                                        (parseFloat(bbox[2]) +
                                            parseFloat(bbox[3])) /
                                        2,
                                },
                            });
                        }
                    }
                }
                throw new Error("No building bounds found for this address");
            })
            .catch((error) => {
                console.error("Error fetching building bounds:", error.message);
                return error.message;
            });
    };

    return (
        <div className="bg-black mx-auto px-4 py-4 h-screen text-white">
            <div className="justify-between flex flex-row items-center p-4 h-32">
                <div className="text-5xl text-center font-bold p-4">
                    Buildings
                </div>
                <div
                    className="text-xl text-center m-4 p-4 border border-white bg-black hover:font-bold hover:border-2 hover:bg-blue-600 hover:text-white active:scale-95 duration-200 rounded-full cursor-pointer"
                    onClick={() => setModal((prev) => !prev)}
                >
                    Add A Building
                </div>
                {modal && (
                    <div className="absolute top-1/4 rounded-3xl right-0 m-12 w-5/12 border bg-black border-white z-50 p-4">
                        <div className="flex flex-col space-y-3">
                            <input
                                type="text"
                                placeholder="Building Name"
                                className="p-2 rounded-2xl bg-slate-800 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                                value={bname}
                                onChange={(e) => setBname(e.target.value)}
                            />
                            <input
                                type="text"
                                placeholder="Address"
                                className="p-2 rounded-2xl bg-slate-800 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                                value={address}
                                onChange={(e) => setAddress(e.target.value)}
                            />
                            {addError && (
                                <div className="p-2, text-red-500">
                                    {addError}
                                </div>
                            )}
                            <div className="flex justify-end space-x-2">
                                <button
                                    onClick={() => setModal(false)}
                                    className="px-4 py-2 rounded-2xl border border-white hover:border-red-500 text-gray-300"
                                >
                                    Cancel
                                </button>
                                <button
                                    className="px-4 py-2 rounded-2xl border border-white hover:font-bold duration-200 bg-blue-600  text-white"
                                    onClick={async () => {
                                        if (!bname || !address) {
                                            setAddError(
                                                "Please Enter Building Name And Address."
                                            );
                                            return;
                                        }
                                        const nameExists = buildings.some(
                                            (building) =>
                                                building.name.toLowerCase() ===
                                                bname.toLowerCase()
                                        );

                                        if (nameExists) {
                                            setAddError(
                                                "A building with this name already exists"
                                            );
                                            return;
                                        }

                                        getCoordinates().then((out) => {
                                            if (!coordinates?.center) {
                                                setAddError(out);
                                                return;
                                            }
                                            createBuilding();
                                        });
                                    }}
                                >
                                    Add
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {buildings.length <= 0 ? (
                <div className="flex-1 h-[60vh] m-4 flex items-center justify-center">
                    <div className="flex flex-col space-y-4 items-center">
                        <div className="text-5xl font-extralight text-white">
                            No Buildings
                        </div>
                        <div className="text-3xl text-white">
                            Click "Add A Building" To Create One
                        </div>
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
                    {buildings.map((building) => (
                        <Link
                            key={building.id}
                            to={`/buildings/${building._id}/floors`}
                            className="group h-fit p-4 bg-black rounded-3xl border-white border hover:scale-95  duration-200"
                        >
                            <h3 className="text-2xl group-hover:font-black duration-200 font-bold text-slate-100">
                                {building.name}
                            </h3>
                            <p className="text-slate-300 group-hover:font-bold duration-200 font-medium mt-2">
                                {building.address}
                            </p>
                        </Link>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Buildings;

import React, { useState, useEffect } from "react";
import { Link, useParams } from "react-router-dom";

const Floors = () => {
    const { buildingId } = useParams();
    const [floors, setFloors] = useState([]);
    const [building, setBuilding] = useState([]);
    const [modal, setModal] = useState(false);
    const [fnum, setFnum] = useState("");
    const [fname, setFname] = useState("");
    const [addError, setAddError] = useState("");

    useEffect(() => {
        getFloors();
        console.log("coordinates", building.coordinates);
    }, [buildingId]);

    useEffect(() => {
        setAddError("");
    }, [modal]);

    const getFloors = () => {
        fetch(`http://localhost:8000/buildings/${buildingId}`)
            .then((res) => {
                if (!res.ok)
                    throw new Error(`Building fetch failed: ${res.status}`);
                return res.json();
            })
            .then((data) => {
                setBuilding(data);
                return fetch(
                    `http://localhost:8000/buildings/${buildingId}/floors`
                );
            })
            .then((res) => {
                if (!res.ok)
                    throw new Error(`Floors fetch failed: ${res.status}`);
                return res.json();
            })
            .then((data) => setFloors(data))
            .then(() => console.log(floors))
            .catch((err) => console.error("Error:", err));
    };
    const createFloor = () => {
        fetch(`http://localhost:8000/buildings/${building._id}/floors`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                building_id: building._id,
                floor_number: fnum,
                name: fname,
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
                alert(
                    "Successfully created " + data.name + " in " + building.name
                );
                setModal(false);
                getFloors();
            })
            .catch((err) => console.error("Error:", err));
    };
    return (
        <div className=" bg-black h-screen text-white  mx-auto px-4 py-8 flex flex-col">
            <div className="flex flex-row justify-between">
                <div className="flex flex-row items-center p-4">
                    <Link
                        to="/buildings"
                        className="text-blue-600 text-4xl hover:text-blue-400  rounded-full hover:bg-slate-800 p-2 px-3 duration-200 flex items-center"
                    >
                        ←
                    </Link>
                    <div className="text-5xl font-bold ml-4">
                        {building.name}
                    </div>
                </div>
                <div>
                    <div
                        className="text-xl text-center m-4 p-4 border hover:border-2 hover:font-bold hover:font border-white bg-black hover:bg-blue-600 active:scale-95 duration-200 rounded-full cursor-pointer"
                        onClick={() => setModal((prev) => !prev)}
                    >
                        Add A Floor
                    </div>
                </div>
            </div>
            {modal && (
                <div className="absolute top-1/4 rounded-3xl right-0 m-12 w-5/12 border bg-black border-white z-50 p-4">
                    <div className="flex flex-col space-y-3">
                        <input
                            type="text"
                            placeholder="Floor Name"
                            className="p-2 rounded-3xl bg-slate-800 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                            value={fname}
                            onChange={(e) => setFname(e.target.value)}
                        />
                        <input
                            type="text"
                            placeholder="Floor Number"
                            className="p-2 rounded-3xl bg-slate-800 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                            value={fnum}
                            onChange={(e) => setFnum(e.target.value)}
                        />
                        {addError && (
                            <div className="p-2, text-red-500">{addError}</div>
                        )}
                        <div className="flex justify-end space-x-4">
                            <button
                                onClick={() => setModal(false)}
                                className="px-4 py-2 rounded-3xl border border-white hover:border-red-500 text-gray-300"
                            >
                                Cancel
                            </button>
                            <button
                                className="px-4 py-2 rounded-3xl border border-white duration-200 bg-black hover:bg-blue-600 text-white"
                                onClick={() => {
                                    if (!fname || !fnum) {
                                        setAddError(
                                            "Please Enter Floor Name And Number."
                                        );
                                        return;
                                    }
                                    const floorExists = floors.some(
                                        (floor) =>
                                            floor.name.toLowerCase() ===
                                            fnum.toLowerCase()
                                    );

                                    if (floorExists) {
                                        setAddError(
                                            "A floor with this name already exists"
                                        );
                                        return;
                                    }
                                    createFloor();
                                }}
                            >
                                Add
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {floors.length === 0 ? (
                <div className="text-5xl font-extralight h-full items-center justify-center m-4 flex-flex-row">
                    No Floors For
                    <span className="font-semibold"> {building.name}</span>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6 m-4">
                    {floors.map((floor) => (
                        <Link
                            key={floor._id}
                            to={`/buildings/${buildingId}/floors/${floor._id}`}
                            className="group h-fit p-6  bg-black rounded-3xl border-white border hover:scale-95  duration-200"
                        >
                            <h3 className="text-2xl group-hover:font-black duration-200 font-bold text-slate-100">
                                {floor.name}
                            </h3>
                            <p className="text-slate-300 group-hover:font-bold duration-200 font-medium mt-2">
                                {" "}
                                {`Floor Number: ${floor.floor_number}`}
                            </p>
                            {floor.images && floor.images.original ? (
                                <img
                                    src={floor.images.original}
                                    className="rounded-2xl h-40 w-full bg-slate-500 mt-4 object-cover"
                                />
                            ) : (
                                <div className="text-center text-2xl font-bold mt-4">
                                    {" "}
                                    No Image Uploaded
                                </div>
                            )}
                        </Link>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Floors;

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const Buildings = () => {
	const [buildings, setBuildings] = useState([]);
	const [modal, setModal] = useState([])
	useEffect(() => {
		fetch('http://localhost:8000/buildings',
		)
			.then(res => res.json())
			.then(data => setBuildings(data))
			.catch(err => console.error('Error:', err));
	}, []);

	return (
		<div className="bg-black mx-auto px-4 py-4 h-screen text-white">
			<div className='justify-between flex flex-row items-center p-4'>
				<div className="text-5xl text-center font-bold p-4">
					Buildings
				</div>
				<div className="text-xl text-center m-4 p-4 border-2 border-white bg-black hover:font-bold hover:bg-blue-600 hover:text-white active:scale-95 active:bg-green-400 duration-200 rounded-full cursor-pointer">
					Add A Building
				</div>
			</div>
			<div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6 ">
				{buildings &&
					buildings === [] ? (<div className="text-xl text-center m-4 p-4 border-2 border-white bg-black hover:font-bold hover:bg-blue-600 hover:text-white active:scale-95 active:bg-green-400 duration-200 rounded-full cursor-pointer">
						No Buildings
					</div>) :
					buildings.map(building => (
						<Link
							key={building.id}
							to={`/buildings/${building._id}/floors`}
							className="h-fit p-4 bg-black rounded-3xl border-white border-2 hover:scale-95 hover:skew-x-6 duration-200"
						>
							<h3 className="text-3xl font-bold text-slate-100">{building.name}</h3>
							<p className="text-slate-300 font-bold mt-2">{building.address}</p>
						</Link>
					))
				}
			</div>
		</div>
	);
};

export default Buildings;
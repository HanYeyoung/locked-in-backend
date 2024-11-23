import React, { useState, useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';

const Floors = () => {
	const { buildingId } = useParams();
	const [floors, setFloors] = useState([]);
	const [building, setBuilding] = useState([]);

	useEffect(() => {
		fetch(`http://localhost:8000/buildings/${buildingId}`)
			.then(res => res.json())
			.then(data => setBuilding(data))
			.catch(err => console.error('Error:', err))
			.then(
				fetch(`http://localhost:8000/buildings/${buildingId}/floors`)
					.then(res => res.json())
					.then(data => setFloors(data))
					.catch(err => console.error('Error:', err))
			);
	}, [buildingId]);
	return (
		<div className=" bg-black h-screen text-white  mx-auto px-4 py-8 flex flex-col">
			<div className='flex flex-row justify-between'>
				<div className='flex flex-row items-center p-4'>
					<Link
						to="/buildings"
						className="text-blue-600 text-4xl hover:text-blue-400 rounded-full hover:bg-slate-800 p-3 px-4 duration-200 flex items-center"
					>
						←
					</Link>
					<div className="text-5xl font-bold ml-4">
						{building.name}
					</div>
				</div>
				<div>
					<div className="text-xl text-center m-4 p-4 border-2 border-white bg-black hover:bg-blue-600 hover:text-white active:scale-95 active:bg-green-400 duration-200 rounded-full cursor-pointer">
						Add A Floor
					</div>
				</div>
			</div>
			{floors ? (<div className="text-5xl font-extralight h-full items-center justify-center m-4 flex-flex-row">
				No Floors For
				<span className='font-semibold'> {building.name}</span>
			</div>) :
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 m-4">
					{floors.map(floor => (
						<div key={floor.id} className="p-6 bg-white rounded-lg shadow-sm">
							<h3 className="text-xl font-semibold text-gray-900">{floor.name}</h3>
							<p className="text-gray-600 mt-2">Floor {floor.floor_number}</p>
						</div>
					))}
				</div>
			}
		</div>
	);
};

export default Floors;
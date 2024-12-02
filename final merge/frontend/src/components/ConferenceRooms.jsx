import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import RoomModal from './RoomModal';
import geoJSONCollection from '../assets/floorMap';

const AMERICAN_CENTER = [-100, 40];
const PADDING = 50;


//floor ids are UNION_SOUTH_IV and UNION_SOUTH_I
function ConferenceRooms({username}) {
    const mapRef = useRef();
    const mapContainerRef = useRef();
    const [selectedRoom, setSelectedRoom] = useState(null);
    const [selectedBuilding, setSelectedBuilding] = useState(null);
    const [buildings, setBuildings] = useState([]);
    //New debugging code
    const [debugInfo, setDebugInfo] = useState({
        startingZoom: null,
        startingFitBounds: null,
        startingMinZoom: null,
        currentMaxBounds: null,
        currentMinZoom: null,
        currentZoom: null,
    });

    useEffect(() => {
        // Parse geoJSONCollection to create the buildings array
        let buildingsArray = Object.entries(geoJSONCollection).map(([id, data]) => ({
          id,
          name: data.name,
          coordinates: JSON.parse(data.map_coordinates),
          geoJSON: data
        }));

        // Filter buildings based on the logged-in user's username
        if (username === 'admin') {
            buildingsArray = buildingsArray.filter(building => ['UNION_SOUTH_IV', 'UNION_SOUTH_I'].includes(building.id));
        } else if (username === 'joeuntrecht') {
            buildingsArray = buildingsArray.filter(building => building.id === 'UNION_SOUTH_IV');
        } else if (username === 'eligauger') {
            buildingsArray = buildingsArray.filter(building => building.id === 'UNION_SOUTH_I');
        }

        setBuildings(buildingsArray);
    }, []);

    useEffect(() => {     
        mapboxgl.accessToken = 'pk.eyJ1IjoianVzdGluYmlsZG5lciIsImEiOiJjbTIyM3c0azUwMnllMmxuNzFwY3V0Y204In0.n4K--AHy82A8xh9JxeLUnw';

        if (!selectedBuilding) {
            // Initialize map with default view if no building is selected
            mapRef.current = new mapboxgl.Map({
                container: mapContainerRef.current,
                style: 'mapbox://styles/mapbox/dark-v11',
                center: AMERICAN_CENTER,
                zoom: 2.9
            });
            return;
        }


        // Calculate bounds of the selected building
        const bounds = new mapboxgl.LngLatBounds();
        selectedBuilding.geoJSON.features.forEach(feature => {
            if (feature.geometry.type === 'Polygon') {
                feature.geometry.coordinates[0].forEach(coord => {
                    bounds.extend(coord);
                });
            }
        });

        // Initialize map with calculated bounds
        mapRef.current = new mapboxgl.Map({
            container: mapContainerRef.current,
            style: 'mapbox://styles/mapbox/dark-v11',
            bounds: bounds,
            fitBoundsOptions: { padding: PADDING }
        });

        mapRef.current.on('load', () => {
            // Add the GeoJSON source
            mapRef.current.addSource('floor-data', {
                type: 'geojson',
                data: selectedBuilding.geoJSON
            });
    
            // // Add a layer to render the GeoJSON data
            // mapRef.current.addLayer({
            //     id: 'floor-layer',
            //     type: 'fill',
            //     source: 'floor-data',
            //     paint: {
            //         'fill-color': '#252525', // Light gray
            //         'fill-opacity': 1
            //     }
            // });
    
            // Add a layer to paint any Rentable rooms green
            mapRef.current.addLayer({
                id: 'floor-layer',
                type: 'fill',
                source: 'floor-data',
                paint: {
                    // Conditionally set the fill color based on Rentable property
                    'fill-color': [
                        'case',
                        ['==', ['get', 'Rentable'], true], // Check if Rentable is true
                        '#28a745', // Green if rentable
                        '#252525' // Default gray if not rentable
                    ],
                    'fill-opacity': 1
                }
            });

             // Add an outline layer
             mapRef.current.addLayer({
                id: 'floor-outline',
                type: 'line',
                source: 'floor-data',
                paint: {
                    'line-color': '#f8f8f8',
                    'line-width': 1
                }
            });


            // Add clickable points for each room -- will eventually be a part of the geoJSON
            selectedBuilding.geoJSON.features.forEach((feature, index) => {
                // if(feature.properties.Rentable){
                //     const coordinates = feature.geometry.coordinates[0];
                //     const center = coordinates.reduce((acc, coord) => {
                //         return [acc[0] + coord[0], acc[1] + coord[1]];
                //     }, [0, 0]).map(sum => sum / coordinates.length);

                //     const el = document.createElement('div'); // creating the dots for each room
                //     el.className = 'room-marker';
                //     el.style.backgroundColor = '#28a745';
                //     el.style.width = '12px';
                //     el.style.height = '12px';
                //     el.style.borderRadius = '50%';
                //     el.style.cursor = 'pointer';

                //     // handles click on each room
                //     el.addEventListener('click', () => {
                //         setSelectedRoom({
                //             name: feature.properties.Name || `Room ${index + 1}`,
                //             hours: feature.properties.Hours || 'Not specified',
                //             lastEntry: feature.properties.LastEntry || 'No recent entries',
                //             lockBattery: feature.properties.LockBattery || 'Unknown'
                //         });
                //     });

                //     new mapboxgl.Marker(el)
                //         .setLngLat(center)
                //         .addTo(mapRef.current);
                // }
                if (feature.geometry.type === 'Polygon') {
                    const coordinates = feature.geometry.coordinates[0];
                    const center = coordinates.reduce((acc, coord) => {
                        return [acc[0] + coord[0], acc[1] + coord[1]];
                    }, [0, 0]).map(sum => sum / coordinates.length);

                    const el = document.createElement('div'); // creating the dots for each room
                    el.className = 'room-marker';
                    el.style.backgroundColor = '#007bff';
                    el.style.width = '12px';
                    el.style.height = '12px';
                    el.style.borderRadius = '50%';
                    el.style.cursor = 'pointer';

                    // handles click on each room
                    el.addEventListener('click', () => {
                        setSelectedRoom({
                            name: feature.properties.Name || `Room ${index + 1}`,
                            hours: feature.properties.Hours || 'Not specified',
                            lastEntry: feature.properties.LastEntry || 'No recent entries',
                            lockBattery: feature.properties.LockBattery || 'Unknown'
                        });
                    });

                    new mapboxgl.Marker(el)
                        .setLngLat(center)
                        .addTo(mapRef.current);
                }
            });

            // Set max bounds with padding
            // const maxBounds = bounds.toArray();
            // const sw = mapRef.current.project(maxBounds[0]);
            // const ne = mapRef.current.project(maxBounds[1]);
            // const paddedSw = mapRef.current.unproject([sw.x - PADDING, sw.y + PADDING]);
            // const paddedNe = mapRef.current.unproject([ne.x + PADDING, ne.y - PADDING]);
            // mapRef.current.setMaxBounds(new mapboxgl.LngLatBounds(paddedSw, paddedNe));

            // Set max bounds based on the initial viewport
            const initialBounds = mapRef.current.getBounds();
            mapRef.current.setMaxBounds(initialBounds);

            // Set minimum zoom level
            const initialZoom = mapRef.current.getZoom();
            mapRef.current.setMinZoom(initialZoom - 0.5); // Allow slight zoom out

            // Update debug info
            setDebugInfo({
                startingZoom: mapRef.current.getZoom(),
                startingFitBounds: initialBounds.toString(),
                startingMinZoom: mapRef.current.getMinZoom(),
                currentMaxBounds: mapRef.current.getMaxBounds().toString(),
                currentMinZoom: mapRef.current.getMinZoom(),
                currentZoom: mapRef.current.getZoom(),
            });

            // Add move event listener to update current zoom
            mapRef.current.on('move', () => {
                setDebugInfo(prevInfo => ({
                    ...prevInfo,
                    currentZoom: mapRef.current.getZoom(),
                    currentMaxBounds: mapRef.current.getMaxBounds().toString(),
                    currentMinZoom: mapRef.current.getMinZoom(),
                }));
            });

        });
        
        return () => {
            if (mapRef.current) {
                mapRef.current.remove();
            }
        };
    }, [selectedBuilding]);

    function handleFloorSelection(event) {
        const buildingId = event.target.value;
        const selected = buildings.find(building => building.id === buildingId);
        setSelectedBuilding(selected);
    }

    return (
        <div className="outer-container">
            <div className='inner-container'>
                <div className="dropdown-container">
                    <select
                        value={selectedBuilding ? selectedBuilding.id : ""}
                        className='dropdown-menu'
                        onChange={handleFloorSelection}
                    >
                        <option value="">Select a Floor Plan</option>
                        {buildings.map((building) => (
                            <option key={building.id} value={building.id}>
                                {building.name}
                            </option>
                        ))} 
                    </select>
                </div>
                <div ref={mapContainerRef} className="mapbox-container" />
                {selectedRoom && (
                    <RoomModal
                        room={selectedRoom}
                        onClose={() => setSelectedRoom(null)}
                    />
                )}
                <div className="debug-overlay" style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    background: 'rgba(0,0,0,0.7)',
                    color: 'white',
                    padding: '10px',
                    borderRadius: '5px',
                    fontSize: '12px',
                    maxWidth: '300px',
                    zIndex: 1000,
                }}>
                   
                </div>
            </div>
        </div>
    );}

export default ConferenceRooms;

// import React, { useRef, useEffect, useState } from 'react';
// import mapboxgl from 'mapbox-gl';
// import 'mapbox-gl/dist/mapbox-gl.css';
// import RoomModal from './RoomModal';
// import geoJSONCollection from '../assets/floorMap';

// const AMERICAN_CENTER = [-100, 40];

// function MapboxContainer() {
//     const mapRef = useRef();
//     const mapContainerRef = useRef();
//     const [selectedRoom, setSelectedRoom] = useState(null);
//     const [selectedBuilding, setSelectedBuilding] = useState(null);
//     const [buildings, setBuildings] = useState([]);

//     useEffect(() => {
//         // Parse geoJSONCollection to create the buildings array
//         const buildingsArray = Object.entries(geoJSONCollection).map(([id, data]) => ({
//           id,
//           name: data.name,
//           coordinates: JSON.parse(data.map_coordinates),
//           geoJSON: data
//         }));
//         setBuildings(buildingsArray);
//     }, []);

//     useEffect(() => {     
//         mapboxgl.accessToken = 'pk.eyJ1IjoianVzdGluYmlsZG5lciIsImEiOiJjbTIyM3c0azUwMnllMmxuNzFwY3V0Y204In0.n4K--AHy82A8xh9JxeLUnw';

//         mapRef.current = new mapboxgl.Map({
//             container: mapContainerRef.current,
//             style: 'mapbox://styles/mapbox/dark-v11',
//             center: selectedBuilding ? selectedBuilding.coordinates : AMERICAN_CENTER,
//             zoom: selectedBuilding ? 18.2 : 2.9
//         });

//         mapRef.current.on('load', () => {
//             // Add the GeoJSON source
//             mapRef.current.addSource('floor-data', {
//                 type: 'geojson',
//                 data: selectedBuilding.geoJSON
//             });
    
//             // Add a layer to render the GeoJSON data
//             mapRef.current.addLayer({
//                 id: 'floor-layer',
//                 type: 'fill',
//                 source: 'floor-data',
//                 paint: {
//                     'fill-color': '#252525', // Light gray
//                     'fill-opacity': 1
//                 }
//             });
    
//             // Add an outline layer
//             mapRef.current.addLayer({
//                 id: 'floor-outline',
//                 type: 'line',
//                 source: 'floor-data',
//                 paint: {
//                     'line-color': '#f8f8f8',
//                     'line-width': 1
//                 }
//             });

//             // Add clickable points for each room -- will eventually be a part of the geoJSON
//             selectedBuilding.geoJSON.features.forEach((feature, index) => {
//                 if (feature.geometry.type === 'Polygon') {
//                     const coordinates = feature.geometry.coordinates[0];
//                     const center = coordinates.reduce((acc, coord) => {
//                         return [acc[0] + coord[0], acc[1] + coord[1]];
//                     }, [0, 0]).map(sum => sum / coordinates.length);

//                     const el = document.createElement('div'); // creating the dots for each room
//                     el.className = 'room-marker';
//                     el.style.backgroundColor = '#007bff';
//                     el.style.width = '12px';
//                     el.style.height = '12px';
//                     el.style.borderRadius = '50%';
//                     el.style.cursor = 'pointer';

//                     // handles click on each room
//                     el.addEventListener('click', () => {
//                         setSelectedRoom({
//                             name: feature.properties.Name || `Room ${index + 1}`,
//                             hours: feature.properties.Hours || 'Not specified',
//                             lastEntry: feature.properties.LastEntry || 'No recent entries',
//                             lockBattery: feature.properties.LockBattery || 'Unknown'
//                         });
//                     });

//                     new mapboxgl.Marker(el)
//                         .setLngLat(center)
//                         .addTo(mapRef.current);
//                 }
//             });
//         });
        
//         return () => {
//             if (mapRef.current) {
//                 mapRef.current.remove();
//             }
//         };
//     }, [selectedBuilding]);

//     function handleFloorSelection(event) {
//         const buildingId = event.target.value;
//         const selected = buildings.find(building => building.id === buildingId);
//         setSelectedBuilding(selected);
//     }

//     return (
//         <div className="outer-container">
//             <div className='inner-container'>
//                 <div className="dropdown-container">
//                     <select
//                         value={selectedBuilding ? selectedBuilding.id : ""}
//                         className='dropdown-menu'
//                         onChange={handleFloorSelection}
//                     >
//                         <option value="">Select a Floor Plan</option>
//                         {buildings.map((building) => (
//                             <option key={building.id} value={building.id}>
//                                 {building.name}
//                             </option>
//                         ))} 
//                     </select>
//                 </div>
//                 <div ref={mapContainerRef} className="mapbox-container" />
//                 {selectedRoom && (
//                     <RoomModal
//                         room={selectedRoom}
//                         onClose={() => setSelectedRoom(null)}
//                     />
//                 )}
//             </div>
//         </div>
//     );}

// export default MapboxContainer;
import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import RoomModal from './RoomModal';
import geoJSONCollection from '../assets/floorMap';
import locksGeoJSON from '../assets/locks';

const AMERICAN_CENTER = [-100, 40];
const PADDING = 50;

function TimeSeries() {
    const mapRef = useRef();
    const mapContainerRef = useRef();
    const [selectedRoom, setSelectedRoom] = useState(null);
    const [selectedBuilding, setSelectedBuilding] = useState(null);
    const [buildings, setBuildings] = useState([]);
    const [time, setTime] = useState(12); // Initial time

    useEffect(() => {
        // Parse geoJSONCollection to create the buildings array
        const buildingsArray = Object.entries(geoJSONCollection).map(([id, data]) => ({
          id,
          name: data.name,
          coordinates: JSON.parse(data.map_coordinates),
          geoJSON: data
        }));
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
    
            // Add a layer to render the GeoJSON data
            mapRef.current.addLayer({
                id: 'floor-layer',
                type: 'fill',
                source: 'floor-data',
                paint: {
                    'fill-color': '#252525', // Light gray
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

            // Add the GeoJSON data as a source for the locks
            console.log('Loading GeoJSON data:', locksGeoJSON);
            mapRef.current.addSource('locks', {
                type: 'geojson',
                data: locksGeoJSON,
            });

            // Verify the presence of the 'hour' property in the GeoJSON data
            locksGeoJSON.features.forEach(feature => {
                if (typeof feature.properties.hour !== 'number') {
                    console.error('Invalid or missing "hour" property in feature:', feature);
                }
            });

            // Add a circle layer to render the lock data
            console.log('Adding circle layer for locks');
            mapRef.current.addLayer({
                id: 'locks-circles',
                type: 'circle',
                source: 'locks',
                paint: {
                    'circle-radius':  [
                        'interpolate',
                        ['linear'],
                        ['get', 'intensity'],
                        1, 5, // Minimum intensity, minimum radius
                        10, 20 // Maximum intensity, maximum radius
                    ],
                    'circle-color': [
    'interpolate',
    ['linear'],
    ['get', 'intensity'], // Get the intensity property from the feature
    1, 'green',  // Minimum intensity, green color
    5, 'yellow', // Medium intensity, yellow color
    10, 'red'   // Maximum intensity, red color
],

                    'circle-stroke-color': 'white',
                    'circle-stroke-width': 1,
                    'circle-opacity': 0.8
                },
                filter: ['==', ['number', ['get', 'hour']], time] // Initial filter based on time
            });

            // Set max bounds based on the initial viewport
            const initialBounds = mapRef.current.getBounds();
            mapRef.current.setMaxBounds(initialBounds);

            // Set minimum zoom level
            const initialZoom = mapRef.current.getZoom();
            mapRef.current.setMinZoom(initialZoom - 0.5); // Allow slight zoom out

            // Add move event listener to update current zoom
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

    const handleTimeChange = (event) => {
        const hour = parseInt(event.target.value);
        setTime(hour);
        console.log('Updating filter for hour:', hour);
        // Update the map filter
        if (mapRef.current.getLayer('locks-circles')) {
            mapRef.current.setFilter('locks-circles', ['==', ['number', ['get', 'hour']], hour]);
        }

        // Convert 0-23 hour to AMPM format
        const ampm = hour >= 12 ? 'PM' : 'AM';
        const hour12 = hour % 12 ? hour % 12 : 12;

        // Update text in the UI
        document.getElementById('active-hour').innerText = hour12 + ampm;
    };

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
                    <h3>Time Series Visualization</h3>
                    <div className="session" id="sliderbar">
                        <h2>Hour: <label id="active-hour">12PM</label></h2>
                        <input
                            id="slider"
                            className="row"
                            type="range"
                            min="0"
                            max="23"
                            step="1"
                            value={time}
                            onChange={handleTimeChange}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default TimeSeries;
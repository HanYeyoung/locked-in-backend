import React, { useEffect } from 'react';
import mapboxgl from 'mapbox-gl';
import locksGeoJSON from '../../assets/locks';

const PADDING = 50;
const ANIM_DUR = 3000;

const MapInitialization = ({ mapRef, selectedBuilding, mapInitialized, markersRef, setSelectedRoom, setDebugInfo, showTimeSeries, showRoomNames, time, setTime, setSelectedRoomHover }) => {
  
    // clears markers when new floor plan is selected  
    const clearMarkers = () => {
        if (markersRef.current) {
          markersRef.current.forEach(marker => marker.remove());
          markersRef.current = [];
        } 
    };

    // resets to initial world view when no floor plan is selected
    const resetToInitialView = () => {
        if (!mapRef.current) return;

        clearMarkers();

        // Remove existing layers and sources
        const layersToRemove = ['floor-layer', 'floor-outline', 'locks-circles'];
        layersToRemove.forEach(layer => {
            if (mapRef.current.getStyle() && mapRef.current.getLayer(layer)) {
                mapRef.current.removeLayer(layer);
            }
        });

        const sourcesToRemove = ['floor-data', 'locks'];
        sourcesToRemove.forEach(source => {
            if (mapRef.current.getStyle() && mapRef.current.getSource(source)) {
                mapRef.current.removeSource(source);
            }
        });

        // Reset bounds and zoom
        mapRef.current.setMaxBounds(null);
        mapRef.current.setMinZoom(null);

        mapRef.current.flyTo({
            center: [-100, 40],
            zoom: 2.9,
            duration: ANIM_DUR,
            essential: true
        });
    };

    // adds layers to the map view to create floor plans and features
    const addMapLayers = () => {
        if (!mapRef.current || !mapRef.current.getStyle()) return;

        // Add floor data source if it doesn't exist
        if (!mapRef.current.getSource('floor-data')) {
            mapRef.current.addSource('floor-data', {
                type: 'geojson',
                data: selectedBuilding.geoJSON
            });
        }

        // Add floor layers if they don't exist
        if (!mapRef.current.getLayer('floor-layer')) {
            mapRef.current.addLayer({
                id: 'floor-layer',
                type: 'fill',
                source: 'floor-data',
                paint: {
                    'fill-color': '#252525', //light gray
                    'fill-opacity': 1
                }
            });
        }

        if (!mapRef.current.getLayer('floor-outline')) {
            mapRef.current.addLayer({
                id: 'floor-outline',
                type: 'line',
                source: 'floor-data',
                paint: {
                    'line-color': '#f8f8f8',
                    'line-width': 1
                }
            });

        }


        // Remove the highlight layer before the source, if they exist
        if (mapRef.current.getLayer('highlight-layer')) {
            mapRef.current.removeLayer('highlight-layer');
        }
        if (mapRef.current.getSource('highlighted-room')) {
            mapRef.current.removeSource('highlighted-room');
        }

        // Add the highlight source for room highlighting
        mapRef.current.addSource('highlighted-room', {
            type: 'geojson',
            data: {
                type: 'FeatureCollection',
                features: []
            }
        });

        // Add the highlight layer for room highlighting
        mapRef.current.addLayer({
            id: 'highlight-layer',
            type: 'fill',
            source: 'highlighted-room',
            paint: {
                'fill-color': '#4a4a4a', // Dark gray color for highlight
                'fill-opacity': 0.6
            }
        });

        

        // Add locks data if it doesn't exist
        if (!mapRef.current.getSource('locks')) {
            mapRef.current.addSource('locks', {
                type: 'geojson',
                data: locksGeoJSON
            });
        }

        // // Verify the presence of the 'hour' property in the GeoJSON data
        // locksGeoJSON.features.forEach(feature => {
        //     if (typeof feature.properties.hour !== 'number') {
        //     console.error('Invalid or missing "hour" property in feature:', feature);
        //     }
        // });
        
        // Add locks visualization if it doesn't exist
        if (!mapRef.current.getLayer('locks-circles')) {
            mapRef.current.addLayer({
                id: 'locks-circles',
                type: 'circle',
                source: 'locks',
                paint: {
                    'circle-radius': [
                        'interpolate',
                        ['linear'],
                        ['get', 'intensity'],
                        1, 10,
                        10, 30
                    ],
                    'circle-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'intensity'],
                        1, 'green',
                        5, 'yellow',
                        10, 'red'
                    ],
                    'circle-stroke-color': 'white',
                    'circle-stroke-width': 1,
                    'circle-opacity': 0.8
                },
                layout: {
                    'visibility': showTimeSeries ? 'visible' : 'none'
                },
                filter: ['==', ['number', ['get', 'hour']], time]
            });
        }
    };

    // adds a clickable marker for each room
    const addRoomMarkers = () => {
        if (!mapRef.current || !selectedBuilding) return;

        selectedBuilding.geoJSON.features.forEach((feature, index) => {
            if (feature.geometry.type === 'Polygon') {
                const coordinates = feature.geometry.coordinates[0];
                const center = coordinates.reduce(
                (acc, coord) => [acc[0] + coord[0], acc[1] + coord[1]],
                [0, 0]
            ).map(sum => sum / coordinates.length);


            if (showRoomNames){
                const el = document.createElement('div');
                el.className = 'room-box';
                
                // Edit colors based on renting conditions: JUST BTW: These conditional IFS should be changed after Justin's work with renting room is done
                if (feature.properties.Rentable) {
                    if (feature.properties.Rented) {
                        el.style.backgroundColor = '#ff0000';
                        el.style.color = '#ffffff';
                    } else if (feature.properties.Approval_Needed) {
                        el.style.backgroundColor = '#FFD966';
                        el.style.color = '#000000';
                    } else {
                        el.style.backgroundColor = '#28a745';
                        el.style.color = '#ffffff';
                    }
                } else {
                    el.style.backgroundColor = '#007bff';
                    el.style.color = '#ffffff';
                }

                el.style.padding = '5px 10px';
                el.style.borderRadius = '8px';
                el.style.cursor = 'pointer';
                el.style.textAlign = 'center';
                el.style.fontSize = '12px';
                el.textContent = feature.properties.Name || `Room ${index + 1}`;

                el.addEventListener('click', (event) => {
                    const rect = event.target.getBoundingClientRect();
                    setSelectedRoom({
                        name: feature.properties.Name || `Room ${index + 1}`,
                        hours: feature.properties.Hours || 'Not specified',
                        lastEntry: feature.properties.LastEntry || 'No recent entries',
                        lockBattery: feature.properties.LockBattery || 'Unknown',
                        clickCoords: {
                            x: rect.x,
                            y: rect.y
                        }
                    });
                });
    
                // Add hover and click event listeners to the element
                // el.addEventListener('mouseenter', () => updateHighlight(feature));
                // el.addEventListener('mouseleave', clearHighlight);
                el.addEventListener('mouseover', () => {
                    updateHighlight(feature);
                    const markerPosition = mapRef.current.project(center);
                    setSelectedRoomHover({
                        name: feature.properties.Name || `Room ${index + 1}`,
                        hours: feature.properties.Hours || 'Not specified',
                        lastEntry: feature.properties.LastEntry || 'No recent entries',
                        lockBattery: feature.properties.LockBattery || 'Unknown',
                        x: markerPosition.x,
                        y: markerPosition.y,
                        color: el.style.backgroundColor
                    });
                });
    
                // handle mouse leave to remove hover
                el.addEventListener('mouseleave', () => {
                    setTimeout(() => {
                        setSelectedRoomHover(null);
                        clearHighlight();
                    }, 200); 
                });

                const marker = new mapboxgl.Marker(el)
                .setLngLat(center)
                .addTo(mapRef.current);

                markersRef.current.push(marker);
            }else{
                const el = document.createElement('div');
                el.className = 'room-marker';
                el.style.cssText = `
                    background-color: #007bff;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    cursor: pointer;
                `;
                // Edit colors based on renting conditions: JUST BTW: These conditional IFS should be changed after Justin's work with renting room is done
                if (feature.properties.Rentable) {
                    if (feature.properties.Rented) {
                        el.style.backgroundColor = '#ff0000';
                        el.style.color = '#ffffff';
                    } else if (feature.properties.Approval_Needed) {
                        el.style.backgroundColor = '#FFD966';
                        el.style.color = '#000000';
                    } else {
                        el.style.backgroundColor = '#28a745';
                        el.style.color = '#ffffff';
                    }
                } else {
                    el.style.backgroundColor = '#007bff';
                    el.style.color = '#ffffff';
                }
                el.addEventListener('click', (event) => {
                    const rect = event.target.getBoundingClientRect();
                    setSelectedRoom({
                        name: feature.properties.Name || `Room ${index + 1}`,
                        hours: feature.properties.Hours || 'Not specified',
                        lastEntry: feature.properties.LastEntry || 'No recent entries',
                        lockBattery: feature.properties.LockBattery || 'Unknown',
                        clickCoords: {
                            x: rect.x,
                            y: rect.y
                        },
                        color: el.style.backgroundColor
                    });
                });
    
                // Add hover and click event listeners to the element
                // el.addEventListener('mouseenter', () => updateHighlight(feature));
                // el.addEventListener('mouseleave', clearHighlight);
                el.addEventListener('mouseover', () => {
                    updateHighlight(feature);
                    const markerPosition = mapRef.current.project(center);
                    setSelectedRoomHover({
                        name: feature.properties.Name || `Room ${index + 1}`,
                        hours: feature.properties.Hours || 'Not specified',
                        lastEntry: feature.properties.LastEntry || 'No recent entries',
                        lockBattery: feature.properties.LockBattery || 'Unknown',
                        x: markerPosition.x,
                        y: markerPosition.y,
                        color: el.style.backgroundColor
                    });
                });
    
                // handle mouse leave to remove hover
                el.addEventListener('mouseleave', () => {
                    setTimeout(() => {               
                        // Clear hover state if not interacting with modal or marker
                        setSelectedRoomHover(null);
                        clearHighlight();
                    }, 200); // Delay for smoother transitions
                });

                const marker = new mapboxgl.Marker(el)
                .setLngLat(center)
                .addTo(mapRef.current);

                markersRef.current.push(marker);
            }

            
          }
        });
    };

    // Functions for updating and clearing highlight
    const updateHighlight = (feature) => {
        mapRef.current.getSource('highlighted-room').setData({
            type: 'FeatureCollection',
            features: [feature]
        });
    };

    const clearHighlight = () => {
        mapRef.current.getSource('highlighted-room').setData({
            type: 'FeatureCollection',
            features: []
        });
    };
    

    // Effect for handling building selection changes
    useEffect(() => {
        if (!mapInitialized || !mapRef.current) return;

        if (!selectedBuilding) {
            resetToInitialView();
            return;
        }

        // Calculate bounds
        const bounds = new mapboxgl.LngLatBounds();
        selectedBuilding.geoJSON.features.forEach(feature => {
            if (feature.geometry.type === 'Polygon') {
                feature.geometry.coordinates[0].forEach(coord => {
                    bounds.extend(coord);
                });
            }
        });

        clearMarkers();

        // Remove existing layers and sources
        const layersToRemove = ['floor-layer', 'floor-outline', 'locks-circles']; //might need to chnage
        layersToRemove.forEach(layer => {
            if (mapRef.current.getLayer(layer)) {
                mapRef.current.removeLayer(layer);
            }
        });
        
        if (mapRef.current.getSource('floor-data')) {
            mapRef.current.removeSource('floor-data');
        }

        // Animate to new bounds
        mapRef.current.fitBounds(bounds, {
            padding: PADDING,
            duration: ANIM_DUR,
            essential: true
        });

        // Add layers and markers after animation
        setTimeout(() => {
            if (!mapRef.current) return;

            addMapLayers();
            addRoomMarkers();

            // Set bounds and zoom constraints
            const initialBounds = mapRef.current.getBounds();
            mapRef.current.setMaxBounds(initialBounds);

            const initialZoom = mapRef.current.getZoom();
            mapRef.current.setMinZoom(initialZoom - 0.5);

            // // Update debug info
            // setDebugInfo({
            //     startingZoom: mapRef.current.getZoom(),
            //     startingFitBounds: initialBounds.toString(),
            //     startingMinZoom: mapRef.current.getMinZoom(),
            //     currentMaxBounds: mapRef.current.getMaxBounds().toString(),
            //     currentMinZoom: mapRef.current.getMinZoom(),
            //     currentZoom: mapRef.current.getZoom(),
            // });

            //  // Add move event listener to update current zoom
            //  mapRef.current.on('move', () => {
            //     setDebugInfo(prevInfo => ({
            //         ...prevInfo,
            //         currentZoom: mapRef.current.getZoom(),
            //         currentMaxBounds: mapRef.current.getMaxBounds().toString(),
            //         currentMinZoom: mapRef.current.getMinZoom(),
            //     }));
            // });

        }, ANIM_DUR / 2);

    }, [selectedBuilding, mapInitialized]);

    // Effect for handling time changes
    useEffect(() => {
        if (mapRef.current && mapRef.current.getStyle() && mapRef.current.getLayer('locks-circles')) {
            mapRef.current.setFilter('locks-circles', ['==', ['number', ['get', 'hour']], time]);
            // console.log('Updated filter for hour:', time);
        }
    }, [time]);
    
    // Effect for handling time series visibility
    useEffect(() => {
        if (!mapRef.current || !mapRef.current.getStyle()) return;

        if (showTimeSeries) {
            setTime(12); // Set default hour to 12
            // console.log('Time series visualization enabled, setting default hour to 12');
            // // Update the map filter
            // if (mapRef.current && mapRef.current.getLayer('locks-heatmap')) {
            //     mapRef.current.setFilter('locks-heatmap', ['==', ['number', ['get', 'hour']], 12]);
            // }
        }

        if (mapRef.current.getLayer('locks-circles')) {
            const visibility = showTimeSeries ? 'visible' : 'none';
            mapRef.current.setLayoutProperty('locks-circles', 'visibility', visibility);
            console.log(`Heatmap visibility set to: ${visibility}`);
        }
    }, [showTimeSeries]);

    useEffect(() => {
        if (!mapRef.current || !mapRef.current.getStyle()) return;
        clearMarkers();
        addRoomMarkers();

    }, [showRoomNames]);

    return null;
};

export default MapInitialization;
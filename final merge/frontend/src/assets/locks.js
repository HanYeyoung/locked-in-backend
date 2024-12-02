const locksGeoJSON = {
    type: "FeatureCollection",
    features: [
      // Building Services Office
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "Building Services Office",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 5) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40806945399994, 43.072113804666714]
        }
      })),
      // Service and Storage
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "Service and Storage",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 5) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40801754671426, 43.071965014714344]
        }
      })),
      // Truck Bay
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "Truck Bay",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 5) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40789142636359, 43.07183004490915]
        }
      })),
      // Men's Restroom
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "Men's Restroom",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 5) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40763996519996, 43.07171140300005]
        }
      })),
      // Lower Climbing Wall
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "Lower Climbing Wall",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 5) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40754081599997, 43.07169336580006]
        }
      })),
      // Bowling
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "Bowling",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 5) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40758938819995, 43.07162557580006]
        }
      })),
      // The Sett
      ...Array.from({ length: 13 }, (_, i) => ({
        type: "Feature",
        properties: {
          name: "The Sett",
          hour: 8 + i,
          intensity: Math.floor(Math.random() * 7) + 1 // Random intensity between 1 and 5
        },
        geometry: {
          type: "Point",
          coordinates: [-89.40762849599996, 43.07157998600006]
        }
      }))
    ]
  };
  
  export default locksGeoJSON;
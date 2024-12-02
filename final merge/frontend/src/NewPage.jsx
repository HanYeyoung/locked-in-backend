import React, { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const NewPage = () => {
  const mapContainerRef = useRef(null);

  useEffect(() => {
    mapboxgl.accessToken = 'pk.eyJ1IjoianVzdGluYmlsZG5lciIsImEiOiJjbTIyM3c0azUwMnllMmxuNzFwY3V0Y204In0.n4K--AHy82A8xh9JxeLUnw';
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: [-74.0059, 40.7128],
      zoom: 12,
    });

    map.on('load', () => {
      let filterHour = ['==', ['number', ['get', 'Hour']], 12];
      let filterDay = ['!=', ['string', ['get', 'Day']], 'placeholder'];

      map.addLayer({
        id: 'collisions',
        type: 'circle',
        source: {
          type: 'geojson',
          data: './collisions1601.geojson', // replace this with the url of your own geojson
        },
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['number', ['get', 'Casualty']],
            0,
            4,
            5,
            24,
          ],
          'circle-color': [
            'interpolate',
            ['linear'],
            ['number', ['get', 'Casualty']],
            0,
            '#2DC4B2',
            1,
            '#3BB3C3',
            2,
            '#669EC4',
            3,
            '#8B88B6',
            4,
            '#A2719B',
            5,
            '#AA5E79',
          ],
          'circle-opacity': 0.8,
        },
        filter: ['all', filterHour, filterDay],
      });

      document.getElementById('slider').addEventListener('input', (event) => {
        const hour = parseInt(event.target.value);
        filterHour = ['==', ['number', ['get', 'Hour']], hour];
        map.setFilter('collisions', ['all', filterHour, filterDay]);

        const ampm = hour >= 12 ? 'PM' : 'AM';
        const hour12 = hour % 12 ? hour % 12 : 12;
        document.getElementById('active-hour').innerText = hour12 + ampm;
      });

      document.getElementById('filters').addEventListener('change', (event) => {
        const day = event.target.value;
        if (day === 'all') {
          filterDay = ['!=', ['string', ['get', 'Day']], 'placeholder'];
        } else if (day === 'weekday') {
          filterDay = ['match', ['get', 'Day'], ['Sat', 'Sun'], false, true];
        } else if (day === 'weekend') {
          filterDay = ['match', ['get', 'Day'], ['Sat', 'Sun'], true, false];
        } else {
          console.error('error');
        }
        map.setFilter('collisions', ['all', filterHour, filterDay]);
      });
    });

    return () => map.remove();
  }, []);

  return (
    <div>
      <div id="map" ref={mapContainerRef} style={{ width: '100%', height: '400px' }}></div>
      <div id="console">
        <h1>Motor vehicle collisions</h1>
        <p>
          Data:
          <a href="https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95">
            Motor vehicle collision injuries and deaths
          </a>
          in NYC, Jan 2016
        </p>
        <div className="session">
          <h2>Casualty</h2>
          <div className="row colors"></div>
          <div className="row labels">
            <div className="label">0</div>
            <div className="label">1</div>
            <div className="label">2</div>
            <div className="label">3</div>
            <div className="label">4</div>
            <div className="label">5+</div>
          </div>
        </div>
        <div className="session" id="sliderbar">
          <h2>Hour: <label id="active-hour">12PM</label></h2>
          <input
            id="slider"
            className="row"
            type="range"
            min="0"
            max="23"
            step="1"
            value="12"
          />
        </div>
        <div className="session">
          <h2>Day of the week</h2>
          <div className="row" id="filters">
            <input id="all" type="radio" name="toggle" value="all" checked="checked" />
            <label htmlFor="all">All</label>
            <input id="weekday" type="radio" name="toggle" value="weekday" />
            <label htmlFor="weekday">Weekday</label>
            <input id="weekend" type="radio" name="toggle" value="weekend" />
            <label htmlFor="weekend">Weekend</label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewPage;
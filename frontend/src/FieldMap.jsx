import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';

function FieldMap() {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const drawnItemsRef = useRef(null);
  const [coordInput, setCoordInput] = useState('');
  const [error, setError] = useState('');
  const [fieldCount, setFieldCount] = useState(0);
  const [panelOpen, setPanelOpen] = useState(false);

  useEffect(() => {
    if (mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current).setView([26.9124, 75.7873], 8);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);

    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    drawnItemsRef.current = drawnItems;

    const drawControl = new L.Control.Draw({
      draw: {
        polygon: true,
        polyline: false,
        rectangle: false,
        circle: false,
        marker: false,
        circlemarker: false,
      },
      edit: {
        featureGroup: drawnItems,
      },
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, (event) => {
      const layer = event.layer;
      drawnItems.addLayer(layer);
      setFieldCount(drawnItems.getLayers().length);
      console.log('Field boundary drawn:', layer.toGeoJSON());
    });

    map.on(L.Draw.Event.EDITED, (event) => {
      event.layers.eachLayer((layer) => {
        console.log('Field boundary edited:', layer.toGeoJSON());
      });
    });

    map.on(L.Draw.Event.DELETED, () => {
      setFieldCount(drawnItems.getLayers().length);
      console.log('Field(s) deleted. Remaining:', drawnItems.getLayers().length);
    });

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  const handleLoadCoordinates = () => {
    setError('');
    try {
      const parsed = JSON.parse(coordInput);

      if (!Array.isArray(parsed) || parsed.length < 3) {
        setError('Kam se kam 3 coordinate pairs chahiye');
        return;
      }

      const polygon = L.polygon(parsed, { color: 'red' });
      drawnItemsRef.current.addLayer(polygon);
      mapInstanceRef.current.fitBounds(polygon.getBounds());
      setFieldCount(drawnItemsRef.current.getLayers().length);

      console.log('Manual field loaded:', polygon.toGeoJSON());
      setCoordInput('');
    } catch (err) {
      setError('Invalid format. Example: [[26.92,75.79],[26.921,75.792],[26.919,75.793]]');
    }
  };

  const handleClearAll = () => {
    if(fieldCount==0) return;
    const confirmClear = window.confirm('Are you sure you want to clear all fields?');
    if(confirmClear){
       drawnItemsRef.current.clearLayers();
      setFieldCount(0);
    }
   
  };

  return (
    <div style={{ position: 'relative', height: '100vh', width: '100%' }}>
      <button
        onClick={() => setPanelOpen(!panelOpen)}
        style={{
          position: 'absolute',
          top: '10px',
          left: '50px',
          zIndex: 1001,
          padding: '8px 12px',
          borderRadius: '6px',
          border: 'none',
          background: '#2c7fb8',
          color: 'white',
          fontWeight: 'bold',
          cursor: 'pointer',
          boxShadow: '0 1px 5px rgba(0,0,0,0.4)',
        }}
      >
        {panelOpen ? '✕ Close' : '☰ Field Tools'}
      </button>

      {panelOpen && (
        <div
          style={{
            position: 'absolute',
            top: '55px',
            left: '50px',
            zIndex: 1000,
            background: 'white',
            padding: '12px',
            borderRadius: '6px',
            boxShadow: '0 1px 5px rgba(0,0,0,0.4)',
            width: '320px',
          }}
        >
          <p style={{ margin: '0 0 6px 0', fontWeight: 'bold', fontSize: '13px' }}>
            Option A: Draw on map
          </p>
          <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#555' }}>
            Use the polygon tool to draw a field boundary directly.
          </p>

          <hr style={{ margin: '10px 0' }} />

          <p style={{ margin: '0 0 6px 0', fontWeight: 'bold', fontSize: '13px' }}>
            Option B: Load known coordinates
          </p>
          <textarea
            value={coordInput}
            onChange={(e) => setCoordInput(e.target.value)}
            placeholder="[[26.92,75.79],[26.921,75.792],[26.919,75.793]]"
            rows={3}
            style={{ width: '100%', fontSize: '12px', boxSizing: 'border-box' }}
          />
          <button onClick={handleLoadCoordinates} style={{ marginTop: '6px', width: '100%' }}>
            Load Field Coordinates
          </button>
          {error && <p style={{ color: 'red', fontSize: '12px', marginTop: '4px' }}>{error}</p>}

          <hr style={{ margin: '10px 0' }} />

          <p style={{ margin: '0 0 6px 0', fontSize: '12px' }}>
            Fields on map: <strong>{fieldCount}</strong>
          </p>
          <button onClick={handleClearAll} style={{ width: '100%', background: '#f5b3b3' }}>
            Clear All Fields
          </button>
        </div>
      )}

      <div ref={mapContainerRef} style={{ height: '100%', width: '100%' }} />
    </div>
  );
}

export default FieldMap;
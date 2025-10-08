// static/js/main.js

let map;
let lastId = 0;
let markers = [];
let pathCoords = [];        // array of google.maps.LatLng for polyline
let polyline = null;

function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 20.5937, lng: 78.9629 },
    zoom: 4,
  });

  // Create polyline and add to map
  polyline = new google.maps.Polyline({
    path: pathCoords,
    geodesic: true,
    strokeOpacity: 0.9,
    strokeWeight: 3,
    map: map
  });

  // Start polling
  setInterval(fetchNewCoords, 3000);
  fetchNewCoords(); // initial load
}

async function fetchNewCoords() {
  try {
    const res = await fetch(`/api/coords?after_id=${lastId}`);
    const data = await res.json();
    console.log("fetched coords:", data);
    data.forEach(coord => {
      addMarker(coord);
      lastId = Math.max(lastId, coord.id);
    });
  } catch (err) {
    console.error("Error fetching coords:", err);
  }
}

function addMarker(c) {
  if (!map) {
    console.warn("Map not initialized yet, skipping marker", c);
    return;
  }

  const position = { lat: Number(c.lat), lng: Number(c.lng) };

  const marker = new google.maps.Marker({
    position,
    map: map,
    label: String(c.id)
  });

  // InfoWindow content: show id, timestamp and meta
  const infoContent = `
    <div style="min-width:140px">
      <strong>Marker ${c.id}</strong><br/>
      <small>${c.ts ? new Date(c.ts).toLocaleString() : ""}</small><br/>
      <div>${c.meta ? escapeHtml(c.meta) : ""}</div>
    </div>`;

  const infowindow = new google.maps.InfoWindow({
    content: infoContent
  });

  marker.addListener("click", () => {
    infowindow.open(map, marker);
  });

  // Optional: open info for the latest marker automatically
  // infowindow.open(map, marker);

  markers.push(marker);

  // Add to polyline path and update map
  pathCoords.push(position);
  polyline.setPath(pathCoords);

  // On first added marker, center and zoom in
  if (markers.length === 1) {
    map.setCenter(position);
    map.setZoom(12);
  }
}

// small helper to avoid HTML injection if meta contains HTML characters
function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>"'`=\/]/g, function (s) {
    return ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
      '/': '&#x2F;',
      '=': '&#x3D;',
      '`': '&#x60;'
    })[s];
  });
}

mapboxgl.accessToken = "pk.eyJ1IjoidGhlLWRlc3Ryb3llciIsImEiOiJjbTdkbWd1ZjIwMDJ3MmpxdXp3dWNpb3VlIn0.GtirZBfgEJOCgwsM9ZB0Zg"; // <--- PASTE YOUR TOKEN HERE

const map = new mapboxgl.Map({
  container: "map",
  style: "mapbox://styles/mapbox/satellite-streets-v12",
  center: [78.9629, 20.5937],
  zoom: 4,
  projection: "globe",
});
map.addControl(
  new MapboxGeocoder({
    accessToken: mapboxgl.accessToken,
    mapboxgl: mapboxgl,
    marker: false,
    placeholder: "Search location...",
  }),
  "top-left"
);
map.addControl(new mapboxgl.NavigationControl(), "bottom-right");

function switchMode(mode) {
  document
    .querySelectorAll(".nav-btn")
    .forEach((b) => b.classList.remove("active"));
  document.getElementById(`btn-${mode}`).classList.add("active");
  document.getElementById("view-aerial").style.display =
    mode === "aerial" ? "block" : "none";
  document.getElementById("view-species").style.display =
    mode === "species" ? "block" : "none";
  const titles = {
    aerial: "🛰️ Aerial Tree Counting",
    species: "🧬 Species Identifier",
  };
  document.getElementById("panelTitle").innerText = titles[mode];
  if (document.getElementById("mainPanel").classList.contains("minimized"))
    togglePanel();
}

function togglePanel() {
  const panel = document.getElementById("mainPanel");
  const icon = document.getElementById("toggleIcon");
  panel.classList.toggle("minimized");
  icon.classList = panel.classList.contains("minimized")
    ? "bi bi-plus-lg"
    : "bi bi-dash-lg";
}

// Handle Uploads
function setupUpload(inputId, loaderId, resultId, imgId, endpoint, callback) {
  document.getElementById(inputId).addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById(loaderId).style.display = "block";
    document.getElementById(resultId).style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence", document.getElementById("confSlider").value);

    try {
      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server Error");
      document.getElementById(imgId).src =
        "data:image/jpeg;base64," + data.image_data;
      document.getElementById(resultId).style.display = "block";
      callback(data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      document.getElementById(loaderId).style.display = "none";
    }
  });
}

setupUpload(
  "aerialFile",
  "aerialLoader",
  "aerialResult",
  "aerialImg",
  "/detect",
  (d) => (document.getElementById("countVal").innerText = d.count)
);
setupUpload(
  "speciesFile",
  "speciesLoader",
  "speciesResult",
  "speciesImg",
  "/classify",
  (d) => {
    document.getElementById("speciesName").innerText = d.species;
    document.getElementById("speciesConf").innerText = d.confidence + "%";
  }
);

// NEW: Report Issue Logic
async function reportIssue(mode) {
  const inputId = mode === "aerial" ? "aerialFile" : "speciesFile";
  const fileInput = document.getElementById(inputId);

  if (!fileInput.files.length) {
    alert("No image to report!");
    return;
  }

  if (
    !confirm(
      "Report this image as incorrect? It will be saved for future model retraining."
    )
  )
    return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("mode", mode);

  try {
    const res = await fetch("/feedback", { method: "POST", body: formData });
    const data = await res.json();
    if (res.ok) alert("✅ Report Sent! Image saved for retraining.");
    else alert("❌ Error reporting: " + data.error);
  } catch (e) {
    alert("Network error: " + e.message);
  }
}

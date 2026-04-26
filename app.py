import warnings
# Suppress tifffile GDAL_NODATA tag parsing warnings (cosmetic noise from GeoTIFF metadata)
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*", category=UserWarning)
warnings.filterwarnings("ignore", module="tifffile")

import sys
import json
import streamlit as st
import folium
from streamlit_folium import st_folium
import torch
import ee
import requests
import io
import numpy as np
import tifffile
import math
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

# --- Path Setup ---
# Ensure the src/ package is importable in all environments (local + HF Spaces)
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from carbon_tracker.models.resnet import get_multispectral_resnet
from carbon_tracker.utils.physics import biomass_to_carbon, carbon_to_co2_equivalent

# Load environment variables (local .env file — ignored on HF where Secrets are used)
load_dotenv()

# --- Geographic Constraints (Oise, France) ---
REGION_MIN_LAT = 49.0
REGION_MAX_LAT = 49.8
REGION_MIN_LON = 2.0
REGION_MAX_LON = 3.5
REGION_CENTER_LAT = 49.4
REGION_CENTER_LON = 2.75

# --- Configuration & Initialization ---

st.set_page_config(
    page_title="Carbon Tracker — Forest Biomass Estimation",
    page_icon="🌲",
    layout="wide",
)

# Initialize session state for map coordinates and zoom
if "target_lat" not in st.session_state:
    st.session_state.target_lat = REGION_CENTER_LAT
if "target_lon" not in st.session_state:
    st.session_state.target_lon = REGION_CENTER_LON
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 10


@st.cache_resource
def init_earth_engine():
    """
    Initializes the Google Earth Engine API.

    Supports two authentication modes:
    - Production (Hugging Face): Service Account key stored in GEE_SERVICE_ACCOUNT_KEY secret.
    - Local development: Standard OAuth2 credentials from ~/.config/earthengine/.
    """
    try:
        project_id = os.environ.get("EE_PROJECT_ID")
        sa_key_json = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")

        if sa_key_json:
            # --- Production mode: Service Account authentication ---
            sa_info = json.loads(sa_key_json)
            sa_email = sa_info.get("client_email")
            if not sa_email:
                st.error("GEE_SERVICE_ACCOUNT_KEY is set but missing 'client_email' field.")
                return False
            credentials = ee.ServiceAccountCredentials(
                email=sa_email, key_data=sa_key_json
            )
            ee.Initialize(credentials=credentials, project=project_id)
        else:
            # --- Local development mode: standard OAuth2 credentials ---
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()

        return True

    except Exception as e:
        st.error(f"Failed to initialize Google Earth Engine: {e}")
        return False


@st.cache_resource
def load_model():
    """
    Loads the custom ResNet-18 model and its trained weights.
    Path is resolved relative to this file to be CWD-independent.
    """
    weights_path = ROOT_DIR / "models" / "weights" / "best_model_carbon.pth"

    if not weights_path.exists():
        st.warning(
            f"Model weights not found at `{weights_path}`. Running in **mock mode** "
            "(a fixed placeholder value will be returned instead of a real prediction)."
        )
        return None

    model = get_multispectral_resnet()
    model.load_state_dict(
        torch.load(str(weights_path), map_location=torch.device("cpu"))
    )
    model.eval()
    return model


# --- Core Logic ---

def get_bounding_box(lat: float, lon: float, size_km: float = 2.5) -> list:
    """
    Calculates the South-West and North-East coordinates of a square bounding box.
    """
    half_side = size_km / 2.0
    delta_lat = half_side / 111.32
    delta_lon = half_side / (111.32 * math.cos(math.radians(lat)))
    sw = [lat - delta_lat, lon - delta_lon]
    ne = [lat + delta_lat, lon + delta_lon]
    return [sw, ne]


def is_within_training_region(lat: float, lon: float) -> bool:
    """
    Validates if the target coordinates fall within the geographical limits
    used during model training (Oise region, France).
    """
    return (REGION_MIN_LAT <= lat <= REGION_MAX_LAT) and (
        REGION_MIN_LON <= lon <= REGION_MAX_LON
    )


def fetch_sentinel2_tensor(lat: float, lon: float, size_km: float = 2.5) -> torch.Tensor:
    """
    Fetches a 256x256 Sentinel-2 image patch (RGB + NIR) from GEE.
    Normalizes the surface reflectance and formats it for PyTorch inference.
    """
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(size_km * 1000 / 2).bounds()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    if collection.size().getInfo() == 0:
        raise ValueError(
            "No cloud-free Sentinel-2 images found for this location within the last year."
        )

    image = collection.first().select(["B4", "B3", "B2", "B8"])

    url = image.getDownloadURL(
        {
            "region": roi,
            "dimensions": "256x256",
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
    )

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with io.BytesIO(response.content) as f:
            img_array = tifffile.imread(f)

    img_array = img_array.astype(np.float32) / 10000.0

    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = np.transpose(img_array, (2, 0, 1))

    tensor = torch.tensor(img_array).unsqueeze(0)
    return tensor


# --- UI Layout ---

st.title("🌲 Carbon Tracker — Forest Biomass Estimation")
st.markdown(
    "Estimate forest carbon stock using **Sentinel-2 satellite imagery** and **Deep Learning**."
)
st.info(
    "⚠️ This model was trained exclusively on data from the **Oise region, France**. "
    "Inference is restricted to this geographical area to prevent erroneous predictions."
)

gee_ready = init_earth_engine()
model = load_model()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Select a target area")

    m = folium.Map(
        location=[st.session_state.target_lat, st.session_state.target_lon],
        zoom_start=st.session_state.map_zoom,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    # Valid training region boundary
    folium.Rectangle(
        bounds=[[REGION_MIN_LAT, REGION_MIN_LON], [REGION_MAX_LAT, REGION_MAX_LON]],
        color="#3776AB",
        fill=False,
        weight=2,
        dash_array="5, 5",
        tooltip="Valid Inference Zone (Oise Region)",
    ).add_to(m)

    # 2.5×2.5 km analysis patch around the selected point
    patch_bounds = get_bounding_box(
        st.session_state.target_lat, st.session_state.target_lon
    )
    folium.Rectangle(
        bounds=patch_bounds,
        color="#FF4B4B",
        fill=True,
        fill_opacity=0.2,
        weight=2,
        tooltip="2.5 km × 2.5 km Analysis Patch",
    ).add_to(m)

    folium.Marker(
        [st.session_state.target_lat, st.session_state.target_lon],
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    map_data = st_folium(m, width=700, height=500)

    if map_data:
        if map_data.get("zoom"):
            st.session_state.map_zoom = map_data["zoom"]

        if map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]

            if (
                clicked_lat != st.session_state.target_lat
                or clicked_lon != st.session_state.target_lon
            ):
                st.session_state.target_lat = clicked_lat
                st.session_state.target_lon = clicked_lon
                st.rerun()

with col2:
    st.subheader("Analysis & Inference")

    lat = st.session_state.target_lat
    lon = st.session_state.target_lon
    st.write(f"**Coordinates:**\n- Lat: `{lat:.4f}`\n- Lon: `{lon:.4f}`")

    if model is None:
        st.caption("🔶 Mock mode active — no model weights loaded.")

    if st.button("Run Estimation", type="primary", use_container_width=True):
        if not is_within_training_region(lat, lon):
            st.error(
                "Target coordinates are outside the valid training region. "
                "Please select a point **within the blue dashed rectangle**."
            )
        elif not gee_ready:
            st.error(
                "Earth Engine is not initialized. Please check your credentials "
                "(EE_PROJECT_ID and GEE_SERVICE_ACCOUNT_KEY environment variables)."
            )
        else:
            with st.spinner(
                "Fetching Sentinel-2 data from Earth Engine & running inference…"
            ):
                try:
                    input_tensor = fetch_sentinel2_tensor(lat, lon)

                    if model is not None:
                        with torch.no_grad():
                            prediction = model(input_tensor)
                            agb_stock = prediction.item()
                    else:
                        agb_stock = 309.15

                    carbon_stock = biomass_to_carbon(agb_stock)
                    co2_equivalent = carbon_to_co2_equivalent(carbon_stock)

                    st.success("Analysis complete ✅")

                    st.markdown("### 📊 Metrics (per hectare)")

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(label="Biomass (AGB)", value=f"{agb_stock:.1f} t")
                    mc2.metric(label="Carbon Stock", value=f"{carbon_stock:.1f} t")
                    mc3.metric(label="CO₂ Eq.", value=f"{co2_equivalent:.1f} t")

                    with st.expander("ℹ️ How are these values calculated?"):
                        st.markdown(
                            """
                            - **Biomass (AGB):** Directly predicted by the modified ResNet-18 model
                              using 4-channel (RGB + NIR) surface reflectance data.
                            - **Carbon Stock:** Calculated using the standard IPCC factor
                              ($AGB \\times 0.47$).
                            - **CO₂ Equivalent:** Converted using the atomic mass ratio of CO₂ to
                              Carbon ($Carbon \\times 3.67$).
                            """
                        )

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

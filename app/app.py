import streamlit as st
import folium
import torch
import traceback
import math
from streamlit_folium import st_folium
from carbon_tracker.models.resnet import get_multispectral_resnet

st.set_page_config(page_title="Carbon Tracker AI", layout="wide")

@st.cache_resource
def load_trained_model():
    model = get_multispectral_resnet()
    weights_path = "models/weights/best_model_carbon.pth"
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Load the model into memory
carbon_model = load_trained_model()

st.title("🌲 Forest Carbon Stock Estimation")
st.markdown("""
This application uses a **Multispectral ResNet-18** to estimate biomass and carbon stock 
from Sentinel-2 satellite imagery.
""")

# Sidebar for coordinates
with st.sidebar:
    st.header("Location Settings")
    lat = st.number_input("Latitude", value=49.4147, format="%.4f")
    lon = st.number_input("Longitude", value=2.8262, format="%.4f")
    
    if st.button("Estimate Carbon Stock"):
        with st.spinner("Running AI Model Inference..."):
            try:
                # Mock input tensor: [batch_size=1, channels=4, height=256, width=256]
                mock_satellite_patch = torch.randn(1, 4, 256, 256)
                
                with torch.no_grad():
                    predicted_biomass = carbon_model(mock_satellite_patch).item()
                    
                predicted_carbon = predicted_biomass * 0.47
                
                st.session_state['biomass'] = predicted_biomass
                st.session_state['carbon'] = predicted_carbon
                st.success("Analysis complete!")
                
            except Exception as e:
                st.error(f"Inference Error: {e}")
                st.code(traceback.format_exc())

# --- MAP RENDERING (Satellite & Patch Visualization) ---

# Calculate approximate boundaries for a 2.5km x 2.5km patch (1280m buffer)
# 1 degree of latitude is ~111.32 km. 
lat_offset = 1280 / 111320.0
# 1 degree of longitude depends on latitude.
lon_offset = 1280 / (111320.0 * math.cos(math.radians(lat)))

# Define the Southwest and Northeast corners of the bounding box
bounds = [
    [lat - lat_offset, lon - lon_offset], 
    [lat + lat_offset, lon + lon_offset]
]

# Initialize map with Esri World Imagery (Satellite View)
m = folium.Map(
    location=[lat, lon], 
    zoom_start=14, # Zoomed in closer to see the patch
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery'
)

# Add center marker
folium.Marker([lat, lon], popup="Center Coordinate").add_to(m)

# Draw the exact analyzed patch as a red rectangle
folium.Rectangle(
    bounds=bounds,
    color="#FF0000",
    fill=True,
    fill_opacity=0.15,
    weight=2,
    popup="Analyzed Satellite Patch (256x256 pixels at 10m/px)"
).add_to(m)

st_folium(m, width=700, height=450, returned_objects=[])

# --- METRICS DISPLAY ---
col1, col2 = st.columns(2)

display_biomass = f"{st.session_state['biomass']:.2f} t/ha" if 'biomass' in st.session_state else "--- t/ha"
display_carbon = f"{st.session_state['carbon']:.2f} tC/ha" if 'carbon' in st.session_state else "--- tC/ha"

col1.metric("Estimated Biomass", display_biomass)
col2.metric("Carbon Stock", display_carbon)
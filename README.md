---
title: Carbon Tracker
emoji: 🌲
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
python_version: 3.11.8
---

<div align="center">

<h2> 🌳 CarbonTracker : AI-Driven Forest Biomass Estimation </h2>

![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![](https://img.shields.io/badge/Google_Earth_Engine-4285F4?style=for-the-badge&logo=google-earth&logoColor=white)
![](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

[👉 Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Nico-GTR/carbon-tracker-oise)

</div>

## 💡 Overview
CarbonTracker is a Machine Learning pipeline built with PyTorch and Google Earth Engine, designed to estimate forest carbon stocks from satellite imagery.

Given a location in the **Oise region, France**, the app:
1. Dynamically fetches a **Sentinel-2** (RGB + NIR) image patch from GEE
2. Runs it through a custom **ResNet-18** (4-channel input, regression output)
3. Returns **Above-Ground Biomass (AGB)**, **Carbon Stock**, and **CO₂ Equivalent** per hectare

> **Note on Model Scope:** The current model was trained on ground-truth data from the Oise region (temperate deciduous forests). Inference is geographically locked to this area to prevent erroneous predictions.

## ✨ Features
- **🌍 Live Satellite Extraction:** Fetches Copernicus Sentinel-2 data on the fly via Google Earth Engine.
- **🧠 Deep Learning Inference:** Custom ResNet-18 modified for 4-channel spectral inputs and regression.
- **🗺️ Interactive Mapping:** Folium map with Esri satellite imagery, valid zone boundary, and analysis patch overlay.
- **⚗️ Physical Conversions:** IPCC-standard factors to convert raw biomass predictions into carbon metrics.

## 👩‍💻 Tech Stack
- **Python (>= 3.10)** — Core language
- **PyTorch** — Deep learning framework
- **Google Earth Engine** — Cloud geospatial API
- **Streamlit & Folium** — Interactive web interface

## 📖 Sources
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/) — Sentinel-2 Level-2A imagery
- [ESA CCI Biomass](https://climate.esa.int/en/projects/biomass/) — Above-Ground Biomass ground truth
- [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)

## 📦 Local Setup

### Prerequisites
- Python >= 3.10
- `uv` (or `pip`)
- A Google Cloud Project with Earth Engine API enabled

### Installation

```bash
git clone https://github.com/Nico-GTR/cnn_carbon.git
cd cnn_carbon

# With uv
uv venv && uv pip install -e .

# Or with pip
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file at the project root:

```env
EE_PROJECT_ID="your-gcp-project-id"
# GEE_SERVICE_ACCOUNT_KEY is only needed in production (HF Spaces)
```

Then authenticate with Earth Engine locally:

```bash
earthengine authenticate
```

### Run locally

```bash
# Entry point is now at the root
uv run streamlit run app.py
```

## 🚀 Hugging Face Spaces Deployment

The app is configured to run on HF Spaces with:
- `app.py` at the root (entry point)
- `requirements.txt` for dependencies
- `.streamlit/config.toml` for production settings (port 7860)

**Required HF Secrets:**
| Secret | Value |
|--------|-------|
| `EE_PROJECT_ID` | Your GCP project ID |
| `GEE_SERVICE_ACCOUNT_KEY` | Full JSON content of the Service Account key |

## 📜 License
Distributed under the MIT License.

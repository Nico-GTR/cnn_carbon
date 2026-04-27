---
title: Carbon Tracker
emoji: 🌲
colorFrom: green
colorTo: blue
sdk: streamlit
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

## � Getting Started (Beginner Friendly)

Want to run this project on your own machine? Follow these simple steps!

### 1. Prerequisites
- **Python 3.10 or 3.11** installed on your computer.
- A **Google Earth Engine account**. If you don't have one, [sign up here](https://earthengine.google.com/).
- Git (optional, but recommended).

### 2. Clone the Repository
Open your terminal (or Command Prompt) and run:
```bash
git clone https://github.com/Nico-GTR/cnn_carbon.git
cd cnn_carbon
```

### 3. Set up a Virtual Environment
It's highly recommended to use a virtual environment so you don't mess up your global Python packages.
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install all the required libraries:
```bash
pip install -r requirements.txt
```

### 5. Authenticate with Google Earth Engine
Before running the app, you need to tell Google Earth Engine who you are:
```bash
earthengine authenticate
```
*This will open a browser window. Log in with your Google account and allow the permissions. Note: If you are setting this up for a server or Hugging Face, you will need a Service Account Key instead.*

### 6. Run the App
Start the Streamlit interface:
```bash
streamlit run app.py
```
Your browser will automatically open `http://localhost:8501` where you can interact with the app!

## 📁 Project Structure
- `app.py`: The main Streamlit web application.
- `src/carbon_tracker/`: The core python package.
  - `data/`: Code to extract Sentinel-2 data and build datasets.
  - `models/`: PyTorch model architectures (ResNet-18).
  - `training/`: Training loops and logic.
  - `utils/`: Physics math to convert Biomass to CO2 equivalents.
- `models/weights/`: Saved trained PyTorch models (`.pth`).

## �📖 Sources
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

<div align="center">

<h2> 🌳 CarbonTracker : AI-Driven Forest Biomass Estimation </h2>

![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![](https://img.shields.io/badge/Google_Earth_Engine-4285F4?style=for-the-badge&logo=google-earth&logoColor=white)
![](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

[👉 Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Nico-GTR/Forest-Carbon-Tracker)

</div>

## 💡 Overview
CarbonTracker is a powerful and intuitive Machine Learning pipeline built using PyTorch and Google Earth Engine, designed to streamline forest carbon monitoring. 

With a focus on scalability and environmental impact, CarbonTracker allows for the instant estimation of **Above-Ground Biomass (AGB)**, **Carbon Stocks**, and **CO₂ Equivalent** directly from satellite imagery.

> **Note on Model Scope:** To ensure scientific accuracy, the current ResNet-18 model was strictly trained on ground-truth data from the Oise region in France (temperate deciduous forests). Inference via the web interface is geographically locked to this area to prevent erroneous predictions in unsupported biomes (e.g., tropical or boreal forests).

## ✨ Features
- **🌍 Live Satellite Extraction:** Dynamically fetches and processes Copernicus Sentinel-2 data (RGB + Near-Infrared) on the fly via the Google Earth Engine API.
- **🧠 Advanced Deep Learning:** Utilizes a custom ResNet-18 architecture, modified specifically for 4-channel spectral inputs and regression tasks.
- **🗺️ Interactive Mapping:** Visualizes the exact 2.5km² spatial footprint analyzed by the model, overlaying a high-resolution Esri satellite view.
- **⚗️ Physical Conversions:** Implements standard IPCC conversion factors to translate raw predicted biomass into actionable climate metrics (Carbon & CO₂ Eq).
- **🚀 Production Ready:** Packaged with `uv`, modular codebase, decoupled training/inference pipelines, and ready for Docker containerization.

Whether you're a data scientist, an environmental researcher, or part of a climate-tech initiative, CarbonTracker serves as a robust proof-of-concept for end-to-end remote sensing engineering. 🛰️

## 👩‍💻 Tech Stack
- **Python (>= 3.10)**: Core programming language for data engineering and model training.
- **PyTorch**: Deep learning framework used for adapting and training the Convolutional Neural Network.
- **Google Earth Engine (GEE)**: Cloud-based geospatial API used to extract and composite multispectral satellite patches without heavy local storage.
- **Streamlit & Folium**: Fast web framework and mapping integration for building the interactive user interface.
- **Package Management**: Managed seamlessly using `uv` and `pyproject.toml`.

## 📖 Sources and External Data
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/) for Sentinel-2 Level-2A optical imagery.
- [ESA Climate Change Initiative (CCI)](https://climate.esa.int/en/projects/biomass/) for the global Above-Ground Biomass ground truth dataset.
- [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets) for accessing and processing Earth observation data.

## 📦 Getting Started
To get a local copy of this project up and running, follow these steps.

### 🚀 Prerequisites
- **Python** (v3.10 or higher).
- **uv** (Extremely fast Python package installer and resolver).
- A **Google Cloud Project** with the Earth Engine API enabled.

### 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/carbon-tracker.git](https://github.com/YOUR_USERNAME/carbon-tracker.git)
   cd carbon-tracker
   ```

2. **Set up the virtual environment and install dependencies:**
   Using `uv`, this creates an isolated environment and installs the package in editable mode.
   ```bash
   uv venv
   
   # Activate the environment (Windows)
   .venv\Scripts\activate
   # Or on Mac/Linux: source .venv/bin/activate
   
   uv pip install -e .
   ```

3. **Set up Google Earth Engine Authentication:**
   You must configure your Earth Engine credentials and project ID.
   First, authenticate locally via the command line:
   ```bash
   earthengine authenticate
   ```
   Then, create a `.env` file at the root of the project and specify your Google Cloud Project ID:
   
```env
   EE_PROJECT_ID="your-gcp-project-id"
   ```

4. **Ensure Model Availability:**
   Make sure the pre-trained weights file (`best_model_carbon.pth`) is located in the `models/weights/` directory.

## 📖 Usage

### ✔️ Running the Web Application
To launch the Streamlit interface locally:
```bash
uv run streamlit run app.py
```
> Open http://localhost:8501 to view the app in your browser.

### ✔️ Running the Training Pipeline
To extract fresh data from the Oise region and train the model from scratch:
```bash
uv run python train_pipeline.py
```

## 🤝 Contributing
Contributions to extend the model's validity to other biomes are highly welcome!
1. **Fork the repository.**
2. **Create a new branch** (`git checkout -b feature/your-feature-name`).
3. **Make your changes** and commit them (`git commit -m 'Add some feature'`).
4. **Push to the branch** (`git push origin feature/your-feature-name`).
5. **Open a pull request**.

## 🐛 Issues
If you encounter any issues while using or setting up the project, please open a new issue detailing the problem. Include the environment details (OS, Python version).

## 📜 License
Distributed under the MIT License.

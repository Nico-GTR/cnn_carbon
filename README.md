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
CarbonTracker is a powerful and intuitive Machine Learning pipeline built using PyTorch and Google Earth Engine, designed to streamline forest carbon monitoring. With a focus on scalability and environmental impact, CarbonTracker allows for the instant estimation of Above-Ground Biomass (AGB) and equivalent carbon stocks directly from satellite imagery.

## ✨ Features
- **🌍 Live Satellite Extraction:** Dynamically fetches and processes Copernicus Sentinel-2 data (including Near-Infrared) anywhere in the world.
- **🧠 Advanced Deep Learning:** Utilizes a custom ResNet18 architecture modified for 4-channel spectral inputs and regression tasks.
- **🗺️ Interactive Mapping:** Visualizes the exact analyzed spatial footprint overlaying a high-resolution Google Maps satellite view.
- **🎲 Random Zone Exploration:** Built-in tool to instantly generate coordinates in relevant forested areas for quick testing.
- **🎯 Physical Safeguards:** Algorithm constraints guarantee logical real-world physical outputs.
- **🚀 Production Ready:** Containerized with Docker and fully deployed for real-time inference.

Whether you're a data scientist, an environmental researcher, or part of a climate-tech initiative, CarbonTracker is the perfect proof-of-concept for remote sensing analysis. 🛰️

## 👩‍💻 Tech Stack
- **Python**: Core programming language for data engineering and model training.
- **PyTorch**: Deep learning framework used for adapting and training the ResNet18 Convolutional Neural Network.
- **Google Earth Engine (GEE)**: Cloud-based geospatial processing engine used to extract and composite multispectral satellite patches.
- **Streamlit**: A fast, Python-based web framework for building the interactive user interface.
- **Folium**: Python wrapper for Leaflet.js, handling the interactive map rendering.
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

3. **Set up environment variables:**
   You must configure your Earth Engine credentials. Create a `.env` file at the root of the project:
   ```env
   EARTHENGINE_TOKEN={"type": "service_account", "project_id": "...", "private_key": "...", "client_email": "..."}
   ```

4. **Ensure Model Availability:**
   Make sure the pre-trained weights file (`best_model_carbon.pth`) is located in the `models/weights/` directory.

## 📖 Usage

### ✔ Running the Web Application
To launch the Streamlit interface locally:
```bash
uv run streamlit run app/app.py
```
> Open http://localhost:8501 to view the app in your browser.

### ✔ Running the Training Pipeline
To extract data and train the model from scratch:
```bash
uv run python train_pipeline.py
```

## 🤝 Contributing
I welcome contributions to this project! Please follow these steps to contribute:
1. **Fork the repository.**
2. **Create a new branch** (`git checkout -b feature/your-feature-name`).
3. **Make your changes** and commit them (`git commit -m 'Add some feature'`).
4. **Push to the branch** (`git push origin feature/your-feature-name`).
5. **Open a pull request**.

## 🐛 Issues
If you encounter any issues while using or setting up the project, please open a new issue detailing the problem. Include the environment details (OS, Python version).

## 📜 License
Distributed under the MIT License.

# Tirupati Forest Fire Prediction & Spread Explorer

A runnable research and demonstration project integrating multi-source satellite remote sensing, classical machine learning classifiers, 2D U-Net deep learning spatial segmentation, and Cellular Automata (CA) fire propagation simulation for the Seshachalam Biosphere Reserve in the Tirupati region.

## Dataset & Architecture

- **Data Source**: Real multi-source geospatial and meteorological rasters extracted via Google Earth Engine (`data/tirupati_real_fire_dataset_2026.csv`), featuring MODIS BurnDate DOY, Sentinel-2 NDVI, MODIS Land Surface Temperature (LST), NASA GPM IMERG precipitation, ERA5 climate reanalysis, and SRTM digital elevation models.
- **Models & Performance Benchmarks**:
  - **Random Forest**: 97.20% Accuracy (0.94 F1-Score)
  - **XGBoost**: 96.50% Accuracy (0.93 F1-Score)
  - **Decision Tree**: 95.80% Accuracy (0.92 F1-Score)
  - **Logistic Regression**: 91.70% Accuracy (0.89 F1-Score)
  - **U-Net Deep Learning**: 93.11% Accuracy (0.90 F1-Score)
- **Cellular Automata (CA) Simulator**: Models directional fire propagation across 1, 3, 6, and 12-hour simulation horizons driven by slope inclination, wind azimuth vectors, and NDVI fuel density.

## Quick Start (Windows / VS Code)

1. Open the project folder containing `app.py` and `requirements.txt` in VS Code.
2. Open **Terminal → New Terminal**, then create and activate a clean virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Run the master training and evaluation pipeline to generate models and plots in `outputs/`:

   ```powershell
   python main.py
   ```

4. Launch the interactive Streamlit decision support web application:

   ```powershell
   python -m streamlit run app.py
   ```

5. Open the local URL printed in your terminal (typically `http://localhost:8501`) to explore the prediction dashboard, U-Net risk segmenter, CA spread simulator, and model benchmarks. Press **Ctrl+C** to stop the server.

## Project Structure

```text
app.py                     Streamlit interactive dashboard
main.py                    Master execution pipeline
src/
  ├── preprocess.py        GEE data loading & feature normalization
  ├── train_ml.py          Tabular classifier training & evaluation
  ├── train_unet.py        2D U-Net deep learning spatial segmentation
  └── ca_simulation.py     Cellular Automata directional fire spread engine
outputs/
  ├── models/              Saved joblib models, scaler, and PyTorch weights (.pth)
  └── plots/               Confusion matrices, ROC curves, feature importance & rasters
data/                      Real Tirupati GEE dataset CSV
requirements.txt           Pinned Python dependencies
```

## Disclaimer
This project is developed as a Major Project Dissertation (ES-452) for academic evaluation. While trained on real satellite data and validated against research benchmarks, operational deployment requires real-time meteorological IoT feeds and field validation.
# Tirupati Forest Fire Prediction & Spread Explorer

A runnable research/demo project using the supplied Tirupati CSV. The Streamlit app trains Logistic Regression, Decision Tree, Random Forest and XGBoost automatically the first time it opens, reports held-out accuracy and F1, and provides a numerical-input prediction form, an optional U-Net patch demonstration, and cellular-automata spread scenarios at 1, 3, 6 and 12 discrete steps.

## Dataset and limitations

`data/tirupati_forest_fire_balanced_synthetic_mar_aug_2026.csv` contains 1,200 balanced synthetic records (600 fire / 600 non-fire) for March–August 2026. Columns include coordinates, NDVI, land-surface temperature, precipitation, wind, humidity, elevation, slope, aspect, MODIS burn-date DOY, and `fire_occurrence`.

The input is tabular point data, not georeferenced raster imagery or an independent real-world incident archive. The retrospective `MODIS_BurnDate_DOY` column is retained in the CSV but excluded from predictors because a burn date is not available before an event and can leak the target. The U-Net reshapes coordinate-sorted feature rows into pseudo-patches and uses point labels as masks; its pixels are not a real image grid. Its score and predicted-mask view demonstrate model mechanics only. The CA grid is likewise a conceptual simulation driven by sampled environmental covariates, not a calibrated forecast; each step is not physically calibrated to one hour. These are not validated operational fire forecasts. Use real raster layers, incident timelines and field validation before any operational use.

The reference repository concerns Algerian data; this implementation uses only the supplied Tirupati CSV.

## Quick start (Windows / VS Code)

1. If you downloaded the ZIP, extract it. In VS Code choose **File → Open Folder…** and open `tirupati_forest_fire_project` (the folder containing `app.py`). The CSV is already in `data/`.
2. Install Python 3.11 if needed. If this is an update over an older copy, close its terminal and delete the old `.venv` folder in Explorer first. Open **Terminal → New Terminal** in VS Code, then create and activate a clean environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install --no-cache-dir -r requirements.txt
   ```

   This installs the web app and tabular models; XGBoost is an optional classifier dependency handled by the trainer. The U-Net is optional because PyTorch has a large platform-specific install.

3. Start the app:

   ```powershell
   streamlit run app.py
   ```

   On its first launch, the app automatically trains the classifiers and saves their fitted models and `artifacts/metrics.json`. You do not need to run a separate training command. First launch takes longer; later launches reuse the saved models. Training uses a stratified 80/20 holdout split and reports accuracy, F1, precision, recall, ROC-AUC and confusion matrices. This split is for demonstration; spatial/time-aware validation is recommended for research claims.

4. Open the local URL printed in the terminal (usually `http://localhost:8501`). The **Forest Fire Prediction** tab opens first. Enter the environmental values, choose Random Forest or XGBoost, and select **Estimate fire occurrence**. The other tabs show the dataset overview, spread simulation and model results. The local server stays live while that terminal process is running; press **Ctrl+C** to stop it.

5. Optional U-Net patch experiment (PyTorch required):

   ```powershell
   pip install -r requirements-unet.txt
   python -m src.train_unet
   ```

## Project map

```text
app.py                   Streamlit prediction and CA dashboard
src/data.py              CSV loading and feature definitions
src/train_models.py      Four classifier pipelines and evaluation
src/ca.py                Wind/slope/vegetation-aware cellular automaton
src/unet.py              Small U-Net architecture for derived patches
src/train_unet.py        Patch building and U-Net training/evaluation
data/                    Supplied Tirupati CSV
artifacts/               Generated models and metrics
```

XGBoost is handled as an optional dependency and reported as unavailable if it cannot be imported. This repository is not production-ready: it has no deployment configuration, monitoring, calibration, security review, or validation against independent real-world incidents.

If you are updating from an older ZIP or see a pandas/NumPy import error, close the VS Code terminal and delete the project's `.venv` folder in Explorer. Recreate it with step 2 so the pinned versions install into a clean environment, then start the app again. Manual retraining is also available with `python -m src.train_models`.


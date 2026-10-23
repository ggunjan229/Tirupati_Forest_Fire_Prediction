from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tirupati_forest_fire_balanced_synthetic_mar_aug_2026.csv"
ARTIFACTS = ROOT / "artifacts"
TARGET = "fire_occurrence"
FEATURES = [
    "latitude", "longitude", "NDVI", "LST_C", "GPM_IMERG_precip_mm_day",
    "IMD_wind_speed_m_s", "IMD_relative_humidity_pct", "elevation_m",
    "slope_deg", "aspect_deg", "month",
]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"date_yyyymmdd", TARGET, "latitude", "longitude", "NDVI", "LST_C",
                "GPM_IMERG_precip_mm_day", "IMD_wind_speed_m_s", "IMD_relative_humidity_pct",
                "elevation_m", "slope_deg", "aspect_deg", "MODIS_BurnDate_DOY"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    date = pd.to_datetime(df["date_yyyymmdd"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.copy()
    df["month"] = date.dt.month
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("fire_occurrence must contain binary labels 0 and 1")
    if len(df) < 20:
        raise ValueError("At least 20 valid rows are required")
    return df

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class TirupatiDataPreprocessor:

  def __init__(self, csv_path, output_dir="outputs"):
    self.csv_path = csv_path
    self.output_dir = output_dir
    self.scaler = StandardScaler()
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

  def load_and_align_data(self):
    """Loads real Tirupati dataset, aligns columns, and balances 1:1 via non-fire sampling."""
    df = pd.read_csv(self.csv_path)

    rename_map = {
        "precip_mm_day": "GPM_IMERG_precip_mm_day",
        "wind_speed_m_s": "IMD_wind_speed_m_s",
        "relative_humidity_pct": "IMD_relative_humidity_pct",
    }
    df = df.rename(columns=rename_map)
    df = df.interpolate(method="linear", limit_direction="both")

    # 1:1 Class Balancing (1,168 fire + 1,168 non-fire) as described in the research paper
    df_fire = df[df["fire_occurrence"] == 1]
    df_nonfire = df[df["fire_occurrence"] == 0].sample(
        n=min(len(df_fire), len(df[df["fire_occurrence"] == 0])),
        random_state=42,
    )
    df = (
        pd.concat([df_fire, df_nonfire], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    lat_bins = pd.cut(df["latitude"], bins=5, labels=False)
    lon_bins = pd.cut(df["longitude"], bins=5, labels=False)
    df["spatial_group"] = lat_bins.astype(str) + "_" + lon_bins.astype(str)
    return df

  def plot_correlation_matrix(self, df, corr_cols):
    """Generates and saves Correlation Matrix of Environmental Factors."""
    corr_matrix = df[corr_cols + ["fire_occurrence"]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    plt.title(
        "Correlation Matrix of Environmental Factors and Fire Occurrence",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(self.output_dir, "plots", "correlation_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Correlation heatmap -> {save_path}")

  def prepare_features(self, df):
    """Plots correlation matrix (including DOY) and returns leak-free environmental features."""
    env_features = [
        "NDVI",
        "LST_C",
        "GPM_IMERG_precip_mm_day",
        "IMD_wind_speed_m_s",
        "IMD_relative_humidity_pct",
        "elevation_m",
        "slope_deg",
        "aspect_deg",
    ]

    # Plot correlation heatmap including MODIS_BurnDate_DOY for visualization
    if "MODIS_BurnDate_DOY" in df.columns:
      self.plot_correlation_matrix(df, env_features + ["MODIS_BurnDate_DOY"])
    else:
      self.plot_correlation_matrix(df, env_features)

    # IMPORTANT: Do NOT include MODIS_BurnDate_DOY in X (prevents 100% target leakage)
    X = df[env_features].values
    y = df["fire_occurrence"].values
    groups = df["spatial_group"].values

    X_scaled = self.scaler.fit_transform(X)
    joblib.dump(
        self.scaler, os.path.join(self.output_dir, "models", "scaler.pkl")
    )
    return X, X_scaled, y, groups, env_features
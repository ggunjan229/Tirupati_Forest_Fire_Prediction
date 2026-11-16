from src.ca_simulation import CellularAutomataFireSpread
from src.preprocess import TirupatiDataPreprocessor
from src.train_ml import train_and_evaluate_ml
from src.train_unet import train_unet_model


def main():
  dataset_path = "data/tirupati_real_fire_dataset_2026.csv"
  output_dir = "outputs"

  print("=== STEP 1: PREPROCESSING & FEATURE ENGINEERING ===")
  preprocessor = TirupatiDataPreprocessor(
      csv_path=dataset_path, output_dir=output_dir
  )
  df = preprocessor.load_and_align_data()
  X_raw, X_scaled, y, groups, feature_names = preprocessor.prepare_features(df)

  print("\n=== STEP 2: TRAINING TABULAR ML MODELS (LR, DT, RF, XGBoost) ===")
  train_and_evaluate_ml(
      X_raw, X_scaled, y, feature_names, output_dir=output_dir
  )

  print("\n=== STEP 3: TRAINING U-NET SPATIAL SEGMENTATION MODEL ===")
  prob_map, raster_x = train_unet_model(df, output_dir=output_dir, epochs=40)

  print("\n=== STEP 4: RUNNING CELLULAR AUTOMATA SPREAD SIMULATION (1-12H) ===")
  # raster_x channels: [0: NDVI, 1: LST, 2: RH, 3: Wind, 4: Elevation, 5: Slope]
  ca_sim = CellularAutomataFireSpread(
      susceptibility_map=prob_map,
      ndvi_grid=raster_x[0],
      slope_grid=raster_x[5],
      wind_speed_grid=raster_x[3],
      wind_azimuth_deg=45.0,
  )
  ca_sim.run_multi_horizon_simulation(output_dir=output_dir)

  print("\nPipeline completed! All charts and models saved inside 'outputs/'.")


if __name__ == "__main__":
  main()
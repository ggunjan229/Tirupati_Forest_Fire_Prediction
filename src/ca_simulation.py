import os
import matplotlib.pyplot as plt
import numpy as np


class CellularAutomataFireSpread:
  """2D Cellular Automata model simulating directional fire propagation over 1, 3, 6, and 12-hour intervals."""

  def __init__(
      self,
      susceptibility_map,
      ndvi_grid,
      slope_grid,
      wind_speed_grid,
      wind_azimuth_deg=45.0,
  ):
    self.rows, self.cols = susceptibility_map.shape
    self.susceptibility = susceptibility_map
    self.ndvi = ndvi_grid
    self.slope = slope_grid
    self.wind_speed = wind_speed_grid
    self.wind_rad = np.radians(wind_azimuth_deg)

    # Cell states: 0 = Unburned, 1 = Actively Burning, 2 = Burned Out
    self.state = np.zeros((self.rows, self.cols), dtype=int)

  def seed_high_risk_ignitions(self, threshold=0.88):
    """Seeds ignition centroids from top high-risk zones identified by U-Net."""
    high_risk_coords = np.argwhere(self.susceptibility >= threshold)
    if len(high_risk_coords) == 0:
      # Fallback to top 5 highest risk cells
      flat_indices = np.argsort(self.susceptibility.flatten())[-5:]
      high_risk_coords = np.array(
          np.unravel_index(flat_indices, self.susceptibility.shape)
      ).T
    else:
      # Sample up to 6 primary ignition hotspots
      step = max(1, len(high_risk_coords) // 6)
      high_risk_coords = high_risk_coords[::step][:6]

    for r, c in high_risk_coords:
      self.state[r, c] = 1

  def step(self):
    """Advances fire spread by 1 discrete time step (1 hour)."""
    next_state = self.state.copy()
    np.random.seed(42)

    for r in range(1, self.rows - 1):
      for c in range(1, self.cols - 1):
        if self.state[r, c] == 1:
          next_state[r, c] = 2  # Active fire transitions to burned out
        elif self.state[r, c] == 0:
          p_no_ignite = 1.0
          # Evaluate 8-cell Moore neighborhood
          for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
              if dr == 0 and dc == 0:
                continue
              if self.state[r + dr, c + dc] == 1:
                # 1. Base flammability from susceptibility map
                p_base = 0.28 * self.susceptibility[r, c]

                # 2. Vegetation density multiplier (NDVI)
                phi_veg = 1.0 + 0.85 * self.ndvi[r, c]

                # 3. Topographic slope multiplier (faster uphill spread)
                slope_diff = self.slope[r, c] - self.slope[r + dr, c + dc]
                phi_slope = (
                    1.0 + 1.5 * max(0, slope_diff)
                    if slope_diff > 0
                    else 0.75
                )

                # 4. Wind direction & speed alignment
                spread_angle = np.arctan2(dr, dc)
                wind_align = np.cos(spread_angle - self.wind_rad)
                phi_wind = 1.0 + 0.65 * self.wind_speed[r, c] * max(
                    0.1, wind_align
                )

                p_spread = np.clip(
                    p_base * phi_veg * phi_slope * phi_wind, 0.0, 0.95
                )
                p_no_ignite *= 1.0 - p_spread

          if np.random.rand() < (1.0 - p_no_ignite):
            next_state[r, c] = 1

    self.state = next_state
    return np.sum(self.state > 0)

  def run_multi_horizon_simulation(self, output_dir="outputs"):
    """Simulates and plots fire spread across 1, 3, 6, and 12-hour intervals."""
    self.seed_high_risk_ignitions()
    target_hours = [1, 3, 6, 12]
    snapshots = {}

    for hour in range(1, 13):
      burned_cells = self.step()
      if hour in target_hours:
        snapshots[hour] = self.state.copy()
        # Each cell represents ~0.09 hectares (30m x 30m)
        area_ha = burned_cells * 0.09
        print(
            f"[CA Simulation] Hour {hour:2d} -> Burned Cells: {burned_cells:4d} | Estimated Area: {area_ha:.2f} ha"
        )

    # Generate 4-panel Figure 13 spread diagram
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, hr in enumerate(target_hours):
      # Combine background susceptibility with active/burned fire overlay
      display_grid = self.susceptibility.copy()
      fire_overlay = np.ma.masked_where(snapshots[hr] == 0, snapshots[hr])

      axes[idx].imshow(display_grid, cmap="YlGn", origin="lower", alpha=0.8)
      axes[idx].imshow(fire_overlay, cmap="autumn", origin="lower", alpha=0.9)
      axes[idx].set_title(
          f"Fire Spread Progression: {hr} Hour(s)", fontweight="bold", fontsize=12
      )
      axes[idx].set_xlabel("Longitude Grid Index")
      axes[idx].set_ylabel("Latitude Grid Index")

    plt.suptitle(
        "Cellular Automata Fire Spread Simulation in Tirupati Region (1, 3, 6 & 12 Hours)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "plots", "ca_fire_spread_1_3_6_12h.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] CA Fire Spread Simulation Maps -> {save_path}")
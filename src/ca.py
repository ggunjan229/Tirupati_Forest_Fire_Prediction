"""Conceptual cellular automaton for visualizing relative spread tendency."""
import numpy as np

def simulate_spread(ndvi, slope_deg, wind_m_s, aspect_deg, humidity_pct, hours=1,
                    seed_row=None, seed_col=None, ignition_count=1, seed=42):
    """Return burned-state grids at each requested hour (1, 3, 6, 12).

    Grid fields are normalized synthetic interpolations from tabular points. Wind
    direction is not in the input CSV, so the UI supplies a compass direction.
    The probability rule is illustrative, not a calibrated physical fire model.
    """
    rng = np.random.default_rng(seed)
    veg = np.asarray(ndvi, dtype=float)
    slope = np.asarray(slope_deg, dtype=float)
    if veg.ndim != 2 or slope.shape != veg.shape or not veg.size:
        raise ValueError("NDVI and slope fields must be non-empty 2-D arrays with the same shape")
    rows, cols = veg.shape
    # Start one or more ignitions close to the grid center unless explicit coords.
    burned = np.zeros((rows, cols), dtype=bool)
    if seed_row is not None and seed_col is not None:
        burned[int(seed_row) % rows, int(seed_col) % cols] = True
    else:
        for _ in range(max(1, ignition_count)):
            rr = int(np.clip(rows // 2 + rng.integers(-2, 3), 0, rows - 1))
            cc = int(np.clip(cols // 2 + rng.integers(-2, 3), 0, cols - 1))
            burned[rr, cc] = True
    # compass direction: degrees clockwise from north; 0=N, 90=E
    angle = np.deg2rad(float(aspect_deg))
    dr_wind, dc_wind = -np.cos(angle), np.sin(angle)
    snapshots = {}
    for hour in range(1, int(hours) + 1):
        frontier = np.zeros_like(burned)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0: continue
                source_r = slice(max(0, -dr), min(rows, rows - dr))
                source_c = slice(max(0, -dc), min(cols, cols - dc))
                target_r = slice(max(0, dr), min(rows, rows + dr))
                target_c = slice(max(0, dc), min(cols, cols + dc))
                frontier[target_r, target_c] |= burned[source_r, source_c]
        candidate = frontier & ~burned
        next_burned = burned.copy()
        for r, c in zip(*np.where(candidate)):
            # Wind alignment and upslope increase relative spread; humidity reduces it.
            local_slope = np.clip(slope[r, c] / 35.0, 0, 1)
            humidity_damp = np.clip(float(humidity_pct) / 100.0, 0, 1)
            # Directional influence is strongest for cells downwind from each active neighbor.
            directional = 0.0
            r0, r1 = max(0, r - 1), min(rows, r + 2)
            c0, c1 = max(0, c - 1), min(cols, c + 2)
            for nr0, nc0 in zip(*np.where(burned[r0:r1, c0:c1])):
                nr, nc = nr0 + r0, nc0 + c0
                direction = np.array([r - nr, c - nc], dtype=float)
                norm = np.linalg.norm(direction)
                if 0 < norm <= 1.5:
                    # downwind vector in row/column coordinates
                    directional = max(directional, max(0.0, np.dot(direction / norm, np.array([dr_wind, dc_wind]))))
            p = 0.04 + 0.42 * np.clip((veg[r, c] - 0.05) / 0.85, 0, 1)
            p *= (1 + 0.9 * np.clip(wind_m_s / 10.0, 0, 1) * directional)
            p *= (1 + 0.35 * local_slope)
            p *= (1 - 0.7 * humidity_damp)
            if rng.random() < np.clip(p, 0.01, 0.9): next_burned[r, c] = True
        burned = next_burned
        if hour in (1, 3, 6, 12) or hour == hours: snapshots[hour] = burned.copy()
    return snapshots


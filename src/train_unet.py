"""Train/evaluate a U-Net on pseudo-patches derived from Tirupati point rows.

This is a methodological demo, not imagery training: coordinate-sorted feature
rows are reshaped into patches and fire_occurrence supplies the mask. The pixels
do not represent a verified satellite raster or a calibrated spatial label map.
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.data import load_data, ARTIFACTS
from src.unet import UNet

PATCH = 16
CHANNELS = ["NDVI", "LST_C", "GPM_IMERG_precip_mm_day", "IMD_wind_speed_m_s",
            "IMD_relative_humidity_pct", "elevation_m", "slope_deg", "aspect_deg"]

def make_patches(df):
    # Sort by coordinates then form non-overlapping sequential patches so each
    # patch has a repeatable local neighborhood, rather than inventing image pixels.
    d = df.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    xrows, yrows = d[CHANNELS].to_numpy(float), d["fire_occurrence"].to_numpy(float)
    means = xrows.mean(0); stds = np.where(xrows.std(0) == 0, 1, xrows.std(0))
    xrows = np.nan_to_num((xrows - means) / stds)
    count = len(d) // (PATCH * PATCH)
    if count < 2: raise ValueError("Need at least 512 usable rows to form two 16x16 patches")
    X = xrows[:count*PATCH*PATCH].reshape(count, PATCH, PATCH, len(CHANNELS)).transpose(0, 3, 1, 2)
    Y = yrows[:count*PATCH*PATCH].reshape(count, 1, PATCH, PATCH)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def main():
    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    X, Y = make_patches(load_data())
    ids = np.arange(len(X)); train_ids, test_ids = train_test_split(ids, test_size=.25, random_state=42)
    train = DataLoader(TensorDataset(X[train_ids], Y[train_ids]), batch_size=4, shuffle=True)
    model = UNet(in_channels=len(CHANNELS)); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, 31):
        model.train(); losses=[]
        for xb, yb in train:
            optimizer.zero_grad(); loss=loss_fn(model(xb), yb); loss.backward(); optimizer.step(); losses.append(loss.item())
        print(f"epoch {epoch:02d}/30 loss={np.mean(losses):.4f}")
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X[test_ids])).numpy().ravel()
        example_prediction = torch.sigmoid(model(X[test_ids[:1]]))[0].cpu()
    truth = Y[test_ids].numpy().ravel().astype(int); pred = (probs >= .5).astype(int)
    metrics = {"accuracy": float(accuracy_score(truth, pred)), "f1": float(f1_score(truth, pred, zero_division=0)),
               "patch_count": int(len(X)), "train_patches": int(len(train_ids)), "test_patches": int(len(test_ids)),
               "note": "Metrics compare point labels binned into patches; not validation on satellite image masks."}
    torch.save({"state_dict": model.state_dict(), "channels": CHANNELS, "patch_size": PATCH,
                "example_input": X[test_ids[0]], "example_mask": Y[test_ids[0]],
                "example_prediction": example_prediction}, ARTIFACTS / "unet_best.pt")
    (ARTIFACTS / "unet_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)

if __name__ == "__main__": main()

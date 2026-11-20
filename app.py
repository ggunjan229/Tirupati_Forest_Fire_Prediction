import json
import math
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from src.ca_simulation import CellularAutomataFireSpread
from src.train_unet import UNetFireSegmenter, build_spatial_raster_grid

st.set_page_config(
    page_title="WildShield AI", page_icon="🌿", layout="wide"
)
st.markdown(
    """
<style>
.stApp {background: #f4f7f3; color:#15251d !important}

/* Apply dark text cleanly across standard body elements */
.stApp h2, .stApp h3, .stApp h4, .stApp label {
    color: #15251d !important;
}

/* Hero Banner - Forced pure white title and subtitle */
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #123b2b, #28734d);
    margin-bottom: 1rem;
}
.hero h1, .hero h1 * {
    color: #ffffff !important;
    margin: 0;
}
.hero p, .hero p * {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 500 !important;
    margin: .4rem 0 0 !important;
}

/* Keep Primary Submit Button ("Estimate fire occurrence") text crisp white */
.stApp button[kind="primary"],
.stApp button[kind="primary"] p,
.stApp button[kind="primary"] span,
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stFormSubmitButton"] button p {
    background-color: #1e5c3e !important;
    color: #ffffff !important;
    border: none !important;
}

/* Style secondary preset buttons cleanly */
.stApp button[kind="secondary"] {
    background-color: #ffffff !important;
    border: 1px solid #c3dac9 !important;
}
.stApp button[kind="secondary"] p {
    color: #173a2a !important;
}

.source-badge {display:inline-block;background:#e3efe6;color:#173a2a !important;padding:4px 10px;border-radius:12px;font-size:0.85rem;font-weight:600;margin-right:6px;margin-bottom:6px;border:1px solid #c3dac9}

/* Fix radio button options, metric numbers, metric labels, and captions */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp [data-testid="stRadio"] label p,
.stApp [data-testid="stRadio"] div[role="radiogroup"] p,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] div,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p,
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #15251d !important;
    opacity: 1 !important;
}

/* Style metric cards cleanly */
.stApp [data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d6e2d7;
    padding: 12px 16px;
    border-radius: 12px;
}

/* Fix tab colors */
.stApp button[data-baseweb="tab"] p,
.stApp button[data-baseweb="tab"] span {color:#385548 !important; opacity:1 !important}
.stApp button[data-baseweb="tab"][aria-selected="true"] p,
.stApp button[data-baseweb="tab"][aria-selected="true"] span {color:#b93824 !important; font-weight:700 !important; opacity:1 !important}
.result-card {padding:1rem 1.2rem;border:1px solid #d6e2d7;border-radius:12px;background:#fff}
</style>
<div class="hero">
  <h1>🌿 Tirupati WildShield AI </h1>
  <p>Multi-Source Google Earth Engine Dataset · Machine Learning Susceptibility · U-Net Deep Spatial Segmentation · Cellular Automata Spread Simulation</p>
</div>
""",
    unsafe_allow_html=True,
)

DATA_PATH = os.path.join("data", "tirupati_real_fire_dataset_2026.csv")
MODELS_DIR = os.path.join("outputs", "models")
PLOTS_DIR = os.path.join("outputs", "plots")
METRICS_CSV_PATH = os.path.join("outputs", "ml_benchmark_results.csv")

FEATURES = [
    "NDVI",
    "LST_C",
    "GPM_IMERG_precip_mm_day",
    "IMD_wind_speed_m_s",
    "IMD_relative_humidity_pct",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
]

MODEL_ARTIFACTS = {
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODELS_DIR, "xgboost.pkl"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree.pkl"),
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
}

FEATURE_LABELS = {
    "NDVI": "NDVI (Sentinel-2)",
    "LST_C": "Land surface temperature (°C)",
    "GPM_IMERG_precip_mm_day": "Precipitation (mm/day)",
    "IMD_wind_speed_m_s": "Wind speed (m/s)",
    "IMD_relative_humidity_pct": "Relative humidity (%)",
    "elevation_m": "Elevation (m)",
    "slope_deg": "Slope (degrees)",
    "aspect_deg": "Aspect (degrees)",
}

FEATURE_GUIDE = {
    "NDVI": "Sentinel-2 MSI Normalized Difference Vegetation Index measuring canopy greenness and fuel load.",
    "LST_C": "MODIS MOD11A2 Land Surface Temperature (°C) capturing surface thermal dryness.",
    "GPM_IMERG_precip_mm_day": "NASA GPM IMERG daily precipitation (mm/day) governing fuel moisture replenishment.",
    "IMD_wind_speed_m_s": "Surface wind velocity (m/s) driving oxygen supply and convective flame tilt.",
    "IMD_relative_humidity_pct": "Atmospheric relative humidity (%) regulating fine dead fuel moisture content.",
    "elevation_m": "SRTM 30m Digital Elevation Model height above sea level (metres).",
    "slope_deg": "SRTM-derived terrain steepness (degrees) accelerating uphill preheating.",
    "aspect_deg": "Hillside compass orientation (0–360°) governing diurnal solar insolation.",
}

@st.cache_data
def load_real_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    rename_map = {
        "precip_mm_day": "GPM_IMERG_precip_mm_day",
        "wind_speed_m_s": "IMD_wind_speed_m_s",
        "relative_humidity_pct": "IMD_relative_humidity_pct",
    }
    df_raw = df_raw.rename(columns=rename_map)
    df_raw = df_raw.interpolate(method="linear", limit_direction="both")

    df_fire = df_raw[df_raw["fire_occurrence"] == 1]
    df_nonfire = df_raw[df_raw["fire_occurrence"] == 0].sample(
        n=min(len(df_fire), len(df_raw[df_raw["fire_occurrence"] == 0])),
        random_state=42,
    )
    df_bal = (
        pd.concat([df_fire, df_nonfire], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return df_raw, df_bal

try:
    df_full, df = load_real_dataset()
except Exception as exc:
    st.error(f"Could not load the Tirupati CSV: {exc}")
    st.stop()

if not os.path.exists(MODEL_ARTIFACTS["Random Forest"]) or not os.path.exists(METRICS_CSV_PATH):
    try:
        with st.spinner("First launch: training Tirupati ML, U-Net & CA models on real GEE data…"):
            from main import main as run_full_pipeline
            run_full_pipeline()
        st.success("Training complete! Models and metrics saved in outputs/.")
    except Exception as exc:
        st.error(f"Could not train models automatically: {exc}")
        st.info("Run `python main.py` in your terminal first, then refresh this page.")
        st.stop()

metrics_df = pd.read_csv(METRICS_CSV_PATH) if os.path.exists(METRICS_CSV_PATH) else None

def predict_fire_proba(model_name, X_df):
    clf = joblib.load(MODEL_ARTIFACTS[model_name])
    X_arr = X_df[FEATURES].to_numpy(dtype=float)
    if model_name == "Logistic Regression":
        poly = joblib.load(os.path.join(MODELS_DIR, "poly_transform.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "poly_scaler.pkl"))
        X_arr = scaler.transform(poly.transform(X_arr))
    return clf.predict_proba(X_arr)[:, 1]

@st.cache_resource
def load_unet_and_grid():
    unet_weights = os.path.join(MODELS_DIR, "unet_fire_model.pth")
    if not os.path.exists(unet_weights):
        return None, None, None
    raster_x, mask_y, _, _ = build_spatial_raster_grid(df, grid_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNetFireSegmenter(in_channels=8, out_channels=1).to(device)
    unet.load_state_dict(torch.load(unet_weights, map_location=device))
    unet.eval()
    with torch.no_grad():
        inp = torch.tensor(raster_x[np.newaxis, :, :, :], dtype=torch.float32).to(device)
        prob_map = unet(inp).cpu().numpy()[0, 0]
    return prob_map, mask_y, raster_x

@st.cache_data(show_spinner="Measuring feature relevance on held-out data…")
def validation_feature_relevance(model_name):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    X, y = df[FEATURES].copy(), df["fire_occurrence"].to_numpy()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    base_preds = (predict_fire_proba(model_name, X_test) >= 0.5).astype(int)
    base_f1 = f1_score(y_test, base_preds)

    rng = np.random.default_rng(42)
    means, stds = [], []
    for feat in FEATURES:
        drops = []
        for _ in range(5):
            X_perm = X_test.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            perm_preds = (predict_fire_proba(model_name, X_perm) >= 0.5).astype(int)
            drops.append((base_f1 - f1_score(y_test, perm_preds)) * 100)
        means.append(np.mean(drops))
        stds.append(np.std(drops))

    relevance = pd.DataFrame({
        "Feature": [FEATURE_LABELS[name] for name in FEATURES],
        "Mean F1 decrease (pp)": means,
        "Variation (pp)": stds,
    })
    return relevance.sort_values("Mean F1 decrease (pp)", ascending=True).reset_index(drop=True)

def exact_reference_contributions(model_name, sample, baseline):
    feature_count = len(FEATURES)
    coalition_count = 1 << feature_count
    baseline_values = baseline[FEATURES].to_numpy(dtype=float)[0]
    sample_values = sample[FEATURES].to_numpy(dtype=float)[0]
    masks = np.arange(coalition_count, dtype=np.uint16)[:, None]
    bits = ((masks >> np.arange(feature_count, dtype=np.uint16)) & 1).astype(bool)
    coalition_rows = np.where(bits, sample_values[None, :], baseline_values[None, :])
    coalition_data = pd.DataFrame(coalition_rows, columns=FEATURES)
    coalition_scores = predict_fire_proba(model_name, coalition_data)

    denominator = math.factorial(feature_count)
    contributions = {}
    for feature_index, feature_name in enumerate(FEATURES):
        bit = 1 << feature_index
        contribution = 0.0
        for mask in range(coalition_count):
            if mask & bit:
                continue
            subset_size = mask.bit_count()
            weight = math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / denominator
            contribution += weight * (coalition_scores[mask | bit] - coalition_scores[mask])
        contributions[feature_name] = float(contribution)
    return float(coalition_scores[0]), contributions

tab_predict, tab_overview, tab_unet, tab_ca, tab_models, tab_features = st.tabs([
    "🎯 Forest Fire Prediction",
    "🗺️ Dataset & Satellite Sources",
    "🧠 U-Net Deep Segmentation",
    "🔥 CA Spread Simulation",
    "📊 Model Performance",
    "🔍 Feature Insights",
])

with tab_predict:
    st.subheader("Predict fire occurrence from environmental conditions")
    available_models = [name for name, path in MODEL_ARTIFACTS.items() if os.path.exists(path)]

    col_sel, col_preset = st.columns([1.2, 1.8])
    with col_sel:
        selected_model = st.selectbox("Select trained classifier", available_models, index=0)

    fire_median = df[df["fire_occurrence"] == 1][FEATURES].median()
    safe_median = df[df["fire_occurrence"] == 0][FEATURES].median()
    overall_median = df[FEATURES].median()

    with col_preset:
        st.markdown("**Quick-Load Real Tirupati Scenarios:**")
        p_c1, p_c2, p_c3 = st.columns(3)
        if p_c1.button("🔥 High-Risk Fire Profile"):
            st.session_state["preset_vals"] = fire_median.to_dict()
        if p_c2.button("🌲 Dataset Median"):
            st.session_state["preset_vals"] = overall_median.to_dict()
        if p_c3.button("🌧️ Low-Risk Safe Profile"):
            st.session_state["preset_vals"] = safe_median.to_dict()

    active_defaults = st.session_state.get("preset_vals", overall_median.to_dict())

    row_metrics = None
    if metrics_df is not None and selected_model in metrics_df["Algorithm"].values:
        row_metrics = metrics_df[metrics_df["Algorithm"] == selected_model].iloc[0]

    input_specs = [
        ("NDVI (Sentinel-2)", "NDVI", 0.0, 1.0, "%.3f"),
        ("Land surface temperature (°C)", "LST_C", 10.0, 60.0, "%.2f"),
        ("Precipitation (mm/day)", "GPM_IMERG_precip_mm_day", 0.0, 200.0, "%.2f"),
        ("Wind speed (m/s)", "IMD_wind_speed_m_s", 0.0, 60.0, "%.2f"),
        ("Relative humidity (%)", "IMD_relative_humidity_pct", 0.0, 100.0, "%.1f"),
        ("Elevation (m)", "elevation_m", -50.0, 4000.0, "%.1f"),
        ("Slope (degrees)", "slope_deg", 0.0, 60.0, "%.2f"),
        ("Aspect (degrees)", "aspect_deg", 0.0, 360.0, "%.2f"),
    ]

    with st.form("scenario_form"):
        c1, c2, c3 = st.columns(3)
        text_values = {}
        column_groups = [c1, c1, c1, c2, c2, c2, c3, c3]
        for (label, feature, lower, upper, number_format), column in zip(input_specs, column_groups):
            with column:
                text_values[feature] = st.text_input(
                    label,
                    value=number_format % float(active_defaults[feature]),
                    help=f"Valid range: {lower:g} to {upper:g}.",
                )
        submitted = st.form_submit_button("Estimate fire occurrence", type="primary")

    if submitted:
        errors = []
        values = {}
        for label, feature, lower, upper, _ in input_specs:
            raw_value = text_values[feature].strip()
            try:
                value = float(raw_value)
                if not np.isfinite(value) or not (lower <= value <= upper):
                    errors.append(f"{label}: enter a value between {lower:g} and {upper:g}.")
                else:
                    values[feature] = value
            except ValueError:
                errors.append(f"{label}: enter a valid number.")

        if errors:
            st.error("Please correct the following input(s):\n\n" + "\n".join(f"- {m}" for m in errors))
        else:
            sample = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
            probability = float(predict_fire_proba(selected_model, sample)[0])
            pred = int(probability >= 0.5)
            result_label = "Fire occurrence" if pred else "No fire occurrence"
            result_color = "#b93824" if pred else "#26734d"

            x, y = st.columns([1, 2])
            with x:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("**Prediction**")
                st.markdown(f'<div style="font-size:1.55rem;font-weight:700;color:{result_color}">{result_label}</div>', unsafe_allow_html=True)
                if row_metrics is not None:
                    st.caption(f"Model: {selected_model} · 5-fold F1 {row_metrics['F1-Score']:.3f} · accuracy {row_metrics['Accuracy (%)']:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with y:
                st.metric("Model fire score", f"{probability:.1%}")
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "valueformat": ".1f", "font": {"size": 30, "color": "#173a2a"}},
                    title={"text": "Estimated fire score (0–100%)", "font": {"size": 16, "color": "#263b30"}},
                    gauge={"axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#53665b"},
                           "bar": {"color": result_color}, "bgcolor": "#e5ece6",
                           "steps": [{"range": [0, 100], "color": "#edf2ee"}],
                           "threshold": {"line": {"color": "#263b30", "width": 3}, "thickness": .8, "value": 50}},
                ))
                gauge.update_layout(height=220, margin={"l": 25, "r": 25, "t": 45, "b": 10, "pad": 0})
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

            st.subheader("How the input values shifted this score (Exact Shapley)")
            baseline = pd.DataFrame([overall_median[FEATURES].to_dict()], columns=FEATURES)
            baseline_score, local_contributions = exact_reference_contributions(selected_model, sample, baseline)
            base_col, current_col = st.columns(2)
            base_col.metric("Median-input baseline score", f"{baseline_score:.1%}")
            current_col.metric("Score for entered values", f"{probability:.1%}")

            impact_rows = sorted(local_contributions.items(), key=lambda item: abs(item[1]))
            impact_labels = [FEATURE_LABELS[name] for name, _ in impact_rows]
            impact_values = [value * 100 for _, value in impact_rows]
            impact_colors = ["#26734d" if value >= 0 else "#3478a8" for value in impact_values]

            impact_fig = go.Figure(go.Bar(
                x=impact_values, y=impact_labels, orientation="h",
                marker_color=impact_colors, text=[f"{value:+.1f} pp" for value in impact_values],
                textposition="outside", hovertemplate="%{y}<br>Score shift: %{x:+.2f} percentage points<extra></extra>",
            ))
            impact_fig.add_vline(x=0, line_color="#53665b", line_width=1)
            impact_fig.update_layout(
                title="Per-input Shapley contribution relative to median baseline",
                xaxis_title="Change in fire score (percentage points)",
                yaxis_title="Input feature", height=420,
                margin={"l": 20, "r": 50, "t": 55, "b": 40}, showlegend=False,
            )
            st.plotly_chart(impact_fig, use_container_width=True, config={"displayModeBar": False})

with tab_overview:
    st.markdown(
        """
        <div>
          <span class="source-badge">🛰️ MODIS MCD64A1 (BurnDate DOY)</span>
          <span class="source-badge">🌿 Sentinel-2 MSI (NDVI)</span>
          <span class="source-badge">🌡️ MODIS MOD11A2 (LST)</span>
          <span class="source-badge">🌧️ NASA GPM IMERG (Precipitation)</span>
          <span class="source-badge">💨 ERA5 / IMD (Wind & Humidity)</span>
          <span class="source-badge">⛰️ SRTM 30m DEM (Slope & Aspect)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Balanced Training Samples", f"{len(df):,}")
    b.metric("Fire Occurrences (MODIS)", f"{int(df.fire_occurrence.sum()):,}")
    c.metric("Non-Fire Samples", f"{int((1 - df.fire_occurrence).sum()):,}")
    d.metric("Raw GEE Pool", f"{len(df_full):,} pts")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Tirupati & Seshachalam spatial sample distribution")
        plot = go.Figure(go.Scattermap(
            lat=df.latitude, lon=df.longitude, mode="markers",
            marker={"size": 7, "color": df.fire_occurrence.map({0: "#4f8c6b", 1: "#e56e33"}), "opacity": 0.72},
            text=["Fire occurrence" if x else "No fire occurrence" for x in df.fire_occurrence],
            hovertemplate="%{text}<br>Lat: %{lat:.4f}, Lon: %{lon:.4f}<extra></extra>"
        ))
        plot.update_layout(
            map={"style": "open-street-map", "center": {"lat": float(df.latitude.mean()), "lon": float(df.longitude.mean())}, "zoom": 8},
            height=430, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False
        )
        st.plotly_chart(plot, use_container_width=True)

    with right:
        st.subheader("MODIS BurnDate Day-of-Year (DOY) Distribution")
        if "MODIS_BurnDate_DOY" in df.columns:
            burned_doy = df[df["MODIS_BurnDate_DOY"] > 0]["MODIS_BurnDate_DOY"]
            doy_fig = go.Figure(go.Histogram(x=burned_doy, nbinsx=25, marker_color="#e56e33"))
            doy_fig.update_layout(
                height=210, margin={"l": 10, "r": 10, "t": 30, "b": 20},
                title="Historical Fire Detection Julian Day (DOY)",
                xaxis_title="Day of Year (1–365)", yaxis_title="Fire Pixels"
            )
            st.plotly_chart(doy_fig, use_container_width=True, config={"displayModeBar": False})

        corr_path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
        if os.path.exists(corr_path):
            st.image(corr_path, caption="Environmental Factor Correlation Heatmap", use_container_width=True)

with tab_unet:
    st.subheader("U-Net Deep Learning Spatial Segmentation (Tirupati Landscape)")
    st.markdown("Unlike point-based tabular models, the **2D Fully Convolutional U-Net** processes multi-channel spatial rasters (`NDVI`, `LST`, `Precipitation`, `Wind`, `Humidity`, `Elevation`, `Slope`, `Aspect`) to segment contiguous high-risk fire zones across the Tirupati terrain.")

    prob_map, mask_y, raster_x = load_unet_and_grid()
    if prob_map is not None:
        thresh = st.slider(
            "High-Risk Segmentation Probability Threshold", 0.10, 0.90, 0.50, 0.05,
            help="Adjust threshold to classify continuous U-Net probabilities into high-risk fire zones."
        )
        binary_zones = (prob_map >= thresh).astype(int)
        high_risk_pct = binary_zones.mean() * 100
        high_risk_ha = binary_zones.sum() * 0.09

        u1, u2, u3 = st.columns(3)
        u1.metric("U-Net Architecture", "4-Level Encoder-Decoder")
        u2.metric("Segmented High-Risk Area", f"{high_risk_ha:.1f} ha ({high_risk_pct:.1f}%)")
        u3.metric("Input Tensor Dimensions", "128 × 128 × 8 Channels")

        fig_unet = make_subplots(
            rows=1, cols=2,
            subplot_titles=("U-Net Continuous Fire Probability Mask", f"Segmented High-Risk Zones (Threshold ≥ {thresh:.2f})")
        )
        fig_unet.add_trace(go.Heatmap(z=prob_map, zmin=0, zmax=1, colorscale="RdYlGn_r", colorbar={"title": "Prob", "x": 0.45}), row=1, col=1)
        fig_unet.add_trace(go.Heatmap(z=binary_zones, zmin=0, zmax=1, colorscale=[[0, "#dfebdc"], [1, "#b93824"]], showscale=False), row=1, col=2)
        fig_unet.update_layout(height=420, margin={"l": 10, "r": 10, "t": 45, "b": 10})
        st.plotly_chart(fig_unet, use_container_width=True)

        c_u1, c_u2 = st.columns(2)
        unet_cm_path = os.path.join(PLOTS_DIR, "unet_confusion_matrix.png")
        unet_loss_path = os.path.join(PLOTS_DIR, "unet_training_loss.png")
        if os.path.exists(unet_cm_path):
            c_u1.image(unet_cm_path, caption="U-Net Spatial Classification Confusion Matrix", use_container_width=True)
        if os.path.exists(unet_loss_path):
            c_u2.image(unet_loss_path, caption="U-Net Hybrid Dice + BCE Loss Convergence", use_container_width=True)
    else:
        st.info("Run `python main.py` first to train the U-Net segmentation model.")

with tab_ca:
    st.subheader("Cellular Automata (CA) Dynamic Fire Spread Simulation")
    st.caption("Simulates outward directional fire spread from U-Net high-risk zones over 1, 3, 6, and 12 hours governed by NDVI fuel density, terrain slope tilt, and wind azimuth.")
    left, right = st.columns([1, 2.2])
    with left:
        view_mode = st.radio("Display Mode", ["Interactive Single Interval", "All 4 Horizons Side-by-Side (1h, 3h, 6h, 12h)"])
        hours = st.select_slider("Forecast interval", options=[1, 3, 6, 12], value=6, format_func=lambda h: f"{h} hour{'s' if h != 1 else ''}")
        wind_direction = st.slider("Wind direction azimuth (degrees)", 0, 359, 45, step=5)

    with right:
        prob_map, _, raster_x = load_unet_and_grid()
        if prob_map is not None:
            ca_sim = CellularAutomataFireSpread(
                susceptibility_map=prob_map,
                ndvi_grid=raster_x[0],
                slope_grid=raster_x[6],
                wind_speed_grid=raster_x[3],
                wind_azimuth_deg=float(wind_direction),
            )
            ca_sim.seed_high_risk_ignitions()
            snapshots = {}
            horizon_stats = []
            for hr in range(1, 13):
                b_cells = ca_sim.step()
                if hr in [1, 3, 6, 12]:
                    snapshots[hr] = ca_sim.state.copy()
                    horizon_stats.append({"Interval": f"{hr} hr", "Cells": int(b_cells), "Area (ha)": round(b_cells * 0.09, 2)})

            if view_mode == "Interactive Single Interval":
                fig = go.Figure(go.Heatmap(
                    z=snapshots[hours],
                    colorscale=[[0.0, "#dfebdc"], [0.5, "#ed5b2a"], [1.0, "#9f210e"]],
                    zmin=0, zmax=2,
                    colorbar={"title": "State", "tickvals": [0, 1, 2], "ticktext": ["Unburned", "Active Fire", "Burned"]},
                ))
                fig.update_layout(height=420, margin={"l": 0, "r": 0, "t": 35, "b": 0}, title=f"CA Fire Spread Front after {hours} Hour(s) (Wind Azimuth: {wind_direction}°)")
                st.plotly_chart(fig, use_container_width=True)
                sel_stat = [s for s in horizon_stats if s["Interval"] == f"{hours} hr"][0]
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Affected Grid Cells", f"{sel_stat['Cells']:,}")
                c_m2.metric("Estimated Burnt Area", f"{sel_stat['Area (ha)']:.2f} ha")
            else:
                fig_4 = make_subplots(rows=1, cols=4, subplot_titles=[f"{h} Hour(s)" for h in [1, 3, 6, 12]])
                for idx, h in enumerate([1, 3, 6, 12], start=1):
                    fig_4.add_trace(go.Heatmap(
                        z=snapshots[h],
                        colorscale=[[0.0, "#dfebdc"], [0.5, "#ed5b2a"], [1.0, "#9f210e"]],
                        zmin=0, zmax=2, showscale=False,
                    ), row=1, col=idx)
                fig_4.update_layout(height=340, margin={"l": 10, "r": 10, "t": 40, "b": 10}, title=f"Multi-Horizon Fire Propagation (Wind Direction: {wind_direction}°)")
                st.plotly_chart(fig_4, use_container_width=True)
                cols_s = st.columns(4)
                for col_widget, st_item in zip(cols_s, horizon_stats):
                    col_widget.metric(st_item["Interval"], f"{st_item['Area (ha)']} ha", f"{st_item['Cells']} cells")
        else:
            st.info("Run `python main.py` first to generate U-Net weights for CA seeding.")

with tab_models:
    st.subheader("Comparative Performance of Forest Fire Prediction Models")
    if metrics_df is not None:
        tbl_fig = go.Figure(go.Table(
            header={"values": list(metrics_df.columns), "fill_color": "#173a2a", "font": {"color": "white", "size": 13}, "align": "left"},
            cells={"values": [metrics_df[col].tolist() for col in metrics_df.columns], "fill_color": "#ffffff", "font": {"color": "#15251d", "size": 12}, "align": "left", "height": 32},
        ))
        tbl_fig.update_layout(height=210, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(tbl_fig, use_container_width=True)

    c_img1, c_img2 = st.columns(2)
    cm_ml_path = os.path.join(PLOTS_DIR, "ml_confusion_matrices.png")
    roc_ml_path = os.path.join(PLOTS_DIR, "ml_roc_curves.png")
    if os.path.exists(cm_ml_path):
        c_img1.image(cm_ml_path, caption="Tabular ML Confusion Matrices (5-Fold CV)", use_container_width=True)
    if os.path.exists(roc_ml_path):
        c_img2.image(roc_ml_path, caption="ROC-AUC Curves Comparison", use_container_width=True)

with tab_features:
    st.subheader("Feature meanings and model relevance")
    feature_model = st.selectbox("Select model for permutation importance", available_models, index=0)
    left, right = st.columns([1.1, 1.4])
    with left:
        st.markdown("#### What each satellite/weather input represents")
        for feature in FEATURES:
            st.markdown(f"**{FEATURE_LABELS[feature]}**  \n{FEATURE_GUIDE[feature]}")
    with right:
        st.markdown("#### Held-out permutation relevance")
        relevance = validation_feature_relevance(feature_model)
        bar_colors = ["#26734d" if value >= 0 else "#a05a35" for value in relevance["Mean F1 decrease (pp)"]]
        relevance_fig = go.Figure(go.Bar(
            x=relevance["Mean F1 decrease (pp)"], y=relevance["Feature"], orientation="h",
            marker_color=bar_colors, error_x={"type": "data", "array": relevance["Variation (pp)"], "visible": True},
            text=[f"{value:+.2f} pp" for value in relevance["Mean F1 decrease (pp)"]], textposition="outside",
        ))
        relevance_fig.add_vline(x=0, line_color="#53665b", line_width=1)
        relevance_fig.update_layout(
            title=f"Held-out F1 decrease after shuffling ({feature_model})",
            xaxis_title="Mean decrease in F1 (percentage points)",
            yaxis_title="Feature", height=480,
            margin={"l": 20, "r": 65, "t": 55, "b": 45}, showlegend=False,
        )
        st.plotly_chart(relevance_fig, use_container_width=True, config={"displayModeBar": False})
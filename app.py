import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data import load_data, FEATURES, ARTIFACTS
from src.ca import simulate_spread

st.set_page_config(page_title="Tirupati Fire Prediction", page_icon="🌿", layout="wide")
st.markdown("""
<style>
.stApp {background: #f4f7f3; color:#15251d} h1,h2,h3 {color:#173a2a}
.hero {padding:1.5rem 1.7rem;border-radius:18px;background:linear-gradient(120deg,#123b2b,#28734d);color:#fff;margin-bottom:1rem}
.hero h1 {color:#fff;margin:0}.hero p {color:#dcebe2;margin:.4rem 0 0}
.note {background:#fff4dd;padding:1rem;border-left:4px solid #df9a2c;border-radius:8px;color:#493514}
/* Keep widget labels legible with both light and dark Streamlit themes. */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label {color:#263b30 !important; opacity:1 !important}
.stApp button[data-baseweb="tab"] {color:#385548 !important; opacity:1 !important}
.stApp button[data-baseweb="tab"] p,
.stApp button[data-baseweb="tab"] span {color:#385548 !important; opacity:1 !important}
.stApp button[data-baseweb="tab"][aria-selected="true"],
.stApp button[data-baseweb="tab"][aria-selected="true"] p,
.stApp button[data-baseweb="tab"][aria-selected="true"] span {color:#b93824 !important; opacity:1 !important}
.result-card {padding:1rem 1.2rem;border:1px solid #d6e2d7;border-radius:12px;background:#fff}
</style>
<div class="hero"><h1>🌿 Tirupati Forest Fire Prediction</h1><p>Fire occurrence classification · environmental risk exploration · conceptual spread simulation</p></div>
""", unsafe_allow_html=True)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the Tirupati CSV: {exc}"); st.stop()

metric_path = ARTIFACTS / "metrics.json"
random_forest_path = ARTIFACTS / "random_forest.joblib"
if not random_forest_path.exists() or not metric_path.exists():
    try:
        with st.spinner("First launch: training the Tirupati classifiers and preparing the dashboard…"):
            from src.train_models import main as train_classifiers
            train_classifiers()
        st.success("Initial training is complete. The trained models are saved in artifacts/.")
    except Exception as exc:
        st.error(f"Could not prepare the prediction models: {exc}")
        st.info("If this is a pandas/NumPy import error, recreate the project virtual environment and install the pinned requirements in README.md.")
        st.stop()
metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else None

tab_predict, tab_overview, tab_ca, tab_models = st.tabs(["Forest Fire Prediction", "Overview", "Spread simulation", "Model results"])

with tab_overview:
    a,b,c,d = st.columns(4)
    a.metric("Records", f"{len(df):,}"); b.metric("Fire labels", f"{int(df.fire_occurrence.sum()):,}")
    c.metric("Non-fire labels", f"{int((1-df.fire_occurrence).sum()):,}")
    d.metric("Coverage", f"{df.date_yyyymmdd.min()} – {df.date_yyyymmdd.max()}")
    left,right = st.columns([1.4,1])
    with left:
        st.subheader("Observed point locations")
        plot = go.Figure(go.Scattermap(lat=df.latitude, lon=df.longitude, mode="markers",
            marker={"size":7,"color":df.fire_occurrence.map({0:"#4f8c6b",1:"#e56e33"}),"opacity":.65},
            text=["Fire occurrence" if x else "No fire occurrence" for x in df.fire_occurrence], hovertemplate="%{text}<br>%{lat:.4f}, %{lon:.4f}<extra></extra>"))
        plot.update_layout(map={"style":"open-street-map", "center":{"lat":float(df.latitude.mean()),"lon":float(df.longitude.mean())},
                                "zoom":8}, height=430, margin={"l":0,"r":0,"t":0,"b":0}, showlegend=False)
        st.plotly_chart(plot, width="stretch")
    with right:
        st.subheader("Fire labels by month")
        dates = pd.to_datetime(df.date_yyyymmdd.astype(str), format="%Y%m%d")
        monthly = df.assign(month=dates.dt.strftime("%b"), monthnum=dates.dt.month).groupby(["monthnum","month","fire_occurrence"]).size().reset_index(name="rows")
        monthly_plot = monthly.pivot_table(index=["monthnum","month"], columns="fire_occurrence", values="rows", fill_value=0).reset_index().sort_values("monthnum")
        monthly_fig = go.Figure()
        for class_id, label, color in [(0, "No fire", "#7ea98c"), (1, "Fire", "#df7540")]:
            values = monthly_plot[class_id] if class_id in monthly_plot.columns else [0] * len(monthly_plot)
            monthly_fig.add_bar(x=monthly_plot["month"], y=values, name=label, marker_color=color)
        monthly_fig.update_layout(barmode="group", height=360, margin={"l":0,"r":0,"t":10,"b":0},
                                  xaxis_title="Month", yaxis_title="Records", legend_title="Label")
        st.plotly_chart(monthly_fig, width="stretch")
        st.caption("Synthetic March–August 2026 records. Coordinates are displayed as provided in the CSV.")

with tab_predict:
    st.subheader("Predict fire occurrence from environmental conditions")
    model_options = ["Random Forest", "XGBoost"]
    available = []
    for name in model_options:
        f = ARTIFACTS / f"{name.lower().replace(' ','_')}.joblib"
        if f.exists(): available.append(name)
    if not available:
        st.warning("No trained prediction model is available. Restart the app to trigger first-run training.")
    model_artifacts = {
        "Logistic Regression": ARTIFACTS / "logistic_regression.joblib",
        "Decision Tree": ARTIFACTS / "decision_tree.joblib",
        "Random Forest": ARTIFACTS / "random_forest.joblib",
        "XGBoost": ARTIFACTS / "xgboost.joblib",
    }
    available = [name for name, path in model_artifacts.items() if path.exists()]
    scored_models = [
        name for name in available
        if metrics and name in metrics.get("models", {})
        and metrics["models"][name].get("f1") is not None
        and metrics["models"][name].get("accuracy") is not None
    ]
    if not scored_models:
        st.warning("No trained tabular model with saved evaluation scores is available. Check the files in artifacts/ or restart the app to train the models.")
    else:
        selected_model = st.selectbox("Prediction model", available)
        model = joblib.load(ARTIFACTS / f"{selected_model.lower().replace(' ','_')}.joblib")
        # Select by held-out F1 first; use accuracy as a deterministic tie-breaker.
        selected_model = max(
            scored_models,
            key=lambda name: (metrics["models"][name]["f1"], metrics["models"][name]["accuracy"]),
        )
        selected_scores = metrics["models"][selected_model]
        model = joblib.load(model_artifacts[selected_model])
        
        defaults = df[FEATURES].median(numeric_only=True)
        input_specs = [
            ("Latitude", "latitude", float(df.latitude.min()), float(df.latitude.max()), "%.5f"),
            ("Longitude", "longitude", float(df.longitude.min()), float(df.longitude.max()), "%.5f"),
            ("NDVI", "NDVI", 0.0, 1.0, "%.3f"),
            ("Land surface temperature (°C)", "LST_C", 10.0, 60.0, "%.2f"),
            ("Precipitation (mm/day)", "GPM_IMERG_precip_mm_day", 0.0, 200.0, "%.2f"),
            ("Wind speed (m/s)", "IMD_wind_speed_m_s", 0.0, 60.0, "%.2f"),
            ("Relative humidity (%)", "IMD_relative_humidity_pct", 0.0, 100.0, "%.1f"),
            ("Elevation (m)", "elevation_m", 0.0, 4000.0, "%.1f"),
            ("Slope (degrees)", "slope_deg", 0.0, 60.0, "%.2f"),
            ("Aspect (degrees)", "aspect_deg", 0.0, 359.0, "%.2f"),
        ]
        with st.form("scenario_form"):
            c1,c2,c3 = st.columns(3)
            text_values = {}
            column_groups = [c1, c1, c1, c1, c2, c2, c2, c2, c3, c3]
            for (label, feature, lower, upper, number_format), column in zip(input_specs, column_groups):
                with column:
                    text_values[feature] = st.text_input(
                        label, value=number_format % float(defaults[feature]),
                        key=f"prediction_input_{feature}",
                        help=f"Type a value from {lower:g} to {upper:g}.",
                    )
            with c3:
                month = st.selectbox("Month", list(range(3,9)), index=2)
            submitted = st.form_submit_button("Estimate fire occurrence", type="primary")
        if submitted:
            errors = []
            values = {}
            for label, feature, lower, upper, _ in input_specs:
                raw_value = text_values[feature].strip()
                try:
                    value = float(raw_value)
                    if not np.isfinite(value):
                        raise ValueError
                    if not lower <= value <= upper:
                        errors.append(f"{label}: enter a value from {lower:g} to {upper:g}.")
                    else:
                        values[feature] = value
                except ValueError:
                    errors.append(f"{label}: enter a valid number.")
            if errors:
                st.error("Please correct the following input(s):\n\n" + "\n".join(f"- {message}" for message in errors))
            else:
                values["month"] = month
                sample = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
                pred = int(model.predict(sample)[0])
                probability = float(model.predict_proba(sample)[0, 1]) if hasattr(model, "predict_proba") else None
                result_label = "Fire occurrence" if pred else "No fire occurrence"
                result_color = "#b93824" if pred else "#26734d"
                x, y = st.columns([1, 2])
                with x:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("**Prediction**")
                    st.markdown(f'<div style="font-size:1.55rem;font-weight:700;color:{result_color}">{result_label}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                with y:
                    if probability is not None:
                        st.metric("Model fire score", f"{probability:.1%}")
                        gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=probability * 100,
                            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 30, "color": "#173a2a"}},
                            title={"text": "Estimated fire score (0–100%)", "font": {"size": 16, "color": "#263b30"}},
                            gauge={"axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#53665b"},
                                   "bar": {"color": result_color}, "bgcolor": "#e5ece6",
                                   "steps": [{"range": [0, 100], "color": "#edf2ee"}],
                                   "threshold": {"line": {"color": "#263b30", "width": 3}, "thickness": .8,
                                                 "value": 50}},
                        ))
                        gauge.update_layout(height=220, margin={"l": 25, "r": 25, "t": 45, "b": 10, "pad": 0})
                        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
                        st.caption("The score shows how strongly this model favors the fire class for these inputs. It is not a calibrated real-world probability or a safety threshold.")
                    else:
                        st.info("The selected model does not provide a fire score.")

with tab_ca:
    st.subheader("Cellular Automata (CA) spread explorer")
    st.caption("The grid interpolates the CSV's point covariates to a regular illustration grid. Wind direction is supplied here because the CSV contains wind speed but no direction.")
    left,right = st.columns([1,2])
    with left:
        hours = st.select_slider("Forecast interval", options=[1,3,6,12], value=3, format_func=lambda h:f"{h} hour{'s' if h != 1 else ''}")
        wind_direction = st.slider("Wind direction (direction fire is pushed toward)",0,359,45,step=5)
        ca_wind = st.slider("Wind speed (m/s)",0.0,20.0, float(df.IMD_wind_speed_m_s.median()), .2)
        ca_humidity = st.slider("Relative humidity (%) ",0,100,int(df.IMD_relative_humidity_pct.median()))
        ignition_count = st.slider("Ignition points",1,5,1)
        st.caption("Each simulation hour is one discrete CA step, not an hour-calibrated physical spread rate.")
    with right:
        @st.cache_data
        def grid_fields(frame, size=40):
            # Assign each observation to a regular lat/lon cell; average duplicate cells.
            lat_edges=np.linspace(frame.latitude.min(),frame.latitude.max(),size+1)
            lon_edges=np.linspace(frame.longitude.min(),frame.longitude.max(),size+1)
            work=frame.copy(); work["ri"]=np.clip(np.digitize(work.latitude,lat_edges)-1,0,size-1); work["ci"]=np.clip(np.digitize(work.longitude,lon_edges)-1,0,size-1)
            fields=[]
            for col in ["NDVI","slope_deg"]:
                g=work.groupby(["ri","ci"])[col].mean(); arr=np.full((size,size),np.nan)
                for (r,c),v in g.items(): arr[r,c]=v
                # Fill empty cells with nearest observed grid cell.
                known=np.argwhere(np.isfinite(arr))
                for r,c in np.argwhere(~np.isfinite(arr)):
                    if len(known):
                        nr,nc=known[np.argmin(((known-[r,c])**2).sum(axis=1))]; arr[r,c]=arr[nr,nc]
                    else:
                        arr[r,c]=float(work[col].median())
                fields.append(arr)
            return fields
        veg,slope_grid=grid_fields(df)
        snaps=simulate_spread(veg,slope_grid,ca_wind,wind_direction,ca_humidity,hours=hours,ignition_count=ignition_count,seed=71)
        burned=snaps[hours]
        fig=go.Figure(go.Heatmap(z=burned.astype(int),colorscale=[[0,"#dfebdc"],[.5,"#dfebdc"],[.5001,"#ed5b2a"],[1,"#9f210e"]],zmin=0,zmax=1,colorbar={"title":"Burned"},hovertemplate="Grid row %{y}, col %{x}<br>Burned: %{z}<extra></extra>"))
        fig.update_layout(height=440,margin={"l":0,"r":0,"t":20,"b":0},xaxis_title="Local grid column",yaxis_title="Local grid row",title=f"Illustrative spread after {hours} discrete steps")
        st.plotly_chart(fig,width="stretch")
        st.metric("Cells burned in illustration",f"{int(burned.sum())} / {burned.size}")
        st.caption("Spread probability increases with NDVI, wind alignment and slope, and decreases with humidity. Randomness is fixed for repeatable display.")

with tab_models:
    st.subheader("Held-out model performance")
    if not metrics:
        st.info("No evaluation results found. Run `python -m src.train_models`.")
    else:
        rows=[]
        for name,values in metrics["models"].items():
            rows.append({"Model":name,"Accuracy":values["accuracy"],"F1-score":values["f1"],"Precision":values["precision"],"Recall":values["recall"],"ROC-AUC":values["roc_auc"]})
        model_table = go.Figure(go.Table(
            header={"values":["Model", "Accuracy", "F1-score", "Precision", "Recall", "ROC-AUC"], "fill_color":"#173a2a", "font":{"color":"white"}, "align":"left"},
            cells={"values":[[r["Model"] for r in rows]] + [[f"{r[key]:.3f}" if r[key] is not None else "n/a" for r in rows]
                              for key in ["Accuracy", "F1-score", "Precision", "Recall", "ROC-AUC"]],
                   "fill_color":"#ffffff", "align":"left", "height":32}))
        model_table.update_layout(height=240, margin={"l":0,"r":0,"t":10,"b":0})
        st.plotly_chart(model_table, width="stretch")
        st.caption(f"Split: {metrics['dataset']['split']}; {metrics['dataset']['rows']} rows; stratified, random_state={metrics['dataset']['random_state']}.")
        st.subheader("U-Net patch experiment")
        unet_metrics = ARTIFACTS / "unet_metrics.json"
        if unet_metrics.exists():
            um=json.loads(unet_metrics.read_text(encoding="utf-8")); u1,u2,u3=st.columns(3)
            u1.metric("Patch accuracy",f"{um['accuracy']:.3f}"); u2.metric("Patch F1-score",f"{um['f1']:.3f}"); u3.metric("Test patches",str(um["test_patches"]))
            st.caption(um["note"])
            weights_path = ARTIFACTS / "unet_best.pt"
            if weights_path.exists():
                try:
                    import torch
                    from src.unet import UNet
                    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
                    unet = UNet(in_channels=len(checkpoint["channels"]))
                    unet.load_state_dict(checkpoint["state_dict"]); unet.eval()
                    with torch.no_grad():
                        probability_mask = torch.sigmoid(unet(checkpoint["example_input"].unsqueeze(0)))[0, 0].numpy()
                    observed_mask = checkpoint["example_mask"][0].numpy()
                    p1,p2 = st.columns(2)
                    p1.plotly_chart(go.Figure(go.Heatmap(z=probability_mask, zmin=0, zmax=1, colorscale="YlOrRd",
                        colorbar={"title":"Fire score"}, hovertemplate="Row %{y}, col %{x}<br>Score %{z:.2f}<extra></extra>"))
                        .update_layout(title="U-Net predicted fire-score mask", height=340, margin={"l":0,"r":0,"t":45,"b":0}), width="stretch")
                    p2.plotly_chart(go.Figure(go.Heatmap(z=observed_mask, zmin=0, zmax=1, colorscale=[[0,"#dfebdc"],[1,"#e56e33"]],
                        colorbar={"title":"Input label"}, hovertemplate="Row %{y}, col %{x}<br>Label %{z:.0f}<extra></extra>"))
                        .update_layout(title="Derived point-label mask for example", height=340, margin={"l":0,"r":0,"t":45,"b":0}), width="stretch")
                    st.caption("Example U-Net output from the pseudo-patch holdout. This is a derived tabular demonstration, not a satellite-image prediction.")
                except Exception as exc:
                    st.warning(f"Could not load the U-Net example prediction: {exc}")
        else: st.info("Optional: install PyTorch and run `python -m src.train_unet` to train the U-Net patch demo.")
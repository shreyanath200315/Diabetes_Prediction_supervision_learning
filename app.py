"""
Diabetes Prediction Web App (Streamlit)
Uses the files saved by the corrected notebook (folder: diabetes_deployment/).

Run:  streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

DEPLOY_DIR = "diabetes_deployment"
DATA_FILE = "diabetes.csv"
RESULTS_FILE = os.path.join("saved_models", "all_tuning_results.csv")
INVALID_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        return {
            "model": joblib.load(f"{DEPLOY_DIR}/diabetes_model.pkl"),
            "scaler": joblib.load(f"{DEPLOY_DIR}/scaler.pkl"),
            "imputer": joblib.load(f"{DEPLOY_DIR}/imputer.pkl"),
            "features": joblib.load(f"{DEPLOY_DIR}/selected_features.pkl"),
            "base": joblib.load(f"{DEPLOY_DIR}/base_features.pkl"),
        }
    except Exception as e:  # missing files / version mismatch
        return {"error": str(e)}


@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else None


def predict(art, raw: dict):
    """Same preprocessing as the notebook: zeros->NaN, impute, ratio feature, scale."""
    row = pd.DataFrame([raw])[art["base"]].astype(float)
    row[INVALID_ZERO_COLS] = row[INVALID_ZERO_COLS].replace(0, np.nan)
    row = pd.DataFrame(art["imputer"].transform(row), columns=art["base"])
    row["Glucose_BMI_Ratio"] = row["Glucose"] / row["BMI"]
    scaled = pd.DataFrame(art["scaler"].transform(row[art["features"]]), columns=art["features"])
    return float(art["model"].predict_proba(scaled)[0, 1])


art = load_artifacts()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes Predictor")
page = st.sidebar.radio("Navigate", ["Prediction", "Visualizations", "About"])
st.sidebar.divider()
threshold = st.sidebar.slider(
    "Decision threshold", 0.1, 0.9, 0.5, 0.05,
    help="Probability above which a person is flagged as diabetic. Lower it to catch more cases (higher recall, more false alarms).",
)
st.sidebar.caption("⚠️ Educational tool – not a medical diagnosis.")

if "error" in art:
    st.error(
        "Could not load the model files from `diabetes_deployment/`. "
        "Run the notebook's deployment cell first and keep that folder next to `app.py`.\n\n"
        f"Details: `{art['error']}`"
    )
    st.stop()


# ----------------------------------------------------------------------------
# PAGE 1: Prediction
# ----------------------------------------------------------------------------
if page == "Prediction":
    st.title("Diabetes Risk Prediction")
    st.write("Enter the patient's details. For Glucose, Blood Pressure, Skin Thickness, Insulin and BMI, "
             "enter **0 if the value is unknown** – it will be treated as missing and estimated.")

    with st.form("patient_form"):
        c1, c2, c3, c4 = st.columns(4)
        pregnancies = c1.number_input("Pregnancies", 0, 20, 2)
        glucose = c2.number_input("Glucose (mg/dL)", 0, 300, 120)
        bp = c3.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin = c4.number_input("Skin Thickness (mm)", 0, 100, 25)
        c5, c6, c7, c8 = st.columns(4)
        insulin = c5.number_input("Insulin (µU/mL)", 0, 900, 0)
        bmi = c6.number_input("BMI", 0.0, 70.0, 28.0, step=0.1)
        dpf = c7.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.45, step=0.01)
        age = c8.number_input("Age", 1, 120, 35)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        prob = predict(art, raw)
        is_diabetic = prob >= threshold

        left, right = st.columns([1, 1])
        with left:
            if is_diabetic:
                st.error(f"### 🔴 Likely DIABETIC\nEstimated probability: **{prob:.1%}**")
            else:
                st.success(f"### 🟢 Likely NOT diabetic\nEstimated probability of diabetes: **{prob:.1%}**")
            band = "Low" if prob < 0.33 else "Moderate" if prob < 0.66 else "High"
            st.metric("Risk level", band)
            st.caption(f"Decision threshold: {threshold:.2f}")

        with right:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                title={"text": "Diabetes probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#c0392b" if is_diabetic else "#27ae60"},
                    "steps": [
                        {"range": [0, 33], "color": "#d5f5e3"},
                        {"range": [33, 66], "color": "#fdebd0"},
                        {"range": [66, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                },
            ))
            fig.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        # Compare patient with dataset averages
        df = load_data()
        if df is not None:
            st.subheader("How this patient compares")
            cols = ["Glucose", "BMI", "Age", "BloodPressure", "Pregnancies"]
            clean = df.copy()
            clean[INVALID_ZERO_COLS] = clean[INVALID_ZERO_COLS].replace(0, np.nan)
            comp = pd.DataFrame({
                "Feature": cols,
                "Patient": [raw[c] if (c not in INVALID_ZERO_COLS or raw[c] > 0) else np.nan for c in cols],
                "Non-diabetic average": clean[clean.Outcome == 0][cols].mean().values,
                "Diabetic average": clean[clean.Outcome == 1][cols].mean().values,
            }).melt("Feature", var_name="Group", value_name="Value")
            fig2 = px.bar(comp, x="Feature", y="Value", color="Group", barmode="group",
                          color_discrete_map={"Patient": "#2c3e50", "Non-diabetic average": "#27ae60",
                                              "Diabetic average": "#c0392b"})
            st.plotly_chart(fig2, use_container_width=True)

        st.info("This prediction is a statistical estimate from a small dataset (768 records, "
                "mostly women of Pima Indian heritage aged 21+). It is **not** a diagnosis – "
                "please consult a doctor for testing (e.g. HbA1c, fasting glucose).")


# ----------------------------------------------------------------------------
# PAGE 2: Visualizations
# ----------------------------------------------------------------------------
elif page == "Visualizations":
    st.title("Data & Model Visualizations")
    uploaded = None
    df = load_data()
    if df is None:
        uploaded = st.file_uploader("Upload diabetes.csv", type="csv")
        df = load_data(uploaded)
    if df is None:
        st.warning("Place `diabetes.csv` next to `app.py` or upload it above.")
        st.stop()

    zeros = sum((df[c] == 0).sum() for c in INVALID_ZERO_COLS)
    if zeros == 0:
        st.warning("This CSV has no zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI, so it looks "
                   "already imputed (possibly using the target). Use the original Pima dataset for reliable results.")
    plot_df = df.copy()
    plot_df[INVALID_ZERO_COLS] = plot_df[INVALID_ZERO_COLS].replace(0, np.nan)
    plot_df["Diagnosis"] = plot_df["Outcome"].map({0: "Non-diabetic", 1: "Diabetic"})
    features = [c for c in df.columns if c != "Outcome"]
    cmap = {"Non-diabetic": "#27ae60", "Diabetic": "#c0392b"}

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Class balance", "Distributions", "Relationships", "Correlation", "Model"])

    with tab1:
        counts = plot_df["Diagnosis"].value_counts().reset_index()
        counts.columns = ["Diagnosis", "Patients"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(counts, names="Diagnosis", values="Patients", hole=0.4,
                               color="Diagnosis", color_discrete_map=cmap), use_container_width=True)
        c2.plotly_chart(px.bar(counts, x="Diagnosis", y="Patients", color="Diagnosis", text_auto=True,
                               color_discrete_map=cmap), use_container_width=True)

    with tab2:
        feat = st.selectbox("Feature", features, index=1)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(plot_df, x=feat, color="Diagnosis", barmode="overlay", opacity=0.65,
                                     marginal="rug", color_discrete_map=cmap), use_container_width=True)
        c2.plotly_chart(px.box(plot_df, x="Diagnosis", y=feat, color="Diagnosis",
                               color_discrete_map=cmap), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", features, index=1)
        y = c2.selectbox("Y axis", features, index=5)
        st.plotly_chart(px.scatter(plot_df, x=x, y=y, color="Diagnosis", opacity=0.7,
                                   color_discrete_map=cmap, hover_data=features), use_container_width=True)

    with tab4:
        corr = plot_df[features + ["Outcome"]].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True)

    with tab5:
        model = art["model"]
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"Feature": art["features"],
                                "Importance": model.feature_importances_}).sort_values("Importance")
            st.subheader("Feature importance (deployed model)")
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", text_auto=".3f"),
                            use_container_width=True)
        res = load_results()
        if res is not None:
            st.subheader("Tuned model comparison (test set)")
            metrics = [c for c in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
                       if c in res.columns]
            long = res.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group")
                            .update_layout(yaxis_range=[0, 1]), use_container_width=True)
            st.dataframe(res.drop(columns=["Best Parameters"], errors="ignore").round(4),
                         use_container_width=True)
        else:
            st.caption("Copy `saved_models/all_tuning_results.csv` next to the app to see the model comparison.")


# ----------------------------------------------------------------------------
# PAGE 3: About
# ----------------------------------------------------------------------------
else:
    st.title("About this project")
    st.markdown(f"""
**Goal:** predict whether a person is likely to have diabetes from 8 routine medical measurements.

**Deployed model:** `{type(art['model']).__name__}` – selected by 5-fold cross-validated F1 among nine tuned
algorithms (Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting,
AdaBoost, XGBoost).

### Pipeline
1. Zeros in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI are physiologically impossible → treated as missing.
2. Missing values filled with the **training-set median** (imputer saved with the model).
3. Engineered feature **Glucose / BMI ratio**.
4. Standard scaling, then prediction.
5. Probability ≥ decision threshold (default 0.5) → "diabetic".
""")
    st.subheader("Input features")
    st.table(pd.DataFrame({
        "Feature": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                    "DiabetesPedigreeFunction", "Age"],
        "Meaning": ["Number of pregnancies", "Plasma glucose (2h oral glucose tolerance test)",
                    "Diastolic blood pressure (mm Hg)", "Triceps skin-fold thickness (mm)",
                    "2-hour serum insulin (µU/mL)", "Body mass index (kg/m²)",
                    "Family-history score of diabetes", "Age in years"],
    }))
    st.subheader("Limitations")
    st.markdown("""
- Trained on the **Pima Indians Diabetes dataset** (768 records, females aged 21+); it may not generalise to other groups.
- Small dataset → results are uncertain; accuracy of ~75–80% is typical on the original data.
- Not a medical device and **not a diagnosis**. Always confirm with laboratory tests and a doctor.
""")
    st.caption(f"scikit-learn model files loaded from `{DEPLOY_DIR}/`.")

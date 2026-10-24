import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoMPG | Fuel Intelligence", page_icon="🚘", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f4f8fc 0%,#e9f2f7 55%,#f8fbfd 100%);}
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {padding: 26px 30px; border-radius: 22px; color: white;
background: linear-gradient(115deg,#102a43,#176b87,#27a6a1); margin-bottom: 20px;}
.hero h1 {font-size: 2.35rem; margin-bottom: .25rem;}
.hero p {font-size: 1rem; opacity: .93; margin-bottom: 0;}
div[data-testid="stMetric"] {background: white; border: 1px solid #dce8ef;
padding: 15px; border-radius: 15px; box-shadow: 0 4px 14px #16324f0c;}
section[data-testid="stSidebar"] {background: #eaf3f7;}
.stButton>button {border-radius: 10px; font-weight: 700;}
</style>
<div class="hero">
<h1>🚘 AutoMPG Fuel Intelligence</h1>
<p>Machine-learning estimates of vehicle fuel efficiency — explore specifications, compare models, and predict MPG.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Educational prediction tool • Dataset: auto-mpg.csv • Target: miles per gallon (MPG)")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        candidates = ["auto-mpg.csv", "auto-mpg (1).csv", "374b3e61-1b79-4bba-8b4a-6c80fa08797b-auto-mpg (1).csv"]
        file_path = next((f for f in candidates if os.path.exists(f)), None)
        if file_path is None:
            return None
        data = pd.read_csv(file_path)
    data.columns = [str(c).strip().lower() for c in data.columns]
    return data

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload auto-mpg.csv", type=["csv"])
    st.caption("If no file is uploaded, the app looks for auto-mpg.csv in the project folder.")
    st.divider()
    st.header("🧭 Navigation")
    page = st.radio("Choose a view", ["Predict MPG", "Model comparison", "Explore dataset"])

data = load_data(uploaded)
if data is None:
    st.warning("Please upload your auto-mpg.csv file using the sidebar. Keep the CSV in the same folder as app.py for automatic loading.")
    st.stop()

required = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
missing_cols = [c for c in required if c not in data.columns]
if missing_cols:
    st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
    st.stop()

# Convert invalid markers such as '?' into NaN, then impute them inside the pipeline.
for col in ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]:
    data[col] = pd.to_numeric(data[col].replace("?", np.nan), errors="coerce")
data = data.dropna(subset=["mpg"]).copy()
data["origin"] = data["origin"].map({1: "USA", 2: "Europe", 3: "Japan"}).fillna("Unknown")

feature_cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
numeric_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
categorical_features = ["origin"]
X = data[feature_cols]
y = data["mpg"]

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("numeric", numeric_pipe, numeric_features),
                                   ("categorical", categorical_pipe, categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_defs = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=2)
}
trained, scores = {}, []
for name, estimator in model_defs.items():
    pipe = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    trained[name] = pipe
    scores.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                   "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R²": r2_score(y_test, pred)})
score_df = pd.DataFrame(scores).sort_values("RMSE").reset_index(drop=True)
best_name = score_df.iloc[0]["Model"]
best_model = trained[best_name]

if page == "Predict MPG":
    st.subheader("Predict a vehicle's fuel efficiency")
    st.write("Adjust the specifications below and select **Predict MPG**. The app trains and compares three regression algorithms on the dataset.")
    a, b, c = st.columns(3)
    with a:
        cylinders = st.selectbox("Engine cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.slider("Displacement (cu. in.)", 60.0, 460.0, 150.0, 5.0)
        horsepower = st.slider("Horsepower", 40.0, 240.0, 95.0, 5.0)
    with b:
        weight = st.slider("Vehicle weight (lbs)", 1500.0, 5200.0, 2800.0, 50.0)
        acceleration = st.slider("Acceleration (0–60 mph time scale)", 8.0, 25.0, 15.0, 0.5)
        model_year = st.slider("Model year (last two digits)", 70, 82, 78)
    with c:
        origin = st.selectbox("Country of origin", ["USA", "Europe", "Japan"])
        st.markdown("#### Current model")
        st.metric("Selected algorithm", best_name)
        st.caption("Chosen automatically by the lowest test-set RMSE.")
    if st.button("⚡ Predict fuel efficiency", type="primary", use_container_width=True):
        sample = pd.DataFrame([{"cylinders": cylinders, "displacement": displacement,
            "horsepower": horsepower, "weight": weight, "acceleration": acceleration,
            "model year": model_year, "origin": origin}])
        estimate = float(best_model.predict(sample)[0])
        st.success("Prediction complete")
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Estimated fuel efficiency", f"{estimate:.1f} MPG")
        with m2:
            st.markdown("**How to read this result**")
            st.write("Higher MPG means the vehicle is estimated to travel farther per gallon. This is a model estimate, not a guarantee of real-world mileage.")
        st.progress(min(max(estimate / 50, 0.0), 1.0), text="Estimated MPG on a 0–50 MPG display scale")

    st.divider()
    st.subheader("Model quality at a glance")
    best_row = score_df.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Training records", f"{len(X_train):,}")
    k2.metric("Test records", f"{len(X_test):,}")
    k3.metric("Best test R²", f"{best_row['R²']:.3f}")

elif page == "Model comparison":
    st.subheader("Compare the three regression algorithms")
    st.write("All models use the same train/test split and preprocessing pipeline. Lower MAE/RMSE is better; higher R² is better.")
    best_row = score_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best_name)
    c2.metric("Lowest RMSE", f"{best_row['RMSE']:.2f} MPG")
    c3.metric("Best-model R²", f"{best_row['R²']:.3f}")
    st.dataframe(score_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"}), use_container_width=True, hide_index=True)
    chart_df = score_df.set_index("Model")[["MAE", "RMSE"]]
    st.bar_chart(chart_df)
    st.info(f"Deployment choice: **{best_name}**, selected by the lowest RMSE on the held-out test set.")
    st.caption("For a formal report, repeat the evaluation with cross-validation and document the random seed and split.")

else:
    st.subheader("Explore the dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", f"{data.shape[1]:,}")
    c3.metric("Average MPG", f"{data['mpg'].mean():.1f}")
    c4.metric("Missing values before imputation", f"{int(data[required].isna().sum().sum()):,}")
    st.markdown("#### MPG distribution")
    hist = data["mpg"].dropna()
    fig, ax = plt.subplots()
    ax.hist(hist, bins=20, edgecolor="white")
    ax.set_xlabel("Miles per gallon (MPG)")
    ax.set_ylabel("Number of vehicles")
    ax.set_title("Distribution of fuel efficiency")
    st.pyplot(fig)
    st.markdown("#### Preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(data.select_dtypes(include=np.number).describe().round(2), use_container_width=True)

st.divider()
st.caption("Note: 'car name' is excluded from the prediction features because it is free text with many unique values. 'origin' is treated as a categorical feature.")

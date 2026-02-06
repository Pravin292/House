import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS (FRONTEND ONLY) ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.block-container {
    padding-top: 2rem;
}

.glass {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}

.title-text {
    font-size: 46px;
    font-weight: 800;
    color: white;
    text-align: center;
}

.subtitle-text {
    color: #d1d5db;
    text-align: center;
    font-size: 18px;
}

.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 14px;
    height: 3em;
    font-weight: 700;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.metric-box {
    background: rgba(255,255,255,0.2);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="glass">
    <div class="title-text">🏠 AI House Price Prediction</div>
    <div class="subtitle-text">
        Smart real-estate valuation powered by XGBoost Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Model Controls")
    st.info("Tune XGBoost parameters to observe model performance changes.")

    n_estimators = st.slider("Boosting Rounds", 50, 500, 200)
    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05)
    max_depth = st.slider("Max Depth", 2, 10, 5)
    subsample = st.slider("Subsample Ratio", 0.5, 1.0, 0.7)
    colsample = st.slider("Colsample By Tree", 0.5, 1.0, 0.7)

# ---------------- DATA LOADING ----------------
try:
    df = pd.read_csv("house_prices.csv")
except Exception as e:
    st.error(f"❌ Could not load 'house_prices.csv' — {e}")
    st.stop()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Data Insights", "🚀 Training & Metrics", "🔮 Prediction"])

# ================= TAB 1 =================
with tab1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Features", df.shape[1])

    st.subheader("💹 Sale Price Distribution")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df["SalePrice"], kde=True, ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DATA PREP ----------------
if "SalePrice" not in df.columns:
    st.error("Dataset must contain 'SalePrice' column.")
    st.stop()

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ================= TAB 2 =================
with tab2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🚀 Model Training & Performance")

    if st.button("🔥 Train XGBoost Model"):
        with st.spinner("Training model with log-scaled target..."):
            y_log = np.log1p(y_train)

            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_log)

            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)

            st.session_state["model"] = model
            st.session_state["X_cols"] = X_encoded.columns
            st.session_state["trained"] = True

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MAE", f"₹ {mae:,.2f}")
            m3.metric("RMSE", f"₹ {rmse:,.2f}")

            st.subheader("📈 Top Feature Importance")
            importance = pd.DataFrame({
                "Feature": X_encoded.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            fig2, ax2 = plt.subplots(figsize=(10,5))
            sns.barplot(
                data=importance,
                x="Importance",
                y="Feature",
                palette="Blues_r",
                ax=ax2
            )
            st.pyplot(fig2)
    else:
        st.info("Click the button to train the model.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:
    if not st.session_state.get("trained"):
        st.warning("Train the model first to enable predictions.")
    else:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("🔮 Predict House Price")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        input_data = {}

        c1, c2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with c1 if i % 2 == 0 else c2:
                input_data[col] = st.number_input(
                    col, value=float(X[col].median()), step=1.0
                )

        for col in st.session_state["X_cols"]:
            if col not in input_data:
                input_data[col] = 0

        if st.button("💎 Predict Value"):
            input_df = pd.DataFrame([input_data])[st.session_state["X_cols"]]
            pred_log = st.session_state["model"].predict(input_df)[0]
            prediction = np.expm1(pred_log)

            st.balloons()
            st.markdown(f"""
            <div class="glass" style="text-align:center;">
                <h2 style="color:white;">Estimated Market Value</h2>
                <h1 style="font-size:52px; color:#00ffcc;">₹ {prediction:,.2f}</h1>
                <p style="color:#d1d5db;">AI-driven real-estate valuation</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

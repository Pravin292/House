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
    page_title="🏠 House Price Prediction Premier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Model Controls")
    st.info("Adjust the parameters for the XGBoost Regressor to see how it affects performance.")
    
    n_estimators = st.slider("Boosting Rounds", 50, 500, 200)
    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05)
    max_depth = st.slider("Max Depth", 2, 10, 5)
    subsample = st.slider("Subsample Ratio", 0.5, 1.0, 0.7)
    colsample = st.slider("Colsample By Tree", 0.5, 1.0, 0.7)

st.title("🏠 House Price Prediction")
st.markdown("### Powered by XGBoost Intelligence")
st.write("Extracting insights from the Ames Housing Dataset to provide high-accuracy valuations.")

# ---------------- DATA LOADING ----------------
# Use local house_prices.csv as the permanent data source
try:
    df = pd.read_csv('house_prices.csv')
except Exception as e:
    st.error(f"❌ Error: Could not find 'house_prices.csv' at the expected location. Details: {e}")
    st.stop()

if df is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Data Insights", "🚀 Training & Metrics", "🔮 Prediction"])
    
    with tab1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), width='stretch')
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total Rows", df.shape[0])
        col_m2.metric("Total Features", df.shape[1])
        
        st.subheader("💹 Price Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
        sns.histplot(df['SalePrice'], kde=True, color='#1e3a8a', ax=ax_dist)
        ax_dist.set_title("Target Variable: SalePrice Distribution")
        st.pyplot(fig_dist)

    # Prepare Data
    if "SalePrice" in df.columns:
        X = df.drop("SalePrice", axis=1)
        y = df["SalePrice"]
        X_encoded = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        with tab2:
            st.subheader("Model Performance")
            if st.button("🔥 Initialize AI Training"):
                with st.spinner('Optimizing XGBoost parameters with log-transformation...'):
                    # Target scaling for better convergence on price data
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
                    
                    # Predict and inverse log transform
                    y_pred_log = model.predict(X_test)
                    y_pred = np.expm1(y_pred_log)
                    
                    # Store model in session state
                    st.session_state['model'] = model
                    st.session_state['X_cols'] = X_encoded.columns
                    st.session_state['trained'] = True

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("R² Score (Accuracy)", f"{r2:.4f}")
                    m_col2.metric("Mean Absolute Error", f"₹ {mae:,.2f}")
                    m_col3.metric("Root Mean Squared Error", f"₹ {rmse:,.2f}")

                    st.subheader("📈 Significant Features")
                    importance = pd.DataFrame({
                        "Feature": X_encoded.columns,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False).head(10)

                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Importance", y="Feature", data=importance, hue="Feature", palette="Blues_r", legend=False, ax=ax_imp)
                    ax_imp.set_title("Market Value Drivers (Top 10)")
                    st.pyplot(fig_imp)
            else:
                st.info("Click the button above to train and see results.")

        with tab3:
            if st.session_state.get('trained'):
                st.subheader("Adjust Features for Valuation")
                
                # Show main feature inputs (top numeric ones)
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                
                input_data = {}
                col_i1, col_i2 = st.columns(2)
                
                for i, col in enumerate(numeric_cols):
                    with col_i1 if i % 2 == 0 else col_i2:
                        # Use sliders or number inputs for better control
                        val = st.number_input(f"{col}", value=float(X[col].median()), step=1.0)
                        input_data[col] = val
                
                # Ensure categorical features are accounted for
                for col in st.session_state['X_cols']:
                    if col not in input_data:
                        input_data[col] = 0 # Default for non-present dummies

                if st.button("💎 Calculate Estimated Value"):
                    input_df = pd.DataFrame([input_data])[st.session_state['X_cols']]
                    # Predict and inverse log
                    prediction_log = st.session_state['model'].predict(input_df)[0]
                    prediction = np.expm1(prediction_log)
                    
                    st.balloons()
                    st.markdown(f"""
                        <div style="background-color: #1e3a8a; padding: 30px; border-radius: 20px; text-align: center;">
                            <h2 style="color: white; margin: 0;">Estimated Market Value</h2>
                            <h1 style="color: #4CAF50; font-size: 50px; margin: 10px 0;">₹ {prediction:,.2f}</h1>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please train the model in the 'Training & Metrics' tab first.")
    else:
        st.error("The dataset must contain a 'SalePrice' column.")
else:
    st.info("Waiting for dataset... Please upload or ensure house_prices.csv is in the folder.")

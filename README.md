# 🏠 House Price Prediction Premier

A professional-grade real estate valuation platform powered by **XGBoost Regression** and **Streamlit**. This project provides high-accuracy house price estimations based on the Ames Housing dataset, utilizing advanced machine learning techniques like log-transformation and hyperparameter optimization.

---

## 🚀 Experience the App

The application provides a seamless, premium interface for data exploration and predictive modeling:

*   **📊 Data Insights**: Comprehensive visualization of house price distributions and feature correlations.
*   **⚙️ Live Training**: Adjust XGBoost hyper-parameters (learning rate, depth, estimators) in real-time and see how they impact R² scores and error metrics.
*   **🔮 Luxury Valuation**: An interactive prediction engine that calculates market values with professional-grade precision.

---

## 🛠️ Key Technologies

- **Core Engine**: `XGBoost Regressor`
- **UI Framework**: `Streamlit`
- **Data Engineering**: `Pandas`, `NumPy`, `Scikit-Learn`
- **Dashboards**: `Matplotlib`, `Seaborn`
- **Optimization**: Log-Target Scaling for handling skewed price distributions.

---

## 📂 Project Structure

- `house_app.py`: The main premium Streamlit application.
- `XGBoost Assignment.ipynb`: Detailed Jupyter Notebook implementing the 5 key tasks of regression analysis.
- `house_prices.csv`: The Ames Housing dataset used for training and inference.
- `README.md`: Project documentation.

---

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prithivirj4706/home.git
   cd home
   ```

2. **Install dependencies**:
   ```bash
   pip install xgboost streamlit pandas numpy scikit-learn matplotlib seaborn
   ```
   *Note: For macOS users, ensure `libomp` is installed via `brew install libomp`.*

3. **Run the Application**:
   ```bash
   streamlit run house_app.py
   ```

---

## 📈 Model Performance

The model utilizes **Log Transformation** on the target `SalePrice` to account for market skewness, resulting in higher precision across all budget tiers. Key metrics tracked:
- **R² Score**: Measures overall variance explained.
- **MAE**: Mean Absolute Error (Average deviation in ₹).
- **RMSE**: Precision for higher-value properties.

---

## 👤 Author
**Prithiviraj**
[GitHub Profile](https://github.com/prithivirj4706)

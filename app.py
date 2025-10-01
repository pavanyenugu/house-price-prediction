import streamlit as st, joblib, pandas as pd, os

MODEL_PATH = "models/house_price_model.pkl"
st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Prediction")

if not os.path.exists(MODEL_PATH):
    st.warning("Run training first with: python src/train_model.py")
    st.stop()

pipeline = joblib.load(MODEL_PATH)
st.sidebar.header("Input features")
inputs = {
    "LotArea": st.sidebar.number_input("LotArea", 100, 100000, 8450),
    "OverallQual": st.sidebar.slider("OverallQual", 1, 10, 7),
    "YearBuilt": st.sidebar.number_input("YearBuilt", 1800, 2025, 2003),
    "TotalBsmtSF": st.sidebar.number_input("TotalBsmtSF", 0, 10000, 856),
    "GrLivArea": st.sidebar.number_input("GrLivArea", 100, 10000, 1710),
    "FullBath": st.sidebar.slider("FullBath", 0, 5, 2),
    "BedroomAbvGr": st.sidebar.slider("BedroomAbvGr", 0, 10, 3),
    "TotRmsAbvGrd": st.sidebar.slider("TotRmsAbvGrd", 1, 20, 8),
    "GarageCars": st.sidebar.slider("GarageCars", 0, 10, 2)
}
df = pd.DataFrame([inputs])
if st.button("Predict"):
    pred = pipeline.predict(df)[0]
    st.success(f"Estimated Price: ${pred:,.2f}")

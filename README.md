# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
[![Made with ML](https://img.shields.io/badge/Made%20with-ML-yellow)](#)

An **end-to-end Machine Learning project** that predicts house prices.  
It includes preprocessing, training, evaluation, and a simple **Streamlit web app** for predictions.

---

## 📂 Repo structure
```
house-price-prediction/
├── app/                    # Streamlit app
│   └── app.py
├── data/                   # dataset folder (sample data provided)
│   └── housing_sample.csv
├── models/                 # trained models (auto-created)
├── notebooks/              # notebooks for EDA / experiments
├── src/                    # source scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt        # dependencies
├── README.md               # this file
├── LICENSE                 # MIT license
└── .gitignore
```

---

## 🚀 Quick Start
1. Clone repo & install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # (Mac/Linux)
# .\venv\Scripts\activate   # (Windows)
pip install -r requirements.txt
```

2. Train the model (uses sample data by default):
```bash
python src/train_model.py
```

3. Run the app:
```bash
streamlit run app/app.py
```

---

## 🧠 Model
- **RandomForestRegressor** (default)
- Modular design — can easily swap in **XGBoost** or others

---

## 📊 Example
Sample input (from sidebar in app):
- LotArea = 8450  
- OverallQual = 7  
- YearBuilt = 2003  
- TotalBsmtSF = 856  
- GrLivArea = 1710  

👉 Output: Predicted Price ≈ **$208,000**

---

## 🔮 Future Improvements
- Add hyperparameter tuning (GridSearch / Optuna)
- Add feature importance visualization
- Add CI/CD and Dockerfile

---

## 📜 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

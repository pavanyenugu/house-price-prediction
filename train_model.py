import os, argparse, joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from src.data_preprocessing import load_data, build_preprocessor, split_features_target

def main(data_path="data/housing_sample.csv", model_dir="models", random_state=42):
    os.makedirs(model_dir, exist_ok=True)
    df = load_data(data_path)
    X, y = split_features_target(df, "SalePrice")
    numeric_cols = [c for c in X.select_dtypes(include=["int64","float64"]).columns if c.lower() != "id"]
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    print("R2:", r2_score(y_test, preds))
    joblib.dump(pipeline, os.path.join(model_dir, "house_price_model.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/housing_sample.csv")
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()
    main(args.data, args.out)

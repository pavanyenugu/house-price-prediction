import joblib, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model_path, test_csv, target_col="SalePrice"):
    pipeline = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    X, y = df.drop(columns=[target_col]), df[target_col].values
    preds = pipeline.predict(X)
    print("RMSE:", mean_squared_error(y, preds, squared=False))
    print("R2:", r2_score(y, preds))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/house_price_model.pkl")
    parser.add_argument("--test", type=str, default="data/housing_sample.csv")
    args = parser.parse_args()
    evaluate(args.model, args.test)

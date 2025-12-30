import joblib, gzip
import pandas as pd
from sklearn.model_selection import train_test_split

from app.data_analysis import load_data
from app.preprocess import preprocess_features, build_preprocess_pipeline
from app.models import (
    get_logistic,
    get_decision_tree,
    get_random_forest,
    get_lightgbm
)
from app.evaluate import evaluate_model


def train_and_save_model(
    data_path="data/credit_risk_dataset.csv",
    model_path="models/lgbm_pipeline.pkl.gz"
):
    # 1. Load data
    df = load_data(data_path)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    # 2. Preprocess
    X = preprocess_features(X)
    preprocess = build_preprocess_pipeline()

    # 3. Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 4. Init models
    models = {
        "Logistic Regression": get_logistic(preprocess),
        "Decision Tree": get_decision_tree(preprocess),
        "Random Forest": get_random_forest(preprocess),
        "LightGBM": get_lightgbm(preprocess, scale_pos_weight)
    }

    results = []

    # 5. Train & evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\nValidation results:")
    print(results_df)

    # 6. Save best model (LightGBM)
    with gzip.open(model_path, "wb") as f:
        joblib.dump(models["LightGBM"], f)

    print(f"\nModel saved to {model_path}")

    return models["LightGBM"], X_test, y_test, results_df

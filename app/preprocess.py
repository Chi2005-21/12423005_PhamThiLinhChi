from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_features(X):
    X = X.copy()

    # Fill missing values
    for col in ["person_emp_length", "loan_int_rate"]:
        X[col] = X[col].fillna(X[col].median())

    # Binary mapping
    X["cb_person_default_on_file"] = X["cb_person_default_on_file"].map({"Y": 1, "N": 0})

    # Ordinal mapping for loan grade
    grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    X["loan_grade"] = X["loan_grade"].map(grade_mapping)

    return X


def build_preprocess_pipeline():
    numeric_features = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    categorical_features = [
        "person_home_ownership",
        "loan_intent"
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("remain", "passthrough", ["loan_grade", "cb_person_default_on_file"])
        ]
    )

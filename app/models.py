from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def get_logistic(preprocess):
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])


def get_decision_tree(preprocess):
    return Pipeline([
        ("preprocess", preprocess),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])


def get_random_forest(preprocess):
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ])


def get_lightgbm(preprocess, scale_pos_weight):
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ))
    ])

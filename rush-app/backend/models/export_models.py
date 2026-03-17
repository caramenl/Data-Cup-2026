import os
import joblib

from rush_gradient_boosting import load_and_prepare_data, train_gbm
from rush_logistic_regression import load_data, train_and_evaluate_model

MODELS_DIR = "models"
DATA_PATH = "engineered_rushes.csv"


def export_gbm():
    print("\n=== Exporting GBM xG model ===")
    X, y, feature_cols = load_and_prepare_data(DATA_PATH)
    model, X_train, X_test, y_train, y_test = train_gbm(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, "rush_xg_gbm.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "rush_features_list.joblib"))

    print("Saved:")
    print(f"- {os.path.join(MODELS_DIR, 'rush_xg_gbm.joblib')}")
    print(f"- {os.path.join(MODELS_DIR, 'rush_features_list.joblib')}")


def export_logit():
    print("\n=== Exporting logistic models ===")
    X, y_shot, y_entry = load_data(DATA_PATH)

    shot_result = train_and_evaluate_model(
        X, y_shot, "Target: P(Shot_10s = 1 | X)"
    )
    entry_result = train_and_evaluate_model(
        X, y_entry, "Target: P(ControlledEntry = 1 | X)"
    )

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(shot_result, os.path.join(MODELS_DIR, "shot_logit.pkl"))
    joblib.dump(entry_result, os.path.join(MODELS_DIR, "entry_logit.pkl"))

    print("Saved:")
    print(f"- {os.path.join(MODELS_DIR, 'shot_logit.pkl')}")
    print(f"- {os.path.join(MODELS_DIR, 'entry_logit.pkl')}")


def main():
    export_gbm()
    export_logit()
    print("\nAll model artifacts exported successfully.")


if __name__ == "__main__":
    main()
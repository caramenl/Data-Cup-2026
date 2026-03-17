import joblib
import pandas as pd
import statsmodels.api as sm

from .config import FEATURES, MODELS_DIR

xg_model = None
shot_model = None
entry_model = None


def load_models():
    global xg_model, shot_model, entry_model

    print("MODELS_DIR =", MODELS_DIR)

    xg_model = joblib.load(MODELS_DIR / "rush_xg_gbm.joblib")
    shot_model = joblib.load(MODELS_DIR / "shot_logit.pkl")
    entry_model = joblib.load(MODELS_DIR / "entry_logit.pkl")

    print("Models loaded successfully")


def build_feature_df(payload):
    return pd.DataFrame([[payload[f] for f in FEATURES]], columns=FEATURES)


def predict_all(payload):
    df = build_feature_df(payload)

    xg_pred = float(xg_model.predict(df)[0])

    df_const = sm.add_constant(df, has_constant="add")
    shot_prob = float(shot_model.predict(df_const)[0])
    entry_prob = float(entry_model.predict(df_const)[0])

    return {
        "shot_probability": shot_prob,
        "controlled_entry_probability": entry_prob,
        "xg_15s": xg_pred
    }
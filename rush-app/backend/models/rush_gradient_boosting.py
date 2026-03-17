import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

"""
Rush xG Regression Pipeline
---------------------------
Goal: Predict offensive value (xG_15s) based on team structure during the first 2s of transition.

xG_15s Calculation: 
Sum of proxy xG values for all shots/goals by the attacking team within 15s of puck recovery.
Proxy xG is a heuristic model based on shot type, distance to net, and shot angle.
"""

def load_and_prepare_data(filepath):
    """Loads engineered rushes and verify xG_15s exists."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Target
    target_col = 'xG_15s'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    # Standard Features (Structural + Kinematic)
    feature_cols = [
        'nearest_support_dist', 'second_support_dist', 'teammates_in_radius',
        'lane_spread', 'max_lane_width', 'lane_balance',
        'depth_range', 'depth_variance', 'is_flat_line',
        'mean_team_speed', 'speed_variance', 'carrier_speed', 'carrier_acceleration',
        'nearest_defender_dist', 'defender_closing_speed', 'defenders_between'
    ]
    
    # Validation
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing features in dataset: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Drop NaNs
    initial_count = len(df)
    df = df.dropna(subset=feature_cols + [target_col])
    print(f"Loaded {len(df)} rows after dropping {initial_count - len(df)} rows with missing values.")
    
    return df[feature_cols], df[target_col], feature_cols

def train_gbm(X, y):
    """Trains GradientBoostingRegressor on xG target."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameters from requested plan
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Prints regression evaluation metrics and summary statistics."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nGBM Performance Metrics:")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R^2:  {r2:.6f}")
    
    print("\nTarget Summary (xG_15s):")
    print(y_test.describe())
    
    print("\nPrediction Summary:")
    print(pd.Series(y_pred).describe())

def save_feature_importance(model, feature_names, output_path):
    """Saves importance plot and prints ranking."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance: Rush Structural Impact on xG")
    plt.tight_layout()
    plt.savefig(output_path)
    
    print("\nRanked Feature Importance:")
    ranking = sorted(zip(importances, feature_names), reverse=True)
    for i, (imp, name) in enumerate(ranking):
        print(f"{i+1}. {name}: {imp:.4f}")

def simulate_tactics(base_rush, adjustments, model, feature_cols):
    """
    Simulates tactical changes on a base transition state.
    base_rush: pandas Series or 1-row DataFrame
    adjustments: dict of {feature_name: delta_value}
    """
    # Ensure DataFrame for consistent feature names
    if isinstance(base_rush, pd.Series):
        sim_rush = base_rush.to_frame().T.copy()
    else:
        sim_rush = base_rush.copy()
        
    for feat, delta in adjustments.items():
        if feat in feature_cols:
            sim_rush[feat] += delta
            
    base_pred = model.predict(base_rush.to_frame().T if isinstance(base_rush, pd.Series) else base_rush)[0]
    sim_pred = model.predict(sim_rush)[0]
    
    abs_gain = sim_pred - base_pred
    pct_gain = (abs_gain / base_pred * 100) if base_pred > 0 else 0
    
    return base_pred, sim_pred, abs_gain, pct_gain

def main():
    data_path = "c:\\Data Cup\\processed data\\engineered_rushes.csv"
    artifact_dir = "c:\\Data Cup\\models"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Prepare Data
    X, y, feature_cols = load_and_prepare_data(data_path)
    
    # 2. Train
    model, X_train, X_test, y_train, y_test = train_gbm(X, y)
    print(f"\nModel trained on {len(X_train)} samples.")
    
    # 3. Evaluate
    evaluate_model(model, X_test, y_test)
    
    # 4. Feature Importance
    importance_path = os.path.join(artifact_dir, "xg_feature_importance.png")
    save_feature_importance(model, feature_cols, importance_path)
    
    # 5. Save Model Assets
    joblib.dump(model, os.path.join(artifact_dir, "rush_xg_gbm.joblib"))
    joblib.dump(feature_cols, os.path.join(artifact_dir, "rush_features_list.joblib"))
    print(f"\nArtifacts saved to {artifact_dir}")
    
    # 6. Example Simulator Usage
    print("\n--- Transition Simulator Example ---")
    # Take a random rush from the test set
    base = X_test.iloc[0]
    
    # Suppose we want to see the impact of tightening support (reducing dist) 
    # and increasing mean speed.
    adjustments = {
        'nearest_support_dist': -5.0, # Tighten support by 5 feet
        'mean_team_speed': 2.0        # Increase team speed by 2 m/s
    }
    
    base_xg, sim_xg, delta, pct = simulate_tactics(base, adjustments, model, feature_cols)
    print(f"Base Predicted xG:      {base_xg:.5f}")
    print(f"Simulated Predicted xG: {sim_xg:.5f}")
    print(f"Value Generation Δ:     {delta:+.5f} ({pct:+.1f}%)")

if __name__ == "__main__":
    main()

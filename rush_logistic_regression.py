import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Independent Variables (Features / X)
    features = [
        'SupportDist', 
        'LaneSpread', 
        'DepthRange', 
        'MeanSpeed', 
        'Pressure',
        'carrier_speed',
        'carrier_acceleration',
        'carrier_speed_relative_to_team',
        'second_support_distance',
        'support_triangle_area',
        'support_angle',
        'lane_balance',
        'max_lane_width',
        'weakside_support_distance',
        'nearest_defender_distance',
        'nearest_defender_speed',
        'number_of_defenders_between_puck_and_goal',
        'distance_to_blue_line',
        'angle_to_net',
        'rush_direction_speed'
    ]
    X = df[features].copy()
    
    # Add constant term (intercept: alpha_0 and beta_0 in your formulas)
    X = sm.add_constant(X)
    
    # Dependent Variables (Targets / y)
    y_shot = df['Shot_10s']       # Target 1
    y_entry = df['ControlledEntry'] # Target 2
    
    return X, y_shot, y_entry

def train_and_evaluate_model(X, y, target_name):
    print(f"\n{target_name}")
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Logistic Regression Model (using maximum likelihood estimation)
    model = sm.Logit(y_train, X_train)
    result = model.fit(disp=False)
    
    # Print the statistical summary 
    # (Shows coefficients, standard errors, z-scores, p-values)
    print(result.summary())
    
    # Evaluate on the test set
    y_pred_prob = result.predict(X_test)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"AUC: {auc:.4f}")
    except ValueError:
        pass
        
    print(classification_report(y_test, y_pred_class))
    
    return result

def generate_synthetic_data(n_samples=2000):
    
    np.random.seed(42)
    
    # Generate random features
    data = pd.DataFrame({
        'SupportDist': np.random.normal(15, 5, n_samples).clip(0),
        'LaneSpread': np.random.normal(20, 8, n_samples).clip(0),
        'DepthRange': np.random.normal(10, 4, n_samples).clip(0),
        'MeanSpeed': np.random.normal(25, 6, n_samples).clip(0),
        'Pressure': np.random.normal(5, 2, n_samples).clip(0)
    })
    
    # Simulate realistic logistical probabilities using dummy coefficients
    
    # 1. Shot within 10 seconds (P(Shot_10s = 1 | X))
    # z = Beta_0 + Beta_1(SupportDist) + ...
    z_shot = -1.5 + 0.05*data['SupportDist'] + 0.08*data['LaneSpread'] - 0.15*data['DepthRange'] + 0.12*data['MeanSpeed'] - 0.4*data['Pressure']
    prob_shot = 1 / (1 + np.exp(-z_shot))
    data['Shot_10s'] = np.random.binomial(1, prob_shot)
    
    # 2. Probability of controlled zone entry (P(ControlledEntry = 1 | X))
    # z = Alpha_0 + Alpha_1(SupportDist) + ...
    z_entry = -0.5 + 0.1*data['SupportDist'] + 0.15*data['LaneSpread'] - 0.05*data['DepthRange'] + 0.2*data['MeanSpeed'] - 0.5*data['Pressure']
    prob_entry = 1 / (1 + np.exp(-z_entry))
    data['ControlledEntry'] = np.random.binomial(1, prob_entry)
    
    X = data[['SupportDist', 'LaneSpread', 'DepthRange', 'MeanSpeed', 'Pressure']]
    X = sm.add_constant(X)
    y_shot = data['Shot_10s']
    y_entry = data['ControlledEntry']
    
    return X, y_shot, y_entry

def main():
    data_path = "c:\\Data Cup\\processed data\\engineered_rushes.csv"
    
    if os.path.exists(data_path):
        X, y_shot, y_entry = load_data(data_path)
    else:
        X, y_shot, y_entry = generate_synthetic_data()

    train_and_evaluate_model(X, y_shot, "Target: P(Shot_10s = 1 | X)")
    train_and_evaluate_model(X, y_entry, "Target: P(ControlledEntry = 1 | X)")

if __name__ == "__main__":
    main()

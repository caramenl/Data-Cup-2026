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
        'nearest_support_dist', 'second_support_dist', 'teammates_in_radius',
        'lane_spread', 'max_lane_width', 'lane_balance',
        'depth_range', 'depth_variance', 'is_flat_line',
        'mean_team_speed', 'speed_variance', 'carrier_speed', 'carrier_acceleration',
        'nearest_defender_dist', 'defender_closing_speed', 'defenders_between'
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
        'nearest_support_dist': np.random.normal(15, 5, n_samples).clip(0),
        'second_support_dist': np.random.normal(25, 8, n_samples).clip(0),
        'teammates_in_radius': np.random.poisson(1.5, n_samples),
        'lane_spread': np.random.normal(20, 8, n_samples).clip(0),
        'max_lane_width': np.random.normal(40, 10, n_samples).clip(0),
        'lane_balance': np.random.normal(5, 2, n_samples).clip(0),
        'depth_range': np.random.normal(10, 4, n_samples).clip(0),
        'depth_variance': np.random.normal(15, 5, n_samples).clip(0),
        'is_flat_line': np.random.binomial(1, 0.2, n_samples),
        'mean_team_speed': np.random.normal(25, 6, n_samples).clip(0),
        'speed_variance': np.random.normal(5, 2, n_samples).clip(0),
        'carrier_speed': np.random.normal(25, 5, n_samples).clip(0),
        'carrier_acceleration': np.random.normal(0, 2, n_samples),
        'nearest_defender_dist': np.random.normal(10, 5, n_samples).clip(0),
        'defender_closing_speed': np.random.normal(5, 2, n_samples).clip(0),
        'defenders_between': np.random.poisson(2, n_samples)
    })
    
    # Simulate realistic logistical probabilities
    z_shot = -1.5 + 0.1*data['carrier_speed'] - 0.2*data['nearest_defender_dist'] + 0.05*data['lane_spread']
    prob_shot = 1 / (1 + np.exp(-z_shot))
    data['Shot_10s'] = np.random.binomial(1, prob_shot)
    
    z_entry = -0.5 + 0.15*data['mean_team_speed'] - 0.3*data['defenders_between'] + 0.1*data['max_lane_width']
    prob_entry = 1 / (1 + np.exp(-z_entry))
    data['ControlledEntry'] = np.random.binomial(1, prob_entry)
    
    features = [
        'nearest_support_dist', 'second_support_dist', 'teammates_in_radius',
        'lane_spread', 'max_lane_width', 'lane_balance',
        'depth_range', 'depth_variance', 'is_flat_line',
        'mean_team_speed', 'speed_variance', 'carrier_speed', 'carrier_acceleration',
        'nearest_defender_dist', 'defender_closing_speed', 'defenders_between'
    ]
    X = data[features]
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

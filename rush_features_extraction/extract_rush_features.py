import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

FEET_TO_METERS = 0.3048

def calculate_speed(tracking_df, time_window=1.0):
    # Sort by player and time to calculate speed
    df = tracking_df.sort_values(['Player Id', 'Seconds'])
    df['dt'] = df.groupby('Player Id')['Seconds'].diff()
    df['dx'] = df.groupby('Player Id')['Rink Location X (Feet)'].diff() * FEET_TO_METERS
    df['dy'] = df.groupby('Player Id')['Rink Location Y (Feet)'].diff() * FEET_TO_METERS
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed'] = np.where(df['dt'] > 0, df['dist'] / df['dt'], 0)
    return df

def calculate_proxy_xg(event_row):
    """
    Proxy xG model based on shot type and location.
    Calculates the expected goal value for a single shot event.
    Net center assumed at (89, 0) feet.
    """
    if event_row['Event'] not in ['Shot', 'Goal']:
        return 0.0
        
    x = event_row['X_Coordinate']
    y = event_row['Y_Coordinate']
    
    # Distance to center of net (89, 0)
    # Using abs(x) to handle both sides if needed, but usually 89 is target end
    dist = np.sqrt((89 - abs(x))**2 + (y)**2)
    
    # Angle to net center (degrees)
    angle = np.degrees(np.abs(np.arctan2(y, 89 - abs(x))))
    
    # Base probability by shot type (based on NHL averages)
    shot_type = str(event_row['Detail_1'])
    base_probs = {
        'Wristshot': 0.08,
        'Snapshot': 0.09,
        'Slapshot': 0.06,
        'Deflection': 0.15,
        'Backhand': 0.07,
        'Tip-In': 0.20,
        'Wrap Around': 0.10
    }
    prob = base_probs.get(shot_type, 0.05)
    
    # Heuristic decay for distance and angle
    # Distance multiplier: exponential decay starting after 10ft
    dist_mult = np.exp(-0.035 * max(0, dist - 10))
    # Angle multiplier: cos of angle (penalty for wide angles)
    angle_mult = np.cos(np.radians(angle))
    
    # Final proxy xG value
    final_xg = prob * dist_mult * max(0.1, angle_mult)
    
    return final_xg

def extract_frame_features(t_frame, attacking_team):
    puck = t_frame[t_frame['Player or Puck'] == 'Puck']
    players = t_frame[t_frame['Player or Puck'] == 'Player']
    
    if players.empty or puck.empty:
        return None
    
    puck_x_vals = puck['Rink Location X (Feet)'].values
    if len(puck_x_vals) == 0: return None
    puck_x = puck_x_vals[0]
    puck_y = puck['Rink Location Y (Feet)'].values[0]
    
    # Determine attackers and defenders based on Team_Name
    attackers = players[players['Team_Name'] == attacking_team].copy()
    defenders = players[players['Team_Name'] != attacking_team].copy()
    
    # Need at least a few players to measure structure
    if len(attackers) < 3 or len(defenders) < 3: 
        return None
        
    # Identify Puck Carrier (Attacker closest to Puck)
    attackers['Dist_to_Puck'] = np.sqrt((attackers['Rink Location X (Feet)'] - puck_x)**2 + 
                                        (attackers['Rink Location Y (Feet)'] - puck_y)**2)
    
    valid_attackers = attackers.dropna(subset=['Dist_to_Puck'])
    if valid_attackers.empty:
        return None
        
    pcarrier = valid_attackers.loc[valid_attackers['Dist_to_Puck'].idxmin()]
    pc_x = pcarrier['Rink Location X (Feet)']
    pc_y = pcarrier['Rink Location Y (Feet)']
    
    # --- 1. Support Structure ---
    other_attackers = attackers[attackers.index != pcarrier.name].copy()
    other_attackers['Dist_to_PC'] = np.sqrt((other_attackers['Rink Location X (Feet)'] - pc_x)**2 + 
                                            (other_attackers['Rink Location Y (Feet)'] - pc_y)**2)
    sorted_supporters = other_attackers.sort_values('Dist_to_PC')
    
    # Nearest teammate distance
    nearest_support_dist = sorted_supporters.iloc[0]['Dist_to_PC'] if not sorted_supporters.empty else np.nan
    # Second closest support distance
    second_support_dist = sorted_supporters.iloc[1]['Dist_to_PC'] if len(sorted_supporters) >= 2 else np.nan
    # Number of teammates within support radius (25 feet)
    teammates_in_radius = (other_attackers['Dist_to_PC'] < 25.0).sum()
    
    # --- 2. Lane Structure ---
    attackers_y = attackers['Rink Location Y (Feet)']
    lane_spread = attackers_y.std() # standard deviation of Y positions
    max_lane_width = attackers_y.max() - attackers_y.min() # total lateral width
    
    # Distribution across L/M/R lanes
    left_count = (attackers_y < -14).sum()
    middle_count = ((attackers_y >= -14) & (attackers_y <= 14)).sum()
    right_count = (attackers_y > 14).sum()
    lane_balance = np.var([left_count, middle_count, right_count])
    
    # --- 3. Depth Layering ---
    attackers_x = attackers['Rink Location X (Feet)']
    depth_range = attackers_x.max() - attackers_x.min() # range of X positions
    depth_variance = attackers_x.var() # variance in forward depth
    is_flat_line = 1 if depth_range < 5.0 else 0 # indicator of "flat-line" formation
    
    # --- 4. Team Speed ---
    mean_team_speed = attackers['speed'].mean() # mean forward speed
    speed_variance = attackers['speed'].var() # variance in player speeds
    carrier_speed = pcarrier['speed'] # baseline speed
    
    # --- 5. Defensive Pressure ---
    defenders = defenders.copy()
    defenders['Dist_to_PC'] = np.sqrt((defenders['Rink Location X (Feet)'] - pc_x)**2 + 
                                      (defenders['Rink Location Y (Feet)'] - pc_y)**2)
    closest_defenders = defenders.sort_values('Dist_to_PC')
    
    nearest_defender_dist = closest_defenders.iloc[0]['Dist_to_PC'] if not closest_defenders.empty else np.nan
    defender_closing_speed = closest_defenders.iloc[0]['speed'] if not closest_defenders.empty else np.nan
    
    # Number of defenders between puck carrier and goal (count on attack side of X)
    # Heuristic for attack direction: towards boards (89 or -89)
    # We'll use the already calculated attack_dir logic
    if hasattr(pcarrier, 'dx') and not pd.isna(pcarrier['dx']):
        attack_dir = np.sign(pcarrier['dx'])
    else:
        attack_dir = 1
    if attack_dir == 0: attack_dir = 1
    
    if attack_dir > 0:
        defenders_between = defenders[defenders['Rink Location X (Feet)'] > pc_x].shape[0]
    else:
        defenders_between = defenders[defenders['Rink Location X (Feet)'] < pc_x].shape[0]
    
    return {
        'nearest_support_dist': nearest_support_dist,
        'second_support_dist': second_support_dist,
        'teammates_in_radius': teammates_in_radius,
        'lane_spread': lane_spread,
        'max_lane_width': max_lane_width,
        'lane_balance': lane_balance,
        'depth_range': depth_range,
        'depth_variance': depth_variance,
        'is_flat_line': is_flat_line,
        'mean_team_speed': mean_team_speed,
        'speed_variance': speed_variance,
        'carrier_speed': carrier_speed,
        'nearest_defender_dist': nearest_defender_dist,
        'defender_closing_speed': defender_closing_speed,
        'defenders_between': defenders_between
    }

def process_game(events_path, tracking_path):
    events = pd.read_csv(events_path)
    tracking = pd.read_csv(tracking_path)
    
    # We only care about valid play frames
    tracking = calculate_speed(tracking)
    
    # Define Episode Start Events
    recovery_events = events[events['Event'].isin(['Puck Recovery', 'Takeaway', 'Rebound Recovery'])].copy()
    
    rush_records = []
    
    for idx, event in recovery_events.iterrows():
        start_time = event['Seconds']
        period = event['Period']
        attacking_team = event['Team']
        
        # Avoid edge cases near the end of the period
        if pd.isna(start_time): continue
        
        # 1. Feature Extraction Window (0 to 2 seconds after recovery)
        feature_window_frames = tracking.loc[
            (tracking['Period'] == period) & 
            (tracking['Seconds'] >= start_time) &
            (tracking['Seconds'] <= start_time + 2.0)
        ]
        
        if feature_window_frames.empty:
            continue
            
        # Extract features for every unique timestamp in the 2-second window
        frame_times = feature_window_frames['Seconds'].unique()
        window_features = []
        
        for t in frame_times:
            t_frame = feature_window_frames[feature_window_frames['Seconds'] == t]
            feats = extract_frame_features(t_frame, attacking_team)
            if feats:
                window_features.append(feats)
                
        if not window_features:
            continue
            
        # Take the mean of structural features across the initial acceleration window
        feat_df = pd.DataFrame(window_features)
        
        # Calculate carrier acceleration as change in carrier speed over the actual window duration
        if len(feat_df) >= 2:
            carrier_accel = (feat_df['carrier_speed'].iloc[-1] - feat_df['carrier_speed'].iloc[0]) / (frame_times[-1] - frame_times[0])
        else:
            carrier_accel = 0.0
            
        mean_feats = feat_df.mean().to_dict()
        mean_feats['carrier_acceleration'] = carrier_accel
        
        # 2. Outcome Window (2 to 15 seconds after recovery)
        outcome_window = events.loc[
            (events['Period'] == period) & 
            (events['Seconds'] > start_time + 2.0) & 
            (events['Seconds'] <= start_time + 15.0) &
            (events['Team'] == attacking_team)
        ]
        
        # Target: Shot within 2-15 seconds
        shot_outcome = outcome_window[outcome_window['Event'].isin(['Shot', 'Goal'])]
        shot_10s = 1 if len(shot_outcome) > 0 else 0
        
        # NEW Target: Continuous xG sum within 15 seconds
        xg_sum_15s = outcome_window.apply(calculate_proxy_xg, axis=1).sum()
        
        # Target: Controlled Zone Entry within 2-15 seconds
        entry_outcome = outcome_window[
            (outcome_window['Event'] == 'Zone Entry') & 
            (outcome_window['Detail_1'].isin(['Carried', 'Played']))
        ]
        controlled_entry = 1 if len(entry_outcome) > 0 else 0
        
        rush_records.append({
            'Game_ID': os.path.basename(events_path).replace('_Events_Cleaned.csv', ''),
            'Period': period,
            'Seconds': start_time,
            'Attacking_Team': attacking_team,
            **mean_feats,
            'ControlledEntry': controlled_entry,
            'Shot_10s': shot_10s,
            'xG_15s': xg_sum_15s
        })
        
    return pd.DataFrame(rush_records)

def main():
    data_dir = "C:\\Data Cup\\processed data"
    event_files = glob.glob(os.path.join(data_dir, "*_Events_Cleaned.csv"))
    
    all_rushes = []
    
    print(f"Found {len(event_files)} games to process.")
    for e_file in tqdm(event_files):
        t_file = e_file.replace('_Events_Cleaned.csv', '_Tracking_Cleaned.csv')
        if os.path.exists(t_file):
            df = process_game(e_file, t_file)
            if not df.empty:
                all_rushes.append(df)
        else:
            print(f"Missing tracking for {e_file}")
            
    if all_rushes:
        final_df = pd.concat(all_rushes, ignore_index=True)
        # Drop rows with NaNs in core features to ensure cleanly defined rushes
        features = [
            'nearest_support_dist', 'second_support_dist', 'teammates_in_radius',
            'lane_spread', 'max_lane_width', 'lane_balance',
            'depth_range', 'depth_variance', 'is_flat_line',
            'mean_team_speed', 'speed_variance', 'carrier_speed', 'carrier_acceleration',
            'nearest_defender_dist', 'defender_closing_speed', 'defenders_between',
            'xG_15s'
        ]
        final_df.dropna(subset=features, inplace=True)
        
        out_path = os.path.join(data_dir, "engineered_rushes.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Successfully extracted {len(final_df)} structured rushes and saved to {out_path}")
    else:
        print("No valid rushes found.")

if __name__ == "__main__":
    main()

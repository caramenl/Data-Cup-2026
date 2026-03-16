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
    
    # F1: SupportDist
    other_attackers = attackers[attackers.index != pcarrier.name]
    support_dist = np.mean(np.sqrt((other_attackers['Rink Location X (Feet)'] - pc_x)**2 + 
                                   (other_attackers['Rink Location Y (Feet)'] - pc_y)**2))
    
    # F2: LaneSpread
    lane_spread = attackers['Rink Location Y (Feet)'].std()
    
    # F3: DepthRange
    depth_range = attackers['Rink Location X (Feet)'].max() - attackers['Rink Location X (Feet)'].min()
    
    # F4: MeanSpeed (Team Mean Speed)
    mean_speed = attackers['speed'].mean()
    
    # --- NEW: Carrier Dynamics ---
    carrier_speed = pcarrier['speed']
    # Acceleration proxy: change in speed over the 0.1s frame interval 
    # (Assuming tracking is 10Hz, dt is often ~0.1s)
    # We will estimate carrier_acceleration in process_game instead to use a true window,
    # or just use the difference from previous frame if tracking has it. For frame-level,
    # we'll extract speed and handle acceleration at the episode level.
    carrier_speed_relative_to_team = carrier_speed - mean_speed

    # distance to all other attackers
    other_attackers = other_attackers.copy()
    other_attackers['Dist_to_Carrier'] = np.sqrt((other_attackers['Rink Location X (Feet)'] - pc_x)**2 + 
                                                 (other_attackers['Rink Location Y (Feet)'] - pc_y)**2)
    # Sort supporters by distance
    sorted_supporters = other_attackers.sort_values('Dist_to_Carrier')
    
    # second_support_distance
    if len(sorted_supporters) >= 2:
        second_support_distance = sorted_supporters.iloc[1]['Dist_to_Carrier']
        
        # support_triangle_area
        # Coordinates of triangle vertices
        p1 = np.array([pc_x, pc_y])
        p2 = np.array([sorted_supporters.iloc[0]['Rink Location X (Feet)'], sorted_supporters.iloc[0]['Rink Location Y (Feet)']])
        p3 = np.array([sorted_supporters.iloc[1]['Rink Location X (Feet)'], sorted_supporters.iloc[1]['Rink Location Y (Feet)']])
        # Area = 0.5 * |xA(yB - yC) + xB(yC - yA) + xC(yA -yB)|
        support_triangle_area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        
        # support_angle (angle between p1->p2 and p1->p3)
        v1 = p2 - p1
        v2 = p3 - p1
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        support_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    else:
        second_support_distance = np.nan
        support_triangle_area = np.nan
        support_angle = np.nan
    
    # --- NEW: Lane Structure ---
    # Left (-42.5 to -14), Middle (-14 to 14), Right (14 to 42.5)
    attackers_y = attackers['Rink Location Y (Feet)']
    left_lane = (attackers_y < -14).sum()
    middle_lane = ((attackers_y >= -14) & (attackers_y <= 14)).sum()
    right_lane = (attackers_y > 14).sum()
    # Lane balance: variance of occupancy across the 3 lanes (lower variance = more balanced)
    lane_balance = np.var([left_lane, middle_lane, right_lane])
    
    max_lane_width = attackers_y.max() - attackers_y.min()
    
    # Weakside support distance (furthest attacker from carrier)
    if not other_attackers.empty:
        weakside_support_distance = other_attackers['Dist_to_Carrier'].max()
    else:
        weakside_support_distance = np.nan
    
    # F5: Pressure & --- NEW: Defender Pressure ---
    defenders['Dist_to_Carrier'] = np.sqrt((defenders['Rink Location X (Feet)'] - pc_x)**2 + 
                                           (defenders['Rink Location Y (Feet)'] - pc_y)**2)
    closest_defenders = defenders.sort_values('Dist_to_Carrier')
    
    if not closest_defenders.empty:
        pressure = closest_defenders.head(2)['Dist_to_Carrier'].mean()
        nearest_defender_distance = closest_defenders.iloc[0]['Dist_to_Carrier']
        
        # Defender closing speed (proxy: difference between defender speed towards carrier and carrier speed away)
        # For a single frame, taking the absolute speed of nearest defender as a simple metric if directional isn't trivial.
        # We will calculate a strict closing speed over the episode window later, or use relative speeds here.
        nearest_def = closest_defenders.iloc[0]
        # Approximation: speed of nearest defender + (relative motion towards carrier)
        # Using a simpler proxy for the frame level: nearest defender speed
        nearest_defender_speed = nearest_def['speed'] 
        
        # Number of defenders between puck and goal. 
        # Assume attacking net is at X=89 feet (typical NHL rink center of goal line)
        attacking_net_x = 89 * np.sign(pc_x) if pc_x != 0 else 89 # simplistic direction heuristic based on carrier half, though real is known from Period/Team
        # A more robust direction check using the team's mean X over time could be here, but we'll use sign(pc_x) as a proxy
        # for an established rush if they are already over center ice. If they start in D zone, this might be backwards.
        # Let's assume play flows towards +X or -X based on the event's "X Coordinate" vs outcome, or just use X distance.
        # Actually, let's just count defenders whose X is between the carrier and the end boards in the direction they are moving.
        # We know carrier's X-velocity (dx). If dx is positive, they are attacking +X. If negative, -X.
        if hasattr(pcarrier, 'dx') and not pd.isna(pcarrier['dx']):
             attack_dir = np.sign(pcarrier['dx'])
        else:
             attack_dir = 1 # default
             
        if attack_dir == 0: attack_dir = 1
        
        # Defenders "between" puck and goal (i.e. further down the ice in attack direction)
        if attack_dir > 0:
            defenders_between = defenders[defenders['Rink Location X (Feet)'] > pc_x].shape[0]
        else:
            defenders_between = defenders[defenders['Rink Location X (Feet)'] < pc_x].shape[0]
            
    else:
        pressure = np.nan
        nearest_defender_distance = np.nan
        nearest_defender_speed = np.nan
        defenders_between = 0
        
    # --- NEW: Transition Geometry ---
    # distance_to_blue_line. Attacking blue line is at X = 25 (if attacking +) or -25 (if attacking -)
    attacking_blue_line_x = 25 * attack_dir
    distance_to_blue_line = abs(attacking_blue_line_x - pc_x)
    
    # angle_to_net
    # Net is at X=89 (or -89), Y=0. Vector from carrier to net:
    dx_net = attacking_net_x - pc_x
    dy_net = 0 - pc_y
    angle_to_net = np.degrees(np.abs(np.arctan2(dy_net, dx_net)))
    
    # rush_direction_speed
    # Component of carrier speed in the attack direction (X-axis)
    if hasattr(pcarrier, 'dx') and hasattr(pcarrier, 'dt') and pcarrier['dt'] > 0:
        rush_direction_speed = (pcarrier['dx'] / pcarrier['dt']) * attack_dir
    else:
        # fallback to simple speed if dx/dt unavailable
        rush_direction_speed = carrier_speed * 1.0  # assume largely forward
    
    return {
        'SupportDist': support_dist,
        'LaneSpread': lane_spread,
        'DepthRange': depth_range,
        'MeanSpeed': mean_speed,
        'Pressure': pressure,
        'carrier_speed': carrier_speed,
        'carrier_speed_relative_to_team': carrier_speed_relative_to_team,
        'second_support_distance': second_support_distance,
        'support_triangle_area': support_triangle_area,
        'support_angle': support_angle,
        'lane_balance': lane_balance,
        'max_lane_width': max_lane_width,
        'weakside_support_distance': weakside_support_distance,
        'nearest_defender_distance': nearest_defender_distance,
        'nearest_defender_speed': nearest_defender_speed,
        'number_of_defenders_between_puck_and_goal': defenders_between,
        'distance_to_blue_line': distance_to_blue_line,
        'angle_to_net': angle_to_net,
        'rush_direction_speed': rush_direction_speed
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
        
        # 2. Outcome Window (2 to 10 seconds after recovery)
        outcome_window = events.loc[
            (events['Period'] == period) & 
            (events['Seconds'] > start_time + 2.0) & 
            (events['Seconds'] <= start_time + 10.0) &
            (events['Team'] == attacking_team)
        ]
        
        # Target: Shot within 2-10 seconds
        shot_outcome = outcome_window[outcome_window['Event'].isin(['Shot', 'Goal'])]
        shot_10s = 1 if len(shot_outcome) > 0 else 0
        
        # Target: Controlled Zone Entry within 2-10 seconds
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
            'Shot_10s': shot_10s
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
            'SupportDist', 'LaneSpread', 'DepthRange', 'MeanSpeed', 'Pressure',
            'carrier_speed', 'carrier_speed_relative_to_team', 'second_support_distance',
            'support_triangle_area', 'support_angle', 'lane_balance', 'max_lane_width',
            'weakside_support_distance', 'nearest_defender_distance', 'nearest_defender_speed',
            'number_of_defenders_between_puck_and_goal', 'distance_to_blue_line',
            'angle_to_net', 'rush_direction_speed', 'carrier_acceleration'
        ]
        final_df.dropna(subset=features, inplace=True)
        
        out_path = os.path.join(data_dir, "engineered_rushes.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Successfully extracted {len(final_df)} structured rushes and saved to {out_path}")
    else:
        print("No valid rushes found.")

if __name__ == "__main__":
    main()

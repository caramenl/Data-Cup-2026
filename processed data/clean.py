import pandas as pd
import numpy as np
import os
import glob
import re

GOALIE_ID = 100
RINK_X_MAX = 100.0
MAX_PUCK_SPEED_KMH = 41.0
MAX_PUCK_SPEED_MS = MAX_PUCK_SPEED_KMH / 3.6
FEET_TO_METERS = 0.3048

# Convert MM:SS clock format to total seconds
def clock_to_seconds(clock_str):
    if pd.isna(clock_str) or not isinstance(clock_str, str):
        return clock_str
    try:
        minutes, seconds = clock_str.split(':')
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return np.nan

# Normalize Player IDs and map Goalie 'Go' to 100
def clean_player_id(player_id):
    if pd.isna(player_id):
        return player_id
    if str(player_id).lower() == 'go':
        return GOALIE_ID
    try:
        return int(float(player_id))
    except (ValueError, TypeError):
        return player_id

def process_game(events_path, shifts_path, tracking_paths):
    print(f"Processing game: {os.path.basename(events_path)}")
    
    # Load Events, Shifts, and Tracking data
    events = pd.read_csv(events_path)
    shifts = pd.read_csv(shifts_path)
    
    tracking_frames = []
    for p_path in tracking_paths:
        if os.path.exists(p_path):
            tracking_frames.append(pd.read_csv(p_path, low_memory=False))
    
    if not tracking_frames:
        print(f"Warning: No tracking data found for {events_path}")
        return
        
    tracking = pd.concat(tracking_frames, ignore_index=True)
    if 'Goal Score' in tracking.columns:
        tracking.drop(columns=['Goal Score'], inplace=True)
    
    # Standardize Player IDs across all datasets
    events['Player_Id'] = events['Player_Id'].apply(clean_player_id)
    shifts['Player_Id'] = shifts['Player_Id'].apply(clean_player_id)
    tracking['Player_Id_Cleaned'] = tracking['Player Jersey Number'].apply(clean_player_id)
    
    # Map Home/Away tracking teams to actual names
    home_team = events['Home_Team'].iloc[0]
    away_team = events['Away_Team'].iloc[0]
    
    tracking['Team_Name'] = tracking['Team'].map({'Home': home_team, 'Away': away_team})
    
    # Convert all time columns to numeric seconds
    events['Seconds'] = events['Clock'].apply(clock_to_seconds)
    tracking['Seconds'] = tracking['Game Clock'].apply(clock_to_seconds)
    shifts['Start_Seconds'] = shifts['start_clock'].apply(clock_to_seconds)
    shifts['End_Seconds'] = shifts['end_clock'].apply(clock_to_seconds)
    shifts['shift_length_seconds'] = shifts['shift_length'].apply(clock_to_seconds)
    
    # Normalize orientation (Flip X/Y in even periods)
    mask_flip = tracking['Period'].isin([2, 4])
    tracking.loc[mask_flip, 'Rink Location X (Feet)'] *= -1
    tracking.loc[mask_flip, 'Rink Location Y (Feet)'] *= -1
    
    # Coordinate Clipping to Rink Bounds
    tracking['Rink Location X (Feet)'] = tracking['Rink Location X (Feet)'].clip(-100, 100)
    tracking['Rink Location Y (Feet)'] = tracking['Rink Location Y (Feet)'].clip(-42.5, 42.5)
    events['X_Coordinate'] = events['X_Coordinate'].clip(-100, 100)
    events['Y_Coordinate'] = events['Y_Coordinate'].clip(-42.5, 42.5)

    # Game Strength Column
    events['Strength'] = events.apply(lambda row: f"{int(row['Home_Team_Skaters'])}v{int(row['Away_Team_Skaters'])}", axis=1)

    # Shift Validation Helper (Vectorized for performance)
    def calculate_shift_flags(data_df, shift_df, p_id_col='Player_Id', time_col='Seconds'):
        # Only check rows with a Player_Id (ignore Puck)
        valid_players = data_df[data_df[p_id_col].notna()].copy()
        if valid_players.empty:
            return pd.Series(0, index=data_df.index)
        
        # Merge data with all possible shifts for those players
        # Note: shifts dataframe always uses 'Player_Id' and 'period'
        merged = valid_players.reset_index().merge(
            shift_df[['Player_Id', 'period', 'Start_Seconds', 'End_Seconds']],
            left_on=[p_id_col, 'Period'],
            right_on=['Player_Id', 'period'],
            how='left'
        )
        
        # Check if clock time is within shift (Start >= Seconds >= End)
        merged['is_on_ice'] = (merged[time_col] <= merged['Start_Seconds']) & \
                               (merged[time_col] >= merged['End_Seconds'])
        
        # Group by the original index to see if the player was on ANY shift at that time
        on_ice_results = merged.groupby('index')['is_on_ice'].any()
        
        # Flag is 0 if on ice, 1 if not on ice (bench error)
        flags = pd.Series(0, index=data_df.index)
        flags.loc[valid_players.index] = (~on_ice_results).astype(int)
        return flags

    events['shift_flag'] = calculate_shift_flags(events, shifts)

    # Transform Incomplete Plays into Turnovers
    events['Next_Team'] = events['Team'].shift(-1)
    events['Next_Event'] = events['Event'].shift(-1)
    
    turnover_mask = (
        (events['Event'] == 'Incomplete Play') & 
        (events['Next_Event'] == 'Puck Recovery') & 
        (events['Team'] != events['Next_Team'])
    )
    events.loc[turnover_mask, 'Event'] = 'Turnover'
    
    # Interpolate missing puck coordinates & Speed Check
    puck_orig = tracking[tracking['Player or Puck'] == 'Puck'].copy()
    puck_data = puck_orig.sort_values(['Period', 'Seconds'], ascending=[True, False])
    
    # Linear interpolation for basic gaps
    puck_data['Rink Location X (Feet)'] = puck_data['Rink Location X (Feet)'].interpolate()
    puck_data['Rink Location Y (Feet)'] = puck_data['Rink Location Y (Feet)'].interpolate()
    
    # Speed check: if puck speed > 41kmh, invalidate and re-interpolate
    # Simple frame-to-frame check. Tracking freq is approx 30fps but we'll use Seconds delta.
    puck_data['dt'] = puck_data['Seconds'].diff().abs()
    puck_data['dx'] = puck_data['Rink Location X (Feet)'].diff() * FEET_TO_METERS
    puck_data['dy'] = puck_data['Rink Location Y (Feet)'].diff() * FEET_TO_METERS
    puck_data['dist'] = np.sqrt(puck_data['dx']**2 + puck_data['dy']**2)
    puck_data['speed'] = puck_data['dist'] / puck_data['dt']

    # Invalidate high speeds and re-interpolate
    high_speed = puck_data['speed'] > MAX_PUCK_SPEED_MS
    puck_data.loc[high_speed, ['Rink Location X (Feet)', 'Rink Location Y (Feet)']] = np.nan
    puck_data['Rink Location X (Feet)'] = puck_data['Rink Location X (Feet)'].interpolate()
    puck_data['Rink Location Y (Feet)'] = puck_data['Rink Location Y (Feet)'].interpolate()

    tracking.loc[tracking['Player or Puck'] == 'Puck', ['Rink Location X (Feet)', 'Rink Location Y (Feet)']] = \
        puck_data[['Rink Location X (Feet)', 'Rink Location Y (Feet)']].values

    # Remove low-quality frames (< 4 players)
    player_counts = tracking[tracking['Player or Puck'] == 'Player'].groupby(['Image Id']).size()
    valid_images = player_counts[player_counts >= 4].index
    tracking = tracking[tracking['Image Id'].isin(valid_images)]
    
    # Calculate shift flags for tracking (players only)
    tracking['shift_flag'] = calculate_shift_flags(
        tracking, shifts, p_id_col='Player_Id_Cleaned', time_col='Seconds'
    )
    
    # Save cleaned data
    game_base = os.path.basename(events_path).replace('.Events.csv', '')
    output_dir = os.path.join("c:\\", "Data Cup", "processed data")
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_events_path = os.path.join(output_dir, f"{game_base}_Events_Cleaned.csv")
    cleaned_tracking_path = os.path.join(output_dir, f"{game_base}_Tracking_Cleaned.csv")
    
    events.to_csv(cleaned_events_path, index=False)
    tracking.to_csv(cleaned_tracking_path, index=False)
    
    print(f"Saved: {game_base}")
    return cleaned_events_path, cleaned_tracking_path

def main():
    events_dir = os.path.join("c:\\", "Data Cup", "data", "events")
    events_files = glob.glob(os.path.join(events_dir, "*.csv"))
    
    for e_file in events_files:
        game_id = os.path.basename(e_file).replace('.Events.csv', '')
        s_file = e_file.replace('data\\events', 'data\\shifts').replace('.Events.csv', '.Shifts.csv')
        
        t_files = [
            e_file.replace('data\\events', 'data\\tracking').replace('.Events.csv', '.Tracking_P1.csv'),
            e_file.replace('data\\events', 'data\\tracking').replace('.Events.csv', '.Tracking_P2.csv'),
            e_file.replace('data\\events', 'data\\tracking').replace('.Events.csv', '.Tracking_P3.csv'),
            e_file.replace('data\\events', 'data\\tracking').replace('.Events.csv', '.Tracking_POT.csv'),
        ]
        
        process_game(e_file, s_file, t_files)

if __name__ == "__main__":
    main()
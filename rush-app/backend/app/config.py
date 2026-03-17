from pathlib import Path

FEATURES = [
    'nearest_support_dist',
    'second_support_dist',
    'teammates_in_radius',
    'lane_spread',
    'max_lane_width',
    'lane_balance',
    'depth_range',
    'depth_variance',
    'is_flat_line',
    'mean_team_speed',
    'speed_variance',
    'carrier_speed',
    'carrier_acceleration',
    'nearest_defender_dist',
    'defender_closing_speed',
    'defenders_between'
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
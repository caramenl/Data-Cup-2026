from pydantic import BaseModel

class RushFeatures(BaseModel):
    nearest_support_dist: float
    second_support_dist: float
    teammates_in_radius: float
    lane_spread: float
    max_lane_width: float
    lane_balance: float
    depth_range: float
    depth_variance: float
    is_flat_line: float
    mean_team_speed: float
    speed_variance: float
    carrier_speed: float
    carrier_acceleration: float
    nearest_defender_dist: float
    defender_closing_speed: float
    defenders_between: float


class PredictionResponse(BaseModel):
    shot_probability: float
    controlled_entry_probability: float
    xg_15s: float
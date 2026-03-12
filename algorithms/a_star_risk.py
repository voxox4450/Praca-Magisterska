from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import base_search
from config import RISK_WEIGHT, TURN_PENALTY_RISK, COLLISION_RADIUS


def run_risk_astar(
        grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
        risk_weight: float = RISK_WEIGHT,
        turn_penalty: float = TURN_PENALTY_RISK,
        drone_radius: float = COLLISION_RADIUS,
        initial_direction: Tuple[int, int] = (0, 0), current_speed: float = 0.0,
        initial_straight_dist: float = 0.0
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    return base_search(
        grid_map, start, goal, risk_weight, turn_penalty, drone_radius,
        initial_direction, current_speed,
        use_heuristic=True,
        use_kinematics=True,
        initial_straight_dist=initial_straight_dist
    )
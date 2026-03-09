from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import base_search


def run_risk_astar(
        grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
        risk_weight: float = 20.0, turn_penalty: float = 2.0, drone_radius: float = 3.0,
        initial_direction: Tuple[int, int] = (0, 0), current_speed: float = 0.0
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    return base_search(
        grid_map, start, goal, risk_weight, turn_penalty, drone_radius,
        initial_direction, current_speed,
        use_heuristic=True,
        use_kinematics=True
    )
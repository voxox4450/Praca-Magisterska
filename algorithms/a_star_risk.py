import math
from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import base_search
from config import (
    RISK_WEIGHT, TURN_PENALTY, COLLISION_RADIUS, DRONE_MASS_KG,
    MAX_THRUST_NET_N, MAX_LATERAL_ACCEL, MIN_TURN_SPEED, V_MAX_MS,
    TURN_RADIUS_CONST,
)


def _plan_braking_buffer(
        grid_map: GridMap,
        drone_pos: Tuple[int, int],
        heading: Tuple[int, int],
        current_speed: float,
        drone_mass: float
) -> Tuple[List[Tuple[int, int]], float, float]:
    if heading == (0, 0) or current_speed <= 0:
        return [], 0.0, current_speed

    accel = MAX_THRUST_NET_N / drone_mass

    r_135 = TURN_RADIUS_CONST / max(0.1, math.sin(math.radians(67.5)))
    v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_135))
    v_safe_ref = min(v_safe_ref, V_MAX_MS)

    if current_speed <= v_safe_ref:
        return [], 0.0, current_speed

    braking_need = (current_speed ** 2 - v_safe_ref ** 2) / (2.0 * accel)

    step_len = math.sqrt(heading[0] ** 2 + heading[1] ** 2)
    if step_len <= 0:
        return [], 0.0, current_speed

    buf_steps = max(1, int(math.ceil(braking_need / step_len)))

    buffer_points: List[Tuple[int, int]] = []
    buffer_dist = 0.0
    for d in range(1, buf_steps + 1):
        bp = (drone_pos[0] + heading[0] * d, drone_pos[1] + heading[1] * d)
        bx, by = int(bp[0]), int(bp[1])
        if 0 <= bx < grid_map.width and 0 <= by < grid_map.height:
            if not grid_map.collision_mask[bx, by]:
                buffer_points.append(bp)
                buffer_dist += step_len
            else:
                break
        else:
            break

    v_after = math.sqrt(max(0.0, current_speed ** 2 - 2.0 * accel * buffer_dist))
    return buffer_points, buffer_dist, v_after


def run_risk_astar(
        grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
        risk_weight: float = RISK_WEIGHT,
        turn_penalty: float = TURN_PENALTY,
        drone_radius: float = COLLISION_RADIUS,
        initial_direction: Tuple[int, int] = (0, 0), current_speed: float = 0.0,
        initial_straight_dist: float = 0.0,
        drone_mass: float = DRONE_MASS_KG
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    buffer_points, buffer_dist, v_after_buffer = _plan_braking_buffer(
        grid_map, start, initial_direction, current_speed, drone_mass
    )

    search_start = buffer_points[-1] if buffer_points else start
    search_start_int = (int(search_start[0]), int(search_start[1]))

    path, stats = base_search(
        grid_map, search_start_int, goal,
        risk_weight, turn_penalty, drone_radius,
        initial_direction, v_after_buffer,
        use_heuristic=True,
        use_kinematics=True,
        initial_straight_dist=initial_straight_dist + buffer_dist,
        drone_mass=drone_mass
    )

    if path and stats.get('found') and buffer_points:
        full_path = [start] + buffer_points + path[1:]
        # Eksponuj informacje o buforze do warstwy wizualizacji / metryk
        stats['buffer_points'] = buffer_points
        stats['buffer_dist'] = buffer_dist
        stats['v_after_buffer'] = v_after_buffer
    else:
        full_path = path
        stats['buffer_points'] = []
        stats['buffer_dist'] = 0.0
        stats['v_after_buffer'] = current_speed

    return full_path, stats
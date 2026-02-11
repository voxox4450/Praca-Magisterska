import heapq
import time
import math
from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import Node, reconstruct_path


def run_risk_astar(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        risk_weight: float = 20.0,
        turn_penalty: float = 2.0  # Kara za skręt (dla płynności)
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()

    start_node = Node(start[0], start[1], 0.0, direction=(0, 0))
    open_list = []
    heapq.heappush(open_list, start_node)

    g_score = {(start[0], start[1]): 0.0}
    visited = set()
    nodes_expanded = 0

    # Promień drona
    DRONE_RADIUS = 2.0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if (current.x, current.y) == goal:
            execution_time = time.time() - t0
            path, length, total_risk, turns = reconstruct_path(current, grid_map)
            return path, {
                "found": True, "time": execution_time, "length": length,
                "risk": total_risk, "turns": turns, "nodes": nodes_expanded
            }

        if (current.x, current.y) in visited:
            continue
        visited.add((current.x, current.y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = current.x + dx, current.y + dy

            # 1. Sprawdzenie kolizji fizycznej (wymiary drona)
            if grid_map.is_collision(nx, ny, drone_radius=DRONE_RADIUS):
                continue

            # 2. Pobranie wartości ryzyka (gradientu)
            cell_risk = grid_map.get_cost(nx, ny)

            # Koszt odległości
            dist_cost = math.sqrt(dx ** 2 + dy ** 2)

            # Koszt Ryzyka (Statyczny)
            static_risk_cost = cell_risk * risk_weight

            # Koszt Skrętu (Dla estetyki/płynności - unikanie zygzaków)
            turn_cost = 0.0
            if current.parent is not None and current.direction != (dx, dy):
                turn_cost = turn_penalty

            # Suma kosztów (bez Momentum)
            new_g = current.cost + dist_cost + static_risk_cost + turn_cost

            if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_g
                h = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2)

                # Zapisujemy direction, aby w następnym kroku wiedzieć czy skręcamy
                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy), heuristic=h)
                heapq.heappush(open_list, neighbor)

    return [], {"found": False, "time": 0, "length": 0, "risk": 0, "turns": 0, "nodes": nodes_expanded}
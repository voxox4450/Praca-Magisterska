import heapq
import time
import math
from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import Node, reconstruct_path, calculate_kinematic_flight_time


def run_dijkstra(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        collision_radius: float = 3.0
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()

    # Dodano inicjalizację kierunku, aby móc wyłapać zakręty
    start_node = Node(start[0], start[1], 0.0, direction=(0, 0))
    open_list = []
    heapq.heappush(open_list, start_node)
    g_score = {(start[0], start[1]): 0.0}
    visited = set()
    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if (current.x, current.y) == goal:
            execution_time = time.time() - t0
            path, length, total_risk, turns = reconstruct_path(current, grid_map)

            # --- OBLICZANIE FIZYCZNEGO CZASU LOTU ---
            flight_time = calculate_kinematic_flight_time(path, mass=30.0, max_thrust_net=120.0, v_max_kmh=65.0)

            return path, {
                "found": True, "time": execution_time, "length": length,
                "risk": total_risk, "turns": turns, "nodes": nodes_expanded,
                "flight_time": flight_time  # <--- NOWA DANA
            }

        if (current.x, current.y) in visited:
            continue
        visited.add((current.x, current.y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = current.x + dx, current.y + dy

            # Sprawdzamy czy dron się zmieści fizycznie
            if grid_map.is_collision(nx, ny, drone_radius=collision_radius):
                continue

            dist_cost = math.sqrt(dx ** 2 + dy ** 2)

            # --- ZMIANA: TIE-BREAKER ELIMINUJĄCY ZYGZAKI ---
            # Dodajemy mikroskopijną karę, jeśli dron nie leci prosto.
            # To nie zmienia realnego wyniku trasy, ale zmusza algorytm do rysowania "ładnych" prostych linii.
            turn_cost = 0.0
            if current.parent is not None:
                if current.direction != (dx, dy):
                    turn_cost = 0.001

            new_g = current.cost + dist_cost + turn_cost

            if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_g
                # Zapisujemy obrany kierunek (dx, dy)
                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy), heuristic=0.0)
                heapq.heappush(open_list, neighbor)

    return [], {"found": False, "time": 0, "length": 0, "risk": 0, "turns": 0, "nodes": nodes_expanded}
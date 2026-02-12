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
        turn_penalty: float = 2.0,  # Kara za skręt (dla płynności)
        drone_radius: float = 3.0
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()

    start_node = Node(start[0], start[1], 0.0, direction=(0, 0))
    open_list = []
    heapq.heappush(open_list, start_node)

    g_score = {(start[0], start[1]): 0.0}
    visited = set()
    nodes_expanded = 0

    # Promień drona


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

            if grid_map.is_collision(nx, ny, drone_radius=drone_radius):
                continue

            cell_risk = grid_map.get_cost(nx, ny)
            dist_cost = math.sqrt(dx ** 2 + dy ** 2)
            static_risk_cost = cell_risk * risk_weight

            # --- POPRAWKA 1: DYNAMICZNA KARA ZA SKRĘT ---
            turn_cost = 0.0
            if current.parent is not None:
                # v1: poprzedni kierunek, v2: obecny kierunek
                v1 = current.direction
                v2 = (dx, dy)

                if v1 != v2:
                    # Obliczamy "ostrość" skrętu.
                    # Jeśli zmieniamy kierunek tylko o 45 stopni, kara powinna być mniejsza.
                    # Iloczyn skalarny v1*v2 powie nam o kącie.
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

                    # Normalizacja dla ruchów diagonalnych
                    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                    cos_theta = dot_product / (mag1 * mag2)

                    # Zapobiegamy błędom zaokrągleń dla acos
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    angle = math.acos(cos_theta)  # kąt w radianach

                    # Kara proporcjonalna do kąta (turn_penalty * kąt)
                    turn_cost = turn_penalty * (angle / (math.pi / 2))

            new_g = current.cost + dist_cost + static_risk_cost + turn_cost

            if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_g

                # --- POPRAWKA 2: TIE-BREAKER (PROSTOWANIE TRASY) ---
                # Mnożymy heurystykę przez 1.001, aby algorytm preferował
                # punkty bliższe linii prostej do celu przy tej samej wadze.
                h = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2) * 1.001

                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy), heuristic=h)
                heapq.heappush(open_list, neighbor)

    return [], {"found": False, "time": 0, "length": 0, "risk": 0, "turns": 0, "nodes": nodes_expanded}
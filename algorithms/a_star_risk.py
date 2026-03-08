import heapq
import time
import math
from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import Node, reconstruct_path, calculate_kinematic_flight_time


def run_risk_astar(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        risk_weight: float = 20.0,
        turn_penalty: float = 2.0,
        drone_radius: float = 3.0,
        initial_direction: Tuple[int, int] = (0, 0),
        current_speed: float = 0.0
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()

    start_node = Node(start[0], start[1], 0.0, direction=initial_direction)
    # Zamiast physical_dist (dystans całkowity), śledzimy dystans przeleciany prosto od ostatniego zakrętu
    start_node.straight_dist = 0.0
    start_node.straight_steps = 100  # Dron na start jest ustabilizowany (może wykonać 1 manewr)

    open_list = []
    heapq.heappush(open_list, start_node)

    # --- NOWOŚĆ: Kinematyczny klucz stanu A* (X, Y, Wektor X, Wektor Y) ---
    # Algorytm wie, z której strony nadlatuje, by blokować nakładające się zakręty!
    g_score = {(start[0], start[1], initial_direction[0], initial_direction[1]): 0.0}
    visited = set()
    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if (current.x, current.y) == goal:
            execution_time = time.time() - t0
            path, length, total_risk, turns = reconstruct_path(current, grid_map)
            flight_time = calculate_kinematic_flight_time(path, mass=30.0, max_thrust_net=120.0, v_max_kmh=65.0)

            return path, {
                "found": True, "time": execution_time, "length": length,
                "risk": total_risk, "turns": turns, "nodes": nodes_expanded,
                "flight_time": flight_time
            }

        # Aktualizujemy sprawdzenie odwiedzonych węzłów o wektor lotu
        state_key = (current.x, current.y, current.direction[0], current.direction[1])
        if state_key in visited:
            continue
        visited.add(state_key)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = current.x + dx, current.y + dy

            if grid_map.is_collision(nx, ny, drone_radius=drone_radius):
                continue

            cell_risk = grid_map.get_cost(nx, ny)
            static_risk_cost = cell_risk * risk_weight
            dist_cost = math.sqrt(dx ** 2 + dy ** 2)

            # --- MODEL FIZYCZNY: PRZYSPIESZANIE W LOCIE PROSTYM ---
            acceleration = 4.0
            v_max = 18.0

            # Pobieramy dystans pokonany po prostej od ostatniego manewru (lub startu)
            straight_dist = getattr(current, 'straight_dist', 0.0)

            # Dron przyspiesza (v^2 = v_0^2 + 2as). Zabezpieczamy przed przekroczeniem V_max
            node_speed = math.sqrt(current_speed ** 2 + 2 * acceleration * straight_dist)
            node_speed = min(v_max, node_speed)

            v1 = current.direction
            v2 = (dx, dy)
            straight_steps = getattr(current, 'straight_steps', 100)

            turn_cost = 0.0
            new_straight_steps = straight_steps + 1

            if v1 != (0, 0) and v1 != v2:
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                angle = math.acos(cos_theta)
                angle_deg = math.degrees(angle)

                # 1. TWARDA BARIERA KĄTA
                if node_speed > 5.0:
                    max_allowed_angle = 45.0
                else:
                    speed_factor = node_speed / 5.0
                    max_allowed_angle = 180.0 - (speed_factor * 135.0)

                if angle_deg > (max_allowed_angle + 1.0):
                    continue  # Kąt zbyt ostry dla tej prędkości!

                # 2. BLOKADA PĘDU (WYMUSZONY PROMIEŃ SKRĘTU)
                # Dron musi przelecieć prosto X metrów, zanim będzie mógł skręcić ponownie.
                # Przy 15 m/s potrzebuje aż 5 metrów stabilizacji. Przy 3 m/s wystarczy 1 metr.
                required_straight_steps = int(node_speed / 3.0)
                if straight_steps < required_straight_steps:
                    continue  # Zakaz skrętu! Dron jest w trakcie ustabilizowania po poprzednim manewrze!

                new_straight_steps = 0  # Skręt udany, resetujemy licznik lotu prosto
                turn_cost = turn_penalty * (angle / (math.pi / 2))

            new_g = current.cost + dist_cost + static_risk_cost + turn_cost
            neighbor_key = (nx, ny, dx, dy)

            if neighbor_key not in g_score or new_g < g_score[neighbor_key]:
                g_score[neighbor_key] = new_g

                # --- POPRAWKA HEURYSTYKI: Twardy limit zapobiegający efektowi "Greedy" ---
                # Używamy łagodniejszego skalowania i blokujemy mnożnik na maksymalnej wartości 2.5
                heuristic_multiplier = min(2.5, 1.0 + (risk_weight * 0.05))
                h = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2) * heuristic_multiplier

                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy), heuristic=h)

                # Przekazanie fizyki do kolejnego węzła
                if turn_cost > 0.0:
                    # Był zakręt! Dron musi zwolnić, by go wykonać, więc resetujemy dystans przyspieszania
                    neighbor.straight_dist = dist_cost
                else:
                    # Leci prosto dalej, kontynuujemy przyspieszanie
                    neighbor.straight_dist = straight_dist + dist_cost

                neighbor.straight_steps = new_straight_steps

                heapq.heappush(open_list, neighbor)

    return [], {"found": False, "time": 0, "length": 0, "risk": 0, "turns": 0, "nodes": nodes_expanded}
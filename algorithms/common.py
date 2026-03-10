from typing import Tuple, List, Optional, Callable, Dict, Any
import math
from environment.grid_map import GridMap


class Node:
    def __init__(self, x: int, y: int, cost: float, parent: Optional['Node'] = None,
                 direction: Tuple[int, int] = (0, 0), heuristic: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.direction = direction
        self.heuristic = heuristic

    @property
    def total_cost(self) -> float:
        return self.cost + self.heuristic

    def __lt__(self, other: 'Node') -> bool:
        return self.total_cost < other.total_cost


def reconstruct_path(node: Node, grid_map: GridMap) -> Tuple[List[Tuple[int, int]], float, float, int]:
    """Odtwarza ścieżkę i oblicza metryki."""
    path = []
    total_risk = 0.0
    total_length = 0.0
    turns = 0

    current = node
    last_dir = None

    while current:
        path.append((current.x, current.y))
        val = grid_map.get_cost(current.x, current.y)
        if val < 1.0:
            total_risk += val

        if current.parent:
            dx = current.x - current.parent.x
            dy = current.y - current.parent.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            total_length += dist

            curr_dir = (dx, dy)
            if last_dir is not None and curr_dir != last_dir:
                turns += 1
            last_dir = curr_dir

        current = current.parent

    return path[::-1], total_length, total_risk, turns


def calculate_segment_risk(path: List[Tuple[int, int]], env: GridMap) -> float:
    """Oblicza całkowite ryzyko na ścieżce."""
    total_risk = 0.0
    for (x, y) in path:
        total_risk += env.get_cost(x, y)
    return total_risk


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    """Oblicza długość ścieżki."""
    length = 0.0
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        length += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length


def generate_analysis_table(
        env: GridMap,
        start_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        search_func: Callable,
        base_len: float,
        base_risk: float,
        collision_radius: float,
        table_title: str = "ANALIZA"
) -> None:
    """
    Wspólna funkcja do generowania tabeli analizy dla różnych wag ryzyka.
    Używana zarówno w trybie offline jak i online.
    """
    risk_weights = [float(x) for x in range(0, 51, 5)]

    print("-" * 90)
    print(f"{table_title}")
    print("-" * 90)
    print(f"Baza: Dystans: {base_len:.2f} | Ryzyko: {base_risk:.2f}")
    print("-" * 90)
    print(f"{'Waga (W)':<10} | {'Dystans':<10} | {'Koszt [%]':<10} | {'Ryzyko':<10} | {'Zmiana Ryzyka [%]':<20}")
    print("-" * 90)

    for w in risk_weights:
        _, stats = search_func(env, start_pos, target_pos, risk_weight=w, turn_penalty=20.0, drone_radius=collision_radius)

        if stats['found'] and base_len > 0:
            len_inc = ((stats['length'] - base_len) / base_len) * 100

            risk_change = 0.0
            if base_risk > 0:
                risk_change = ((stats['risk'] - base_risk) / base_risk) * 100

            print(
                f"{w:<10.1f} | {stats['length']:<10.2f} | +{len_inc:<9.2f} | {stats['risk']:<10.2f} | {risk_change:<+19.2f}")
        else:
            print(f"{w:<10.1f} | BRAK TRASY")

    print("-" * 90)

def calculate_kinematic_flight_time(path: List[Tuple[int, int]],
                                    mass: float = 30.0,
                                    max_thrust_net: float = 120.0,
                                    v_max_kmh: float = 65.0) -> float:
    """
    Model kinematyczny BSP: Oblicza realistyczny czas przelotu trasy.
    Uwzględnia masę, przyspieszenie, V_max oraz konieczność hamowania przed zakrętami.
    """
    if len(path) < 2:
        return 0.0

    # 1. Konwersja jednostek i parametry fizyczne
    v_max = v_max_kmh / 3.6  # 65 km/h -> ~18.05 m/s
    a = max_thrust_net / mass  # np. 120N / 30kg = 4.0 m/s^2

    # 2. Wyodrębnienie prostych odcinków (zamiast analizować siatkę kratka po kratce)
    segments = []  # Długości prostych odcinków w metrach
    turn_angles = []  # Kąty między odcinkami w radianach

    current_len = 0.0
    last_dir = None

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        curr_dir = (dx, dy)

        if last_dir is not None and curr_dir != last_dir:
            # Obliczenie kąta zakrętu (Iloczyn skalarny)
            dot = last_dir[0] * curr_dir[0] + last_dir[1] * curr_dir[1]
            mag1 = math.sqrt(last_dir[0] ** 2 + last_dir[1] ** 2)
            mag2 = math.sqrt(curr_dir[0] ** 2 + curr_dir[1] ** 2)
            cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle = math.acos(cos_theta)

            segments.append(current_len)
            turn_angles.append(angle)
            current_len = dist
        else:
            current_len += dist

        last_dir = curr_dir

    segments.append(current_len)

    # 3. Modelowanie Prędkości w węzłach (zakrętach)
    # Dron musi zwolnić na zakręcie proporcjonalnie do jego ostrości
    turn_velocities = []
    # Maksymalne przyspieszenie boczne (dośrodkowe), jakie dron może wytrzymać
    # 7.0 m/s^2 to ok. 0.7g - bezpieczna wartość dla ciężkiego drona 30kg
    max_lat_a = 7.0

    for angle in turn_angles:
        # 1. Stary model (rzut wektora)
        v_vector = v_max * max(0.0, math.cos(angle))

        # 2. Model siły odśrodkowej
        # Przyjmujemy przybliżony promień skrętu r.
        # Na gridzie zakręt o kąt 'angle' przy 1m kratkach
        # wymusza rzędu r = 1.5 / sin(angle/2)
        r_approx = 1.5 / max(0.1, math.sin(angle / 2))
        v_centripetal = math.sqrt(max_lat_a * r_approx)

        # Prędkość na zakręcie to minimum z obu modeli
        v_turn = min(v_vector, v_centripetal)

        turn_velocities.append(max(0.5, v_turn))

    # Prędkość początkowa (V_0) = 0 i końcowa (V_k) = 0
    node_velocities = [0.0] + turn_velocities + [0.0]

    total_time = 0.0

    # 4. Obliczenia kinematyczne dla każdego odcinka z wykorzystaniem profilu trapezowego
    for i, L in enumerate(segments):
        v_start = node_velocities[i]
        v_end = node_velocities[i + 1]

        # Max prędkość możliwa do osiągnięcia z uwzględnieniem przyspieszania i hamowania
        # Wynika ze wzoru kinematycznego na dystans przy stałym przyspieszeniu
        v_reach = math.sqrt(max(0, a * L + (v_start ** 2 + v_end ** 2) / 2.0))

        if v_reach >= v_max:
            # Dron osiąga V_max (profil trapezowy)
            t_acc = (v_max - v_start) / a
            t_dec = (v_max - v_end) / a
            d_acc = (v_max ** 2 - v_start ** 2) / (2 * a)
            d_dec = (v_max ** 2 - v_end ** 2) / (2 * a)
            d_cruise = max(0, L - d_acc - d_dec)
            t_cruise = d_cruise / v_max
            total_time += (t_acc + t_cruise + t_dec)
        else:
            # Dron nie ma miejsca na osiągnięcie V_max (profil trójkątny)
            t_acc = abs(v_reach - v_start) / a
            t_dec = abs(v_reach - v_end) / a
            total_time += (t_acc + t_dec)

    return total_time


import heapq
import time
import math
from typing import List, Tuple, Dict, Any
from environment.grid_map import GridMap
from algorithms.common import Node, reconstruct_path, calculate_kinematic_flight_time


def base_search(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        risk_weight: float = 20.0,
        turn_penalty: float = 2.0,
        drone_radius: float = 3.0,
        initial_direction: Tuple[int, int] = (0, 0),
        current_speed: float = 0.0,
        use_heuristic: bool = True,
        use_kinematics: bool = False
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()
    start_node = Node(start[0], start[1], 0.0, direction=initial_direction)

    # Inicjalizacja parametrów fizycznych dla Risk A*
    if use_kinematics:
        start_node.straight_dist = 0.0
        start_node.straight_steps = 100

    open_list = []
    heapq.heappush(open_list, start_node)

    # Przestrzeń stanów zależna od algorytmu
    if use_kinematics:
        g_score = {(start[0], start[1], initial_direction[0], initial_direction[1]): 0.0}
    else:
        g_score = {(start[0], start[1]): 0.0}

    visited = set()
    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        # Cel osiągnięty
        if (current.x, current.y) == goal:
            execution_time = time.time() - t0
            path, length, total_risk, turns = reconstruct_path(current, grid_map)
            flight_time = calculate_kinematic_flight_time(path, mass=30.0, max_thrust_net=120.0, v_max_kmh=65.0)

            return path, {
                "found": True, "time": execution_time, "length": length,
                "risk": total_risk, "turns": turns, "nodes": nodes_expanded,
                "flight_time": flight_time
            }

        # Weryfikacja odwiedzonych węzłów
        state_key = (current.x, current.y, current.direction[0], current.direction[1]) if use_kinematics else (
            current.x, current.y)
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

            turn_cost = 0.0
            v1 = current.direction
            v2 = (dx, dy)

            # Zmienne kinematyczne
            new_straight_steps = 0
            straight_dist = getattr(current, 'straight_dist', 0.0)

            # ----- MODEL ZAAWANSOWANY (Risk A*) -----
            if use_kinematics:
                acceleration = 4.0
                v_max = 18.0
                node_speed = math.sqrt(current_speed ** 2 + 2 * acceleration * straight_dist)
                node_speed = min(v_max, node_speed)
                straight_steps = getattr(current, 'straight_steps', 100)
                new_straight_steps = straight_steps + 1

                if v1 != (0, 0) and v1 != v2:
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                    cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                    angle = math.acos(cos_theta)
                    angle_deg = math.degrees(angle)

                    if node_speed > 5.0:
                        max_allowed_angle = 45.0
                    else:
                        speed_factor = node_speed / 5.0
                        max_allowed_angle = 180.0 - (speed_factor * 135.0)

                    if angle_deg > (max_allowed_angle + 1.0) or straight_steps < int(node_speed / 3.0):
                        continue

                    new_straight_steps = 0
                    turn_cost = turn_penalty * (angle / (math.pi / 2))

            # ----- MODEL KLASYCZNY (Dijkstra / A*) -----
            else:
                if current.parent is not None:
                    if v1 != (0, 0) and v1 != v2:
                        turn_cost = turn_penalty

            new_g = current.cost + dist_cost + static_risk_cost + turn_cost
            neighbor_key = (nx, ny, dx, dy) if use_kinematics else (nx, ny)

            if neighbor_key not in g_score or new_g < g_score[neighbor_key]:
                g_score[neighbor_key] = new_g

                h = 0.0
                if use_heuristic:
                    # Różne mnożniki heurystyki (tie-breakery) wg Twojego kodu
                    multiplier = min(2.5, 1.0 + (risk_weight * 0.05)) if use_kinematics else 1.001
                    h = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2) * multiplier

                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy), heuristic=h)

                if use_kinematics:
                    neighbor.straight_dist = dist_cost if turn_cost > 0.0 else straight_dist + dist_cost
                    neighbor.straight_steps = new_straight_steps

                heapq.heappush(open_list, neighbor)

    return [], {"found": False, "time": 0, "length": 0, "risk": 0, "turns": 0, "nodes": nodes_expanded}
from typing import Tuple, List, Optional, Callable, Dict, Any
import math
import heapq
import time
from environment.grid_map import GridMap
from config import (
    V_MAX_MS, ACCELERATION, MAX_LATERAL_ACCEL, MIN_TURN_SPEED,
    DRONE_MASS_KG, MAX_THRUST_NET_N,
    HEURISTIC_MULT_ASTAR, HEURISTIC_MULT_RISK,
    RISK_WEIGHT, TURN_PENALTY, COLLISION_RADIUS,
    TURN_RADIUS_CONST
)


BRAKING_BUCKET_SIZE: float = 5.0   # [kratki] – dokładność dyskretyzacji drogi hamowania


def _braking_bucket(straight_dist: float) -> int:
    """Konwertuje ciągły straight_dist na dyskretny kubełek dla klucza stanu."""
    return min(int(straight_dist / BRAKING_BUCKET_SIZE), 20)


class Node:
    def __init__(self, x: int, y: int, cost: float, parent: Optional['Node'] = None,
                 direction: Tuple[int, int] = (0, 0), heuristic: float = 0.0,
                 speed: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.direction = direction
        self.heuristic = heuristic
        self.speed = speed

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


# ─────────────────────────────────────────────────────────────────────────────
# [FIX #21] Wspólna funkcja ryzyka — sumuje WSZYSTKIE komórki (bez [:-1])
# Używana spójnie przez offline i online.
# ─────────────────────────────────────────────────────────────────────────────
def calculate_segment_risk(path: List[Tuple[int, int]], env: GridMap) -> float:
    """Oblicza całkowite ryzyko na ścieżce (suma wartości grid dla każdej komórki)."""
    total_risk = 0.0
    for p in path:
        val = env.grid[int(p[0]), int(p[1])]
        if val < 1.0:
            total_risk += val
    return total_risk


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    """Oblicza długość ścieżki [kratki = metry przy CELL_SIZE_M=1]."""
    length = 0.0
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        length += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length


# ─────────────────────────────────────────────────────────────────────────────
# [FIX #2] Wspólna funkcja kary za zakręt (proporcjonalna do kąta).
# Używana przez WSZYSTKIE algorytmy.
# ─────────────────────────────────────────────────────────────────────────────
def compute_turn_cost(v1: Tuple[int, int], v2: Tuple[int, int],
                      turn_penalty: float) -> Tuple[float, float]:
    """
    Koszt zakrętu proporcjonalny do kąta: turn_cost = penalty × (angle / (π/2)).
    Zwraca (turn_cost, angle_rad). Brak zmiany kierunku → (0, 0).
    """
    if v1 == (0, 0) or v1 == v2:
        return 0.0, 0.0

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
    angle = math.acos(cos_theta)

    cost = turn_penalty * (angle / (math.pi / 2))
    return cost, angle


# ─────────────────────────────────────────────────────────────────────────────
# [FIX #18] Wspólna funkcja promienia zakrętu (z udokumentowaną stałą)
# ─────────────────────────────────────────────────────────────────────────────
def compute_turn_radius(angle: float) -> float:
    """
    Przybliżony promień zakrętu na siatce: r = TURN_RADIUS_CONST / sin(angle/2).
    Uzasadnienie stałej — patrz komentarz w config.py (TURN_RADIUS_CONST).
    """
    return TURN_RADIUS_CONST / max(0.1, math.sin(angle / 2))


def compute_safe_turn_speed(angle: float) -> float:
    """Bezpieczna prędkość w zakręcie z fizyki dośrodkowej: v = √(a_lat · r)."""
    r = compute_turn_radius(angle)
    v = math.sqrt(MAX_LATERAL_ACCEL * r)
    return max(MIN_TURN_SPEED, min(v, V_MAX_MS))


def generate_analysis_table(
        env: GridMap,
        start_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        search_func: Callable,
        base_len: float,
        base_risk: float,
        collision_radius: float,
        table_title: str = "ANALIZA",
        turn_penalty: float = TURN_PENALTY
) -> None:
    from config import PARETO_WEIGHT_MAX, PARETO_WEIGHT_STEP, RISK_WEIGHT
    risk_weights = sorted(set(
        [float(x) for x in range(0, PARETO_WEIGHT_MAX + 1, PARETO_WEIGHT_STEP)] + [RISK_WEIGHT]
    ))

    print("-" * 90)
    print(f"{table_title}")
    print("-" * 90)
    print(f"Baza: Dystans: {base_len:.2f} | Ryzyko: {base_risk:.2f}")
    print("-" * 90)
    print(f"{'Waga (W)':<10} | {'Dystans':<10} | {'Koszt [%]':<10} | {'Ryzyko':<10} | {'Zmiana Ryzyka [%]':<20}")
    print("-" * 90)

    for w in risk_weights:
        _, stats = search_func(env, start_pos, target_pos, risk_weight=w, turn_penalty=turn_penalty,
                               drone_radius=collision_radius)

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


# ─────────────────────────────────────────────────────────────────────────────
# [FIX #9, #19] Czas lotu z forward-backward pass na prędkościach węzłów.
# Eliminuje magiczną karę +2.0s i gwarantuje fizyczną spójność.
# ─────────────────────────────────────────────────────────────────────────────
def calculate_kinematic_flight_time(
        path: List[Tuple[int, int]],
        mass: float = DRONE_MASS_KG,
        max_thrust_net: float = MAX_THRUST_NET_N,
        v_max: float = V_MAX_MS
) -> float:
    if len(path) < 2:
        return 0.0

    a = max_thrust_net / mass

    segments: List[float] = []
    turn_angles: List[float] = []

    current_len = 0.0
    last_dir = None

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        curr_dir = (dx, dy)

        if last_dir is not None and curr_dir != last_dir:
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

    # Prędkości w węzłach (zakrętach): fizyka dośrodkowa
    turn_velocities = []
    for angle in turn_angles:
        turn_velocities.append(compute_safe_turn_speed(angle))

    # Prędkości: [start=0] + [zakręty] + [stop=0]
    node_velocities = [0.0] + turn_velocities + [0.0]

    # Forward pass: nie możemy przyspieszać szybciej niż fizyka pozwala
    for i in range(1, len(node_velocities)):
        seg_idx = min(i - 1, len(segments) - 1)
        seg_len = segments[seg_idx]
        v_reachable = math.sqrt(max(0.0, node_velocities[i - 1] ** 2 + 2 * a * seg_len))
        node_velocities[i] = min(node_velocities[i], v_reachable, v_max)

    # Backward pass: musimy zdążyć wyhamować do prędkości następnego węzła
    for i in range(len(node_velocities) - 2, -1, -1):
        seg_idx = min(i, len(segments) - 1)
        seg_len = segments[seg_idx]
        v_reachable = math.sqrt(max(0.0, node_velocities[i + 1] ** 2 + 2 * a * seg_len))
        node_velocities[i] = min(node_velocities[i], v_reachable)

    total_time = 0.0
    for i, L in enumerate(segments):
        if L <= 0:
            continue

        v_start = node_velocities[i]
        v_end = node_velocities[i + 1]

        # Po forward-backward pass, fizyczne ograniczenia są gwarantowane.
        # Obliczamy szczytową prędkość na segmencie.
        v_peak_sq = (2 * a * L + v_start ** 2 + v_end ** 2) / 2.0
        v_peak = min(math.sqrt(max(0.0, v_peak_sq)), v_max)

        if v_peak >= v_max:
            # Segment z fazą cruise
            d_acc = max(0.0, (v_max ** 2 - v_start ** 2) / (2 * a))
            d_dec = max(0.0, (v_max ** 2 - v_end ** 2) / (2 * a))
            d_cruise = max(0.0, L - d_acc - d_dec)
            t_acc = (v_max - v_start) / a if v_max > v_start else 0.0
            t_dec = (v_max - v_end) / a if v_max > v_end else 0.0
            t_cruise = d_cruise / v_max if v_max > 0 else 0.0
            total_time += t_acc + t_cruise + t_dec
        else:
            # Trójkąt prędkości: rozpędzanie do v_peak, potem hamowanie
            t_acc = abs(v_peak - v_start) / a if a > 0 else 0.0
            t_dec = abs(v_peak - v_end) / a if a > 0 else 0.0
            total_time += t_acc + t_dec

    return total_time


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNA FUNKCJA PRZESZUKIWANIA
# [FIX #2, #5]  Ujednolicona proporcjonalna formuła kary dla wszystkich algo
# [FIX #16]     braking_penalty w jednostkach dystansu (nie czasu)
# [FIX #22]     Zwraca rzeczywisty czas obliczeń przy braku trasy
# ─────────────────────────────────────────────────────────────────────────────
def base_search(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        risk_weight: float = RISK_WEIGHT,
        turn_penalty: float = TURN_PENALTY,
        drone_radius: float = COLLISION_RADIUS,
        initial_direction: Tuple[int, int] = (0, 0),
        current_speed: float = 0.0,
        use_heuristic: bool = True,
        use_kinematics: bool = False,
        initial_straight_dist: float = 0.0
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()
    start_node = Node(start[0], start[1], 0.0, direction=initial_direction, speed=current_speed)

    if use_kinematics:
        start_node.straight_dist = initial_straight_dist

    open_list = []
    heapq.heappush(open_list, start_node)

    if use_kinematics:
        init_bucket = _braking_bucket(initial_straight_dist)
        g_score: dict = {
            (start[0], start[1], initial_direction[0], initial_direction[1], init_bucket): 0.0
        }
    else:
        g_score = {(start[0], start[1]): 0.0}

    visited: set = set()
    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if (current.x, current.y) == goal:
            execution_time = time.time() - t0
            path, length, total_risk, turns = reconstruct_path(current, grid_map)
            flight_time = calculate_kinematic_flight_time(path)

            return path, {
                "found": True, "time": execution_time, "length": length,
                "risk": total_risk, "turns": turns, "nodes": nodes_expanded,
                "flight_time": flight_time
            }

        if use_kinematics:
            sd_current = getattr(current, 'straight_dist', 0.0)
            state_key = (
                current.x, current.y,
                current.direction[0], current.direction[1],
                _braking_bucket(sd_current)
            )
        else:
            state_key = (current.x, current.y)

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

            straight_dist = getattr(current, 'straight_dist', 0.0)
            new_speed = current.speed

            # ── MODEL ZAAWANSOWANY (Risk-Aware A*) ────────────────────────
            if use_kinematics:
                node_speed = current.speed

                new_speed = min(V_MAX_MS, math.sqrt(node_speed ** 2 + 2 * ACCELERATION * dist_cost))
                new_straight_dist = straight_dist + dist_cost

                if v1 != (0, 0) and v1 != v2:
                    # [FIX #2] Kąt obliczany wspólną funkcją
                    base_turn_cost, angle = compute_turn_cost(v1, v2, turn_penalty)

                    # HARD LIMIT: zawrócenie (~180°) — fizycznie niemożliwe
                    if angle >= math.radians(170):
                        continue

                    # [FIX #18] Promień z udokumentowanej funkcji
                    v_safe_turn = compute_safe_turn_speed(angle)

                    # [FIX #16] braking_penalty wyrażone w DYSTANSIE [kratki=metry],
                    # spójne z dist_cost i turn_cost.
                    # Fizyczny sens: dodatkowa droga „utracona" na hamowanie.
                    braking_penalty = 0.0
                    if node_speed > v_safe_turn:
                        available_braking_dist = straight_dist + dist_cost
                        braking_dist_needed = (node_speed ** 2 - v_safe_turn ** 2) / (2 * ACCELERATION)
                        if braking_dist_needed > available_braking_dist:
                            continue  # Fizycznie niemożliwe

                        braking_penalty = braking_dist_needed  # [FIX #16] dystans, nie czas

                    new_speed = v_safe_turn
                    turn_cost = base_turn_cost + braking_penalty
                    new_straight_dist = 0.0

            # ── MODEL KLASYCZNY (Dijkstra / A* Standard) ──────────────────
            # [FIX #2, #5] Ta sama formuła proporcjonalna do kąta
            else:
                new_straight_dist = 0.0
                if current.parent is not None:
                    base_turn_cost, angle = compute_turn_cost(v1, v2, turn_penalty)
                    turn_cost = base_turn_cost

            new_g = current.cost + dist_cost + static_risk_cost + turn_cost

            if use_kinematics:
                neighbor_key = (nx, ny, dx, dy, _braking_bucket(new_straight_dist))
            else:
                neighbor_key = (nx, ny)

            if neighbor_key not in g_score or new_g < g_score[neighbor_key]:
                g_score[neighbor_key] = new_g

                h = 0.0
                if use_heuristic:
                    multiplier = HEURISTIC_MULT_RISK if use_kinematics else HEURISTIC_MULT_ASTAR
                    h = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2) * multiplier

                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy), heuristic=h,
                                speed=new_speed)

                if use_kinematics:
                    neighbor.straight_dist = new_straight_dist

                heapq.heappush(open_list, neighbor)

    execution_time = time.time() - t0
    return [], {
        "found": False, "time": execution_time, "length": 0, "risk": 0,
        "turns": 0, "nodes": nodes_expanded, "flight_time": 0
    }
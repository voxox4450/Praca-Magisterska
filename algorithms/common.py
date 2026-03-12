from typing import Tuple, List, Optional, Callable, Dict, Any
import math
import heapq
import time
from environment.grid_map import GridMap
from config import (
    V_MAX_MS, ACCELERATION, MAX_LATERAL_ACCEL, MIN_TURN_SPEED,
    DRONE_MASS_KG, MAX_THRUST_NET_N,
    HEURISTIC_MULT_ASTAR, HEURISTIC_MULT_RISK,
    RISK_WEIGHT, TURN_PENALTY_CLASSIC, TURN_PENALTY_RISK, COLLISION_RADIUS
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
        self.speed = speed          # Prędkość drona w tym węźle [m/s]

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
    for p in path:
        val = env.grid[int(p[0]), int(p[1])]
        if val < 1.0:
            total_risk += val
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
        table_title: str = "ANALIZA",
        turn_penalty: float = TURN_PENALTY_CLASSIC
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

    turn_velocities = []
    for angle in turn_angles:
        r_approx = 1.5 / max(0.1, math.sin(angle / 2))
        v_centripetal = math.sqrt(MAX_LATERAL_ACCEL * r_approx)
        v_turn = min(v_centripetal, v_max)
        turn_velocities.append(max(MIN_TURN_SPEED, v_turn))

    node_velocities = [0.0] + turn_velocities + [0.0]
    total_time = 0.0

    for i, L in enumerate(segments):
        v_start = node_velocities[i]
        v_end = node_velocities[i + 1]

        min_breaking_dist = abs(v_start ** 2 - v_end ** 2) / (2 * a)
        if min_breaking_dist > L:
            total_time += (L / max(0.1, (v_start + v_end) / 2)) + 2.0
            continue

        v_reach_squared = a * L + (v_start ** 2 + v_end ** 2) / 2.0
        v_reach = math.sqrt(v_reach_squared)

        if v_reach >= v_max:
            t_acc = (v_max - v_start) / a
            t_dec = (v_max - v_end) / a
            d_acc = (v_max ** 2 - v_start ** 2) / (2 * a)
            d_dec = (v_max ** 2 - v_end ** 2) / (2 * a)
            d_cruise = max(0, L - d_acc - d_dec)
            t_cruise = d_cruise / v_max
            total_time += (t_acc + t_cruise + t_dec)
        else:
            t_acc = abs(v_reach - v_start) / a
            t_dec = abs(v_reach - v_end) / a
            total_time += (t_acc + t_dec)

    return total_time


def base_search(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        risk_weight: float = RISK_WEIGHT,
        turn_penalty: float = TURN_PENALTY_CLASSIC,
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
        # straight_dist = łączna długość prostego odcinka kończącego się w tym węźle
        # (od ostatniego zakrętu lub startu do bieżącego węzła)
        start_node.straight_dist = initial_straight_dist

    open_list = []
    heapq.heappush(open_list, start_node)

    # ─────────────────────────────────────────────────────────────────────────
    # POPRAWKA #1 (kontynuacja): Spójny 5-elementowy klucz dla obu struktur.
    #   Oryginał: g_score używał 5-elementowego klucza przy inicjalizacji,
    #             ale 4-elementowego (bez straight_steps) przy sprawdzaniu
    #             sąsiadów → lookup zawsze zwracał "klucz nie istnieje"
    #             → każda ścieżka przez (x,y,dx,dy) była akceptowana.
    # ─────────────────────────────────────────────────────────────────────────
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

        # Klucz odwiedzin – dla kinematyki uwzględniamy kubełek straight_dist,
        # bo ten sam (x,y,dir) z większym straight_dist otwiera inne możliwości hamowania
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

            # ── MODEL ZAAWANSOWANY (Risk A*) ──────────────────────────────
            if use_kinematics:
                node_speed = current.speed

                # Domyślnie: lot na wprost – przyspieszenie do V_max
                new_speed = min(V_MAX_MS, math.sqrt(node_speed ** 2 + 2 * ACCELERATION * dist_cost))
                # straight_dist sąsiada = akumulacja jeśli idziemy prosto
                new_straight_dist = straight_dist + dist_cost

                if v1 != (0, 0) and v1 != v2:
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                    cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                    angle = math.acos(cos_theta)

                    # HARD LIMIT: zawrócenie w locie (~180°) – fizycznie niemożliwe
                    if angle >= math.radians(170):
                        continue

                    # Prędkość bezpieczna dla tego zakrętu (fizyka dośrodkowa)
                    r_turn = 1.5 / max(0.1, math.sin(angle / 2))
                    v_safe_turn = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_turn))
                    v_safe_turn = min(v_safe_turn, V_MAX_MS)

                    braking_penalty = 0.0
                    if node_speed > v_safe_turn:
                        available_braking_dist = straight_dist + dist_cost
                        braking_dist_needed = (node_speed ** 2 - v_safe_turn ** 2) / (2 * ACCELERATION)
                        if braking_dist_needed > available_braking_dist:
                            continue  # Fizycznie niemożliwe – odrzuć tę krawędź

                        braking_penalty = (node_speed - v_safe_turn) / ACCELERATION

                    # Po zakręcie: nowa prędkość = v_safe_turn
                    new_speed = v_safe_turn
                    turn_cost = turn_penalty * (angle / (math.pi / 2)) + braking_penalty

                    new_straight_dist = 0.0

            # ── MODEL KLASYCZNY (Dijkstra / A* Standard) ──────────────────
            else:
                new_straight_dist = 0.0   # nieużywane, ale dla spójności
                if current.parent is not None:
                    if v1 != (0, 0) and v1 != v2:
                        turn_cost = turn_penalty

            new_g = current.cost + dist_cost + static_risk_cost + turn_cost

            # Spójny klucz g_score – 5-elementowy dla kinematyki
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

    return [], {"found": False, "time": 0, "length": 0, "risk": 0, "turns": 0, "nodes": nodes_expanded}
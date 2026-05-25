from typing import Tuple, List, Optional, Dict, Any
import math
import heapq
import time
from environment.grid_map import GridMap
from config import (
    V_MAX_MS, ACCELERATION, MAX_LATERAL_ACCEL, MIN_TURN_SPEED,
    DRONE_MASS_KG, MAX_THRUST_NET_N,
    HEURISTIC_MULTIPLIER,
    RISK_WEIGHT, TURN_PENALTY, COLLISION_RADIUS,
    TURN_RADIUS_CONST, SAFE_MARGIN_M
)


BRAKING_BUCKET_SIZE: float = 3.0   # [kratki] – dokładność dyskretyzacji drogi hamowania (mniejsza = więcej stanów, dokładniej)
_RAD_170: float = math.radians(170)  # [OPT] Prekomputowany próg zawracania


def drone_radius_for_mass(mass: float) -> float:
    """
    Promień fizyczny drona: r = r_ref × √(m / m_ref).
    Skalowanie izometryczne — zakłada że większy dron (większa masa)
    wymaga proporcjonalnie większej konstrukcji (ramion, śmigieł),
    przy zachowaniu podobieństwa geometrycznego. Promień rośnie jak
    pierwiastek masy, bo masa ∝ objętość ∝ r³, ale w praktyce drony
    wielowirnikowe skalują się bliżej r² (płaska konstrukcja).
    Przyjęto √ jako kompromis między skalowaniem 2D i 3D.
    Referencja: m_ref=30 kg → r_ref=1.0 m (typowy dron dostawczy).
    """
    return 1.0 * math.sqrt(mass / 30.0)


def collision_radius_for_mass(mass: float) -> float:
    """Promień kolizji = promień fizyczny + margines bezpieczeństwa."""
    return drone_radius_for_mass(mass) + SAFE_MARGIN_M


def _braking_bucket(straight_dist: float) -> int:
    """Konwertuje ciągły straight_dist na dyskretny kubełek dla klucza stanu."""
    return min(int(straight_dist / BRAKING_BUCKET_SIZE), 30)


class Node:
    __slots__ = ('x', 'y', 'cost', 'parent', 'direction', 'heuristic', 'speed', 'straight_dist')

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
        self.straight_dist = 0.0

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
    """Oblicza całkowite ryzyko na ścieżce (uwzględniając powierzchnię drona)."""
    total_risk = 0.0
    for p in path:
        val = env.risk_grid[int(p[0]), int(p[1])] # ZMIANA na risk_grid
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
# [FIX #29] Dekompozycja kosztów ścieżki — diagnostyka post-hoc.
# Oblicza DOKŁADNIE te same składniki, co base_search() na ścieżce wynikowej.
# Pozwala zweryfikować: g_cost ≈ cost_dist + cost_risk + cost_turn
# ─────────────────────────────────────────────────────────────────────────────
def decompose_path_costs(
        path: List[Tuple[int, int]],
        grid_map: GridMap,
        risk_weight: float,
        turn_penalty: float
) -> Dict[str, float]:
    """
    Rozkłada koszt ścieżki na 3 składniki: dystans, ryzyko, zakręty.
    Zwraca dict z kluczami 'dist', 'risk', 'turn'.
    """
    cost_dist = 0.0
    cost_risk = 0.0
    cost_turn = 0.0
    last_dir = None

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        curr_dir = (dx, dy)

        dist = math.sqrt(dx ** 2 + dy ** 2)
        cost_dist += dist

        cell_risk = grid_map.get_cost(path[i][0], path[i][1])
        if cell_risk < 1.0:
            cost_risk += cell_risk * risk_weight

        if last_dir is not None and last_dir != (0, 0) and last_dir != curr_dir:
            tc, _ = compute_turn_cost(last_dir, curr_dir, turn_penalty)
            cost_turn += tc

        last_dir = curr_dir

    return {'dist': cost_dist, 'risk': cost_risk, 'turn': cost_turn}


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNA FUNKCJA PRZESZUKIWANIA
# [FIX #2, #5]  Ujednolicona proporcjonalna formuła kary dla wszystkich algo
# [FIX #32]     Identyczny format klucza stanu (x,y,dx,dy,bucket)
# [OPT]         collision_mask, bezpośredni dostęp do grid, prekomp. dystanse
# ─────────────────────────────────────────────────────────────────────────────

# [OPT] Prekomputowane dystanse dla 8 kierunków — unika math.sqrt w pętli
_SQRT2 = math.sqrt(2)
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


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
        initial_straight_dist: float = 0.0,
        drone_mass: float = DRONE_MASS_KG
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    t0 = time.time()

    # Przyspieszenie zależne od masy (cięższy dron → wolniej hamuje)
    accel = MAX_THRUST_NET_N / drone_mass

    start_node = Node(start[0], start[1], 0.0, direction=initial_direction, speed=current_speed)

    # [FIX #32] WSZYSTKIE algorytmy śledzą straight_dist → identyczny format klucza.
    start_node.straight_dist = initial_straight_dist

    open_list = []
    heapq.heappush(open_list, start_node)

    # [FIX #32] Format klucza: (x, y, dx, dy, bucket) dla WSZYSTKICH algorytmów.
    # [OPT] Dijkstra/A* nie korzystają z bucket do kosztów, więc bucket=0 zawsze.
    # Risk-Aware A* używa rzeczywistego kubełka.
    # Format identyczny → ta sama struktura grafu, ale Dijkstra/A* nie marnują
    # czasu na eksplorację identycznych stanów z różnymi kubełkami.
    init_bucket = _braking_bucket(initial_straight_dist) if use_kinematics else 0
    g_score: dict = {
        (start[0], start[1], initial_direction[0], initial_direction[1], init_bucket): 0.0
    }

    visited: set = set()
    nodes_expanded = 0

    # [OPT] Lokalne referencje — unika wielokrotnego rozwiązywania atrybutów
    collision_mask = grid_map.collision_mask
    risk_grid = grid_map.risk_grid
    goal_x, goal_y = goal
    map_w, map_h = grid_map.width, grid_map.height

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if current.x == goal_x and current.y == goal_y:
            execution_time = time.time() - t0
            path, length, total_risk, turns = reconstruct_path(current, grid_map)
            flight_time = calculate_kinematic_flight_time(path, mass=drone_mass)

            decomposed = decompose_path_costs(path, grid_map, risk_weight, turn_penalty)

            return path, {
                "found": True, "time": execution_time, "length": length,
                "risk": total_risk, "turns": turns, "nodes": nodes_expanded,
                "flight_time": flight_time,
                "g_cost": current.cost,
                "cost_dist": decomposed['dist'],
                "cost_risk": decomposed['risk'],
                "cost_turn": decomposed['turn'],
            }

        # [FIX #32] Ujednolicony format klucza stanu
        sd_current = current.straight_dist
        if use_kinematics:
            #bucket = 0
            bucket = _braking_bucket(sd_current)
        else:
            # bucket = _braking_bucket(sd_current)
            bucket = 0

        state_key = (current.x, current.y,
                     current.direction[0], current.direction[1], bucket)

        if state_key in visited:
            continue
        visited.add(state_key)

        cx, cy = current.x, current.y
        v1 = current.direction
        cur_cost = current.cost
        cur_speed = current.speed
        straight_dist = sd_current

        # [OPT] Prekomputowane dystanse, collision_mask zamiast is_collision()
        for dx, dy, dist_cost in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy

            # [FIX] Sprawdzenie granic mapy — zapobiega IndexError przy krawędziach
            if not (0 <= nx < map_w and 0 <= ny < map_h):
                continue

            # [OPT] Bezpośredni odczyt z numpy bool array — bez wywołania funkcji
            if collision_mask[nx, ny]:
                continue

            # [OPT] Bezpośredni dostęp do tablicy zamiast get_cost()
            cell_risk = risk_grid[nx, ny]
            static_risk_cost = cell_risk * risk_weight

            turn_cost = 0.0
            v2 = (dx, dy)
            new_speed = cur_speed

            # ── WSPÓLNE OGRANICZENIA GEOMETRYCZNE PLATFORMY BSP ───────────
            # Dron jako fizyczna maszyna nie potrafi zawrócić w punkcie —
            # wymaga to łuku o niezerowym promieniu. To cecha PLATFORMY,
            # nie algorytmu. Egzekwowane jednolicie dla wszystkich trzech
            # systemów planowania, niezależnie od posiadania modelu fizyki.
            is_turn = (v1 != (0, 0) and v1 != v2)
            base_turn_cost = 0.0
            angle = 0.0
            if is_turn:
                base_turn_cost, angle = compute_turn_cost(v1, v2, turn_penalty)
                if angle >= _RAD_170:
                    continue

            # ── MODEL ZAAWANSOWANY (Risk-Aware A* / A*-KIN) ───────────────
            if use_kinematics:
                node_speed = cur_speed

                new_speed = min(V_MAX_MS, math.sqrt(node_speed * node_speed + 2.0 * accel * dist_cost))
                new_straight_dist = straight_dist + dist_cost

                if is_turn:
                    v_safe_turn = compute_safe_turn_speed(angle)

                    # Twarde odrzucenie manewrów fizycznie niewykonalnych:
                    # jeśli dron leci za szybko i nie zdąży wytracić prędkości
                    # do bezpiecznego poziomu na dostępnym odcinku prostym,
                    # zakręt nie istnieje jako legalna opcja.
                    if node_speed > v_safe_turn:
                        available_braking_dist = straight_dist + dist_cost
                        braking_dist_needed = (node_speed * node_speed - v_safe_turn * v_safe_turn) / (2.0 * accel)
                        if braking_dist_needed > available_braking_dist:
                            continue

                    new_speed = v_safe_turn
                    turn_cost = base_turn_cost
                    new_straight_dist = 0.0

                new_bucket = _braking_bucket(new_straight_dist)
                #new_bucket = 0

            # ── MODEL KLASYCZNY (Dijkstra / A* Standard) ──────────────────
            else:
                new_straight_dist = straight_dist + dist_cost
                if is_turn:
                    turn_cost = base_turn_cost
                    new_straight_dist = 0.0

                # [OPT] Bucket zawsze 0 — ten sam format klucza, brak duplikatów
                new_bucket = 0
                #new_bucket = _braking_bucket(new_straight_dist)

            new_g = cur_cost + dist_cost + static_risk_cost + turn_cost

            neighbor_key = (nx, ny, dx, dy, new_bucket)

            if neighbor_key not in g_score or new_g < g_score[neighbor_key]:
                g_score[neighbor_key] = new_g

                h = 0.0
                if use_heuristic:
                    # Identyczna heurystyka dla A* Standard i Risk-Aware A* —
                    # różnica między nimi to model fizyki, nie agresywność
                    # przeszukiwania.
                    h = math.sqrt((nx - goal_x) ** 2 + (ny - goal_y) ** 2) * HEURISTIC_MULTIPLIER

                neighbor = Node(nx, ny, new_g, current, direction=v2, heuristic=h,
                                speed=new_speed)

                neighbor.straight_dist = new_straight_dist

                heapq.heappush(open_list, neighbor)

    execution_time = time.time() - t0
    return [], {
        "found": False, "time": execution_time, "length": 0, "risk": 0,
        "turns": 0, "nodes": nodes_expanded, "flight_time": 0,
        "g_cost": 0, "cost_dist": 0, "cost_risk": 0, "cost_turn": 0
    }
def sensor_range_for_mass(mass: float) -> float:
    """
    Zasięg sensora [m] rośnie liniowo wraz z udźwigiem (lepszy sprzęt),
    ale jest nasycany na wartości 60.0 m ze względu na typowe ograniczenia
    widoczności w terenie zurbanizowanym (tzw. miejskie kaniony).
    - 1 kg  -> 16.5 m
    - 15 kg -> 37.5 m
    - >= 30 kg -> 60.0 m
    """
    return min(60.0, 15.0 + 1.5 * mass)

def processing_delay_for_mass(mass: float) -> float:
    """
    Stały czas reakcji [s] = 0.80 s.
    Obejmuje czas przetwarzania sensorycznego, replanowania trasy
    oraz inicjacji manewru unikowego. Przyjęto wartość stałą niezależną
    od masy — uzasadnienie: czas obliczeń zależy od procesora pokładowego
    (identyczny dla wszystkich wariantów drona w badaniu), a bezwładność
    rotacyjna jest kompensowana przez proporcjonalnie większy moment obrotowy
    silników w cięższych maszynach.
    """
    return 0.80
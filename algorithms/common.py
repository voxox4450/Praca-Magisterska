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
    for angle in turn_angles:
        # Fizyczny rzut wektora prędkości.
        # 0 st = V_max (cos(0)=1)
        # 90 st = 0 m/s (cos(90)=0) - dron musi wyhamować, żeby skręcić pod kątem prostym!
        v_turn = v_max * max(0.0, math.cos(angle))

        # Dajemy minimalną prędkość 0.5 m/s na "obrócenie się" w miejscu
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
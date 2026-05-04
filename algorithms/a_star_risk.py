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
    """
    Planowanie bufora hamowania awaryjnego — integralna część logiki
    Risk-Aware A*. Algorytm "wie", że pierwszy zakręt nowej trasy będzie
    wymagał prędkości v_safe < current_speed, i z wyprzedzeniem rezerwuje
    odcinek prostoliniowy na wytracenie nadmiaru prędkości.

    Bez modelu kinematycznego ten krok nie ma żadnego sensu — wymaga
    znajomości aktualnej prędkości, dostępnego przyspieszenia (a = F/m)
    oraz docelowej prędkości w zakręcie wynikającej z fizyki dośrodkowej.

    Zwraca:
        buffer_points — lista punktów składających się na bufor
        buffer_dist   — całkowita długość bufora [m]
        v_after       — prędkość drona po zakończeniu hamowania [m/s]
    """
    if heading == (0, 0) or current_speed <= 0:
        return [], 0.0, current_speed

    accel = MAX_THRUST_NET_N / drone_mass

    # Bezpieczna prędkość wejścia w zakręt 45° na siatce 8-kierunkowej
    # (najczęstszy scenariusz pierwszego zakrętu po replanowaniu)
    r_45 = TURN_RADIUS_CONST / max(0.1, math.sin(math.radians(22.5)))
    v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_45))
    v_safe_ref = min(v_safe_ref, V_MAX_MS)

    if current_speed <= v_safe_ref:
        return [], 0.0, current_speed

    # Wymagana droga hamowania z fizyki: d = (v² - v_safe²) / (2a)
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
    """
    Risk-Aware A* z pełnym modelem kinematycznym drona.
    Ta sama formuła kary za zakręt co Dijkstra/A* (proporcjonalna do kąta).
    DODATKOWO względem A* Standard:
    - planowanie bufora hamowania awaryjnego, gdy dron startuje z niezerową
      prędkością (current_speed > v_safe) — algorytm rezerwuje odcinek
      prostoliniowy na wytracenie prędkości, ZANIM zacznie przeszukiwać graf,
    - ograniczenie prędkości w zakrętach (fizyka dośrodkowa),
    - kara za dystans hamowania w funkcji kosztu,
    - odrzucenie zakrętów fizycznie niemożliwych do wyhamowania.
    Parametr drone_mass wpływa na przyspieszenie (a = F/m).

    Bufor hamowania jest WEWNĘTRZNĄ częścią tego algorytmu — nie jest
    obliczany w warstwie symulacji ani doczepiany z zewnątrz. Decyzja
    o jego długości i pozycji wynika wprost z modelu kinematycznego, do
    którego klasyczne algorytmy (Dijkstra, A*) nie mają dostępu.
    """
    # ── Faza 1: Planowanie bufora hamowania (logika kinematyczna) ──────
    buffer_points, buffer_dist, v_after_buffer = _plan_braking_buffer(
        grid_map, start, initial_direction, current_speed, drone_mass
    )

    # Punkt rozpoczęcia przeszukiwania grafu — po zakończeniu bufora
    search_start = buffer_points[-1] if buffer_points else start
    search_start_int = (int(search_start[0]), int(search_start[1]))

    # ── Faza 2: Przeszukiwanie grafu od końca bufora do celu ───────────
    path, stats = base_search(
        grid_map, search_start_int, goal,
        risk_weight, turn_penalty, drone_radius,
        initial_direction, v_after_buffer,
        use_heuristic=True,
        use_kinematics=True,
        initial_straight_dist=initial_straight_dist + buffer_dist,
        drone_mass=drone_mass
    )

    # ── Faza 3: Złożenie pełnej trasy: [start] + bufor + ścieżka ───────
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
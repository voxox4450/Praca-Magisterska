"""
metrics_terminal.py — Porównawczy raport hamowania w trybie dynamicznym.
"""

import math
from typing import List, Tuple, Dict, Any

from algorithms.common import compute_safe_turn_speed
from config import MAX_THRUST_NET_N


# Najgorszy zakręt na siatce 8-kierunkowej: 135° (3π/4 rad).
WORST_TURN_ANGLE_RAD = math.radians(135)


def _path_length(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        total += math.hypot(path[i][0] - path[i - 1][0],
                            path[i][1] - path[i - 1][1])
    return total


def _first_turn_on_path(path: List[Tuple[int, int]]) -> Tuple[float, int]:
    """
    Znajduje pierwszy zakręt na ścieżce.
    """
    if len(path) < 3:
        return 0.0, -1

    last_dir = None
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        curr_dir = (dx, dy)
        if last_dir is not None and curr_dir != last_dir:
            dot = last_dir[0] * curr_dir[0] + last_dir[1] * curr_dir[1]
            m1 = math.hypot(*last_dir)
            m2 = math.hypot(*curr_dir)
            cos_t = max(-1.0, min(1.0, dot / (m1 * m2)))
            angle = math.acos(cos_t)
            return angle, i - 1
        last_dir = curr_dir

    return 0.0, -1


def _first_turn_with_entry(
        replan_path: List[Tuple[int, int]],
        approach_heading: Tuple[int, int] = (0, 0),
) -> Tuple[float, float]:
    if len(replan_path) < 2:
        return 0.0, 0.0

    first_dir = (replan_path[1][0] - replan_path[0][0],
                 replan_path[1][1] - replan_path[0][1])

    if approach_heading != (0, 0) and first_dir != (0, 0) and first_dir != approach_heading:
        dot = approach_heading[0] * first_dir[0] + approach_heading[1] * first_dir[1]
        m1 = math.hypot(*approach_heading)
        m2 = math.hypot(*first_dir)
        if m1 * m2 > 0:
            cos_t = max(-1.0, min(1.0, dot / (m1 * m2)))
            entry_angle = math.acos(cos_t)
            if entry_angle > 0.01:
                return entry_angle, 0.0

    angle, turn_idx = _first_turn_on_path(replan_path)
    if turn_idx == -1:
        return 0.0, _path_length(replan_path)
    return angle, _path_length(replan_path[:turn_idx + 1])


def analyze_braking_scenario(
        scenario_result: Dict[str, Any],
        mass: float,
) -> Dict[str, Any]:
    """
    Analizuje scenariusz dynamiczny pod kątem hamowania (formuła jednofazowa).
    """
    mode = scenario_result.get("mode", "UNKNOWN")

    if mode in ("NO_PATH", "NO_SENSOR", "TRAPPED", "CLEAR"):
        return {
            "valid": False,
            "message": f"Brak replanowania (tryb: {mode})",
        }

    if mode == "CRASH":
        v_detect = scenario_result.get("v_detect", 0.0)
        return {
            "valid": True,
            "v_detect": v_detect,
            "d_reaction": 0.0, "d_buffer": 0.0,
            "d_straight_post_buffer": 0.0,
            "d_available": 0.0,
            "v_safe_uniform": 0.0, "d_required": 0.0,
            "first_turn_angle": 0.0,
            "feasible": False, "deficit": float("inf"),
            "message": "KOLIZJA W FAZIE REAKCJI",
        }

    a = MAX_THRUST_NET_N / mass

    v_detect = scenario_result.get("v_detect", 0.0)
    v_react_end = scenario_result.get("v_react_end", v_detect)
    detect_idx = scenario_result.get("detect_idx", -1)
    react_idx = scenario_result.get("react_idx", -1)
    clean_path = scenario_result.get("clean_path", [])
    buffer_dist = scenario_result.get("buffer_dist", 0.0)
    replan_path = scenario_result.get("replan_path", [])
    heading = scenario_result.get("heading", (0, 0))

    if detect_idx >= 0 and react_idx > detect_idx and clean_path:
        d_reaction = _path_length(clean_path[detect_idx:react_idx + 1])
    else:
        d_reaction = 0.0

    first_angle, d_to_turn_raw = _first_turn_with_entry(
        replan_path, approach_heading=heading
    )

    d_straight_post_buffer = max(0.0, d_to_turn_raw - buffer_dist)

    v_safe_uniform = compute_safe_turn_speed(WORST_TURN_ANGLE_RAD)

    if v_detect > v_safe_uniform:
        d_required = (v_detect ** 2 - v_safe_uniform ** 2) / (2.0 * a)
    else:
        d_required = 0.0

    d_available = d_reaction + buffer_dist + d_straight_post_buffer

    deficit = max(0.0, d_required - d_available)
    feasible = (deficit < 0.01)

    if first_angle > 0:
        msg_turn = f"zakręt {math.degrees(first_angle):.0f}°"
    else:
        msg_turn = "brak zakrętu"

    return {
        "valid": True,
        "v_detect": v_detect,
        "v_react_end": v_react_end,
        "d_reaction": d_reaction,
        "d_buffer": buffer_dist,
        "d_straight_post_buffer": d_straight_post_buffer,
        "d_available": d_available,
        "d_required": d_required,
        "v_safe_uniform": v_safe_uniform,
        "first_turn_angle": first_angle,
        "feasible": feasible,
        "deficit": deficit,
        "message": msg_turn,
    }


def print_braking_comparison(
        results: Dict[str, Dict[str, Any]],
        mass: float,
        risk_weight: float,
        obstacle_pos: Tuple[int, int],
) -> None:
    """
    Drukuje porównawczy raport dla 3 algorytmów.

    """
    print()
    print("═" * 100)
    print(f"  ANALIZA HAMOWANIA OD WYKRYCIA ZAGROŻENIA DO BEZPIECZNEJ PRĘDKOŚCI")
    print(f"  m = {mass:.0f} kg | W = {risk_weight:.0f} | "
          f"przeszkoda: ({obstacle_pos[0]}, {obstacle_pos[1]})")

    # Informacja o jednolitym progu oceny
    v_safe_uniform = compute_safe_turn_speed(WORST_TURN_ANGLE_RAD)
    print(f"  Próg oceny: najgorszy zakręt na siatce (135°), "
          f"v_safe = {v_safe_uniform:.2f} m/s (jednolity dla wszystkich algorytmów)")
    print("═" * 100)

    # Nagłówek tabeli
    print(f"  {'Algorytm':<16}"
          f"{'v_det':>8}"
          f"{'reak':>8}"
          f"{'bufor':>8}"
          f"{'prosta':>8}"
          f"{'DOSTĘPNA':>11}"
          f"{'WYMAGANA':>11}"
          f"{'deficyt':>10}"
          f"{'status':>15}")
    print(f"  {'':<16}"
          f"{'[m/s]':>8}"
          f"{'[m]':>8}"
          f"{'[m]':>8}"
          f"{'[m]':>8}"
          f"{'[m]':>11}"
          f"{'[m]':>11}"
          f"{'[m]':>10}"
          f"{'':>15}")
    print("  " + "─" * 98)

    for algo_name in ("Dijkstra", "A* Standard", "Risk-Aware A*"):
        r = results.get(algo_name)
        if r is None or not r.get("valid"):
            msg = r.get("message", "—") if r else "—"
            print(f"  {algo_name:<16}{msg:>82}")
            continue

        status_text = "WYKONALNY" if r["feasible"] else "NIEWYKONALNY"

        print(f"  {algo_name:<16}"
              f"{r['v_detect']:>8.2f}"
              f"{r['d_reaction']:>8.2f}"
              f"{r['d_buffer']:>8.2f}"
              f"{r['d_straight_post_buffer']:>8.2f}"
              f"{r['d_available']:>11.2f}"
              f"{r['d_required']:>11.2f}"
              f"{r['deficit']:>10.2f}"
              f"{status_text:>15}")

    print("  " + "─" * 98)

    print(f"  Pierwsze zakręty na trasach replanowania (informacyjnie, nie wpływają na ocenę):")
    for algo_name in ("Dijkstra", "A* Standard", "Risk-Aware A*"):
        r = results.get(algo_name)
        if r and r.get("valid") and r.get("first_turn_angle", 0) > 0:
            angle_deg = math.degrees(r["first_turn_angle"])
            print(f"    {algo_name:<16} → {angle_deg:.0f}°")
        elif r and r.get("valid"):
            print(f"    {algo_name:<16} → brak zakrętu (trasa prosta)")

    print("═" * 100)
    print()
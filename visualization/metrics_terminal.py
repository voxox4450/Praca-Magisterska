"""
metrics_terminal.py — Porównawczy raport hamowania w trybie dynamicznym.

Wywoływany RAZ po kliknięciu przeszkody (oraz po każdej zmianie suwaków
w trybie po kliknięciu). Drukuje w terminalu krótkie zestawienie 3 algorytmów
z jedną kluczową miarą: czy dron zdąży wyhamować od wykrycia zagrożenia
do bezpiecznej prędkości w pierwszym zakręcie nowej trasy.

[METODOLOGIA - PORÓWNANIE SYSTEMÓW NAWIGACYJNYCH]
Praca porównuje KOMPLETNE SYSTEMY NAWIGACYJNE, nie izolowane algorytmy grafowe.
Tytuł pracy ("Optymalizacja tras BSP z uwzględnieniem stref ryzyka") wymaga
oceny pełnego systemu planowania w dynamicznym środowisku miejskim.

System Risk-Aware A* obejmuje:
  - algorytm grafowy z modelem kinematycznym w funkcji kosztu
  - mechanizm bufora hamowania awaryjnego przed replanowaniem
    (wynika ze świadomości fizyki - algorytm "wie", że dron musi wytracić
    prędkość zanim zacznie nowy manewr)

Systemy klasyczne (Dijkstra, A* Standard) obejmują:
  - algorytm grafowy bez modelu kinematycznego
  - replanowanie natychmiast po fazie reakcji, bez bufora hamowania
    (algorytm nie posiada wiedzy o fizyce, więc nie wie, że hamowanie
    jest konieczne - to nie jest "krzywda" dla klasyków, lecz konsekwencja
    braku modelu kinematycznego)

Wnioski z porównania dotyczą zatem nie tylko jakości samego planowania
trasy, ale całościowego zachowania drona w środowisku z dynamicznymi
zagrożeniami - zgodnie z praktycznym zastosowaniem BSP w przestrzeni
zurbanizowanej.

Logika dla każdego algorytmu:
  1. Wywołujemy _compute_full_scenario → otrzymujemy pełny scenariusz
     (clean_path, detekcja, reakcja, bufor, replan_path).
  2. Liczymy:
       - droga reakcji  = długość odcinka clean_path[detect_idx..react_idx]
                          (dron leci dalej, nie hamuje — bezwładność systemu).
       - droga bufora   = długość zaplanowanego bufora hamowania
                          (>0 tylko dla Risk-Aware - klasyki tego nie planują).
       - odcinek prostej do 1. zakrętu = długość prostoliniowego początku
                          replan_path (tu dron może hamować bez zmiany kierunku).
       - droga dostępna na hamowanie = bufor + odcinek prostoliniowy
       - droga wymagana = (v_detect² - v_safe²) / (2a), gdzie v_safe to
                          bezpieczna prędkość pierwszego zakrętu nowej trasy.
  3. Status: WYKONALNY jeśli dostępna ≥ wymagana, NIEWYKONALNY w przeciwnym razie.
"""

import math
from typing import List, Tuple, Dict, Any, Callable, Optional

from algorithms.common import compute_safe_turn_speed
from config import MAX_THRUST_NET_N, V_MAX_MS


def _path_length(path: List[Tuple[int, int]]) -> float:
    """Suma odległości euklidesowych między kolejnymi węzłami ścieżki."""
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

    Zwraca: (kąt_w_radianach, indeks_węzła_zakrętu).
    Jeśli brak zakrętu — zwraca (0.0, -1).
    """
    if len(path) < 3:
        return 0.0, -1

    last_dir = None
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        curr_dir = (dx, dy)
        if last_dir is not None and curr_dir != last_dir:
            # zakręt w węźle i-1 (tam zmienia się kierunek)
            dot = last_dir[0] * curr_dir[0] + last_dir[1] * curr_dir[1]
            m1 = math.hypot(*last_dir)
            m2 = math.hypot(*curr_dir)
            cos_t = max(-1.0, min(1.0, dot / (m1 * m2)))
            angle = math.acos(cos_t)
            return angle, i - 1
        last_dir = curr_dir

    return 0.0, -1


def _straight_distance_to_first_turn(path: List[Tuple[int, int]]) -> float:
    """Długość prostoliniowego początku ścieżki — od path[0] do pierwszego zakrętu."""
    angle, turn_idx = _first_turn_on_path(path)
    if turn_idx == -1:
        return _path_length(path)
    return _path_length(path[:turn_idx + 1])


def _find_critical_turn(
        replan_path: List[Tuple[int, int]],
        v_start: float,
        accel: float,
        approach_heading: Tuple[int, int] = (0, 0),
) -> Dict[str, Any]:
    """
    Sprawdza wykonalność hamowania na replan_path metodą forward-pass.

    Symuluje drona startującego na początku replan_path z prędkością v_start
    i kierunkiem approach_heading (kierunek lotu w momencie wejścia w replan_path).
    Dla każdego zakrętu na trasie sprawdza czy dron zdąży wyhamować do
    v_safe(angle) na dostępnym odcinku prostoliniowym przed tym zakrętem.

    UWAGA: Pierwszy "zakręt" to zmiana kierunku z approach_heading na
    pierwszy kierunek replan_path. Dla algorytmów klasycznych (które
    nie planują z aktualnym kierunkiem lotu) ten zakręt często ma 0m
    odcinka prostego przed sobą — bo dron MA się skręcić natychmiast.

    Zwraca:
        worst_deficit       — największy deficyt drogi hamowania na trasie [m]
        critical_angle      — kąt zakrętu który okazał się problemowy (rad)
        critical_v_safe     — bezpieczna prędkość krytycznego zakrętu [m/s]
        critical_v_approach — prędkość drona w krytycznym zakręcie [m/s]
        critical_straight   — dostępny prosty odcinek przed krytycznym zakrętem [m]
        critical_brake_req  — wymagana droga hamowania [m]
        first_angle         — kąt pierwszego zakrętu na trasie (do informacji)
        first_straight      — odcinek do pierwszego zakrętu [m]
        all_feasible        — czy wszystkie zakręty są wykonalne
    """
    n = len(replan_path)
    if n < 2:
        return {
            "worst_deficit": 0.0,
            "critical_angle": 0.0, "critical_v_safe": 0.0,
            "critical_v_approach": v_start, "critical_straight": 0.0,
            "critical_brake_req": 0.0,
            "first_angle": 0.0, "first_straight": _path_length(replan_path),
            "all_feasible": True,
        }

    # Lista (idx_w_replan_path, angle_rad, distance_from_prev_turn)
    # gdzie idx to indeks węzła w którym następuje zakręt.
    # Pierwszy element listy uwzględnia zakręt wjazdowy
    # (z approach_heading na pierwszy kierunek replan_path).
    turns = []

    # Pierwszy kierunek replan_path
    first_dir = (replan_path[1][0] - replan_path[0][0],
                 replan_path[1][1] - replan_path[0][1])

    # Zakręt wjazdowy (jeżeli approach_heading != first_dir i znamy heading)
    if approach_heading != (0, 0) and first_dir != (0, 0) and first_dir != approach_heading:
        dot = approach_heading[0] * first_dir[0] + approach_heading[1] * first_dir[1]
        m1 = math.hypot(*approach_heading)
        m2 = math.hypot(*first_dir)
        if m1 * m2 > 0:
            cos_t = max(-1.0, min(1.0, dot / (m1 * m2)))
            entry_angle = math.acos(cos_t)
            # Zakręt wjazdowy: indeks 0, brak prostej przed nim
            turns.append((0, entry_angle, 0.0))

    # Zakręty wewnętrzne replan_path (zmiana kierunku między i-1, i)
    last_dir = first_dir
    last_turn_idx = 0
    for i in range(2, n):
        dx = replan_path[i][0] - replan_path[i - 1][0]
        dy = replan_path[i][1] - replan_path[i - 1][1]
        curr_dir = (dx, dy)
        if curr_dir != last_dir:
            dot = last_dir[0] * curr_dir[0] + last_dir[1] * curr_dir[1]
            m1 = math.hypot(*last_dir)
            m2 = math.hypot(*curr_dir)
            if m1 * m2 > 0:
                cos_t = max(-1.0, min(1.0, dot / (m1 * m2)))
                angle = math.acos(cos_t)
                straight = _path_length(replan_path[last_turn_idx:i])
                turns.append((i - 1, angle, straight))
                last_turn_idx = i - 1
        last_dir = curr_dir

    if not turns:
        return {
            "worst_deficit": 0.0,
            "critical_angle": 0.0, "critical_v_safe": 0.0,
            "critical_v_approach": 0.0, "critical_straight": 0.0,
            "critical_brake_req": 0.0,
            "first_angle": 0.0, "first_straight": _path_length(replan_path),
            "all_feasible": True,
        }

    first_angle = turns[0][1]
    first_straight = turns[0][2]

    # Forward pass: dla każdego zakrętu sprawdź czy dron zdąży wyhamować
    worst_deficit = 0.0
    crit = None
    v = v_start

    for (turn_idx, angle, straight) in turns:
        v_safe = compute_safe_turn_speed(angle)

        if v > v_safe:
            brake_required = (v ** 2 - v_safe ** 2) / (2 * accel)
            if brake_required > straight + 0.01:
                # Nie zdąży wyhamować — fizycznie niewykonalny zakręt
                deficit = brake_required - straight
                if deficit > worst_deficit:
                    worst_deficit = deficit
                    crit = {
                        "angle": angle, "v_safe": v_safe,
                        "v_approach": v, "straight": straight,
                        "brake_req": brake_required,
                    }
                # Załóż że jakoś przeszedł — kontynuuj z v_safe (modeluje hipotetyczny
                # nieidealny manewr; w rzeczywistości dron wypadłby z trasy).
                v = v_safe
            else:
                # Zdąży wyhamować na czas
                v = v_safe
        else:
            v = min(v, v_safe)

    if crit is None:
        return {
            "worst_deficit": 0.0,
            "critical_angle": 0.0, "critical_v_safe": 0.0,
            "critical_v_approach": 0.0, "critical_straight": 0.0,
            "critical_brake_req": 0.0,
            "first_angle": first_angle, "first_straight": first_straight,
            "all_feasible": True,
        }

    return {
        "worst_deficit": worst_deficit,
        "critical_angle": crit["angle"],
        "critical_v_safe": crit["v_safe"],
        "critical_v_approach": crit["v_approach"],
        "critical_straight": crit["straight"],
        "critical_brake_req": crit["brake_req"],
        "first_angle": first_angle, "first_straight": first_straight,
        "all_feasible": False,
    }


def analyze_braking_scenario(
        scenario_result: Dict[str, Any],
        mass: float,
) -> Dict[str, Any]:
    """
    Analizuje scenariusz dynamiczny pod kątem hamowania.

    Logika:
      - faza reakcji: dron leci dalej z prędkością v_detect przez czas
        reakcji systemu (proc_delay), pokonując d_reaction. NIE hamuje.
      - faza bufora (tylko Risk-Aware): dron hamuje na zaplanowanym buforze,
        wytracając prędkość z v_react_end do v_buffer_end.
      - faza replanowania: dron leci po nowej trasie. Sprawdzamy każdy
        zakręt — czy dron zdąży wyhamować do v_safe(angle) na odcinku
        prostoliniowym przed tym zakrętem.

    Wynik wykonalności = wszystkie zakręty na nowej trasie są możliwe
    do bezpiecznego przejścia z aktualnym profilem prędkości.

    Argumenty:
        scenario_result: wynik _compute_full_scenario z plotter.py
        mass: masa drona [kg]
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
            "d_to_first_turn": 0.0, "d_available": 0.0,
            "v_safe_first_turn": 0.0, "d_required": 0.0,
            "feasible": False, "deficit": float("inf"),
            "message": "KOLIZJA W FAZIE REAKCJI",
        }

    a = MAX_THRUST_NET_N / mass

    v_detect = scenario_result.get("v_detect", 0.0)
    v_react_end = scenario_result.get("v_react_end", v_detect)
    v_buffer_end = scenario_result.get("v_buffer_end", v_react_end)
    detect_idx = scenario_result.get("detect_idx", -1)
    react_idx = scenario_result.get("react_idx", -1)
    clean_path = scenario_result.get("clean_path", [])
    buffer_dist = scenario_result.get("buffer_dist", 0.0)
    replan_path = scenario_result.get("replan_path", [])
    heading = scenario_result.get("heading", (0, 0))

    # Faza reakcji: dron leci nie hamując (dystans = ile przeleciał między
    # detect_idx a react_idx na clean_path)
    if detect_idx >= 0 and react_idx > detect_idx and clean_path:
        d_reaction = _path_length(clean_path[detect_idx:react_idx + 1])
    else:
        d_reaction = 0.0

    # Prędkość startowa replan_path = po zakończeniu bufora (lub po reakcji
    # jeśli brak bufora — wtedy v_buffer_end == v_react_end).
    v_start_replan = v_buffer_end

    # Sprawdź wszystkie zakręty na replan_path (forward-pass fizyki)
    crit = _find_critical_turn(replan_path, v_start_replan, a,
                                approach_heading=heading)

    if crit["all_feasible"]:
        feasible = True
        deficit = 0.0
        # Pierwszy zakręt jako referencja (informacyjnie)
        ref_angle = crit["first_angle"]
        ref_straight = crit["first_straight"]
        msg_turn = (f"zakręt {math.degrees(ref_angle):.0f}° "
                    f"(wszystkie wykonalne)" if ref_angle > 0
                    else "brak zakrętu")
    else:
        feasible = False
        deficit = crit["worst_deficit"]
        ref_angle = crit["critical_angle"]
        ref_straight = crit["critical_straight"]
        msg_turn = f"krytyczny zakręt {math.degrees(ref_angle):.0f}°"

    # Bezpieczna prędkość zakrętu referencyjnego (pierwszy lub krytyczny)
    v_safe_ref = compute_safe_turn_speed(ref_angle) if ref_angle > 0 else 0.0

    # ── PEŁNA DROGA HAMOWANIA OD v_detect DO v_safe ─────────────────────
    # Tyle metrów potrzeba aby od momentu wykrycia (v=18 m/s) wyhamować
    # do bezpiecznej prędkości pierwszego/krytycznego zakrętu.
    # NIE uwzględnia drogi reakcji (bo w reakcji dron NIE hamuje).
    if v_detect > v_safe_ref:
        d_total_braking = (v_detect ** 2 - v_safe_ref ** 2) / (2.0 * a)
    else:
        d_total_braking = 0.0

    # ── RZECZYWIŚCIE PRZEBYTA DROGA OD DETEKCJI DO REFERENCYJNEGO ZAKRĘTU ──
    # Suma: reakcja + bufor + odcinek prostoliniowy replan_path do zakrętu
    d_total_traveled = d_reaction + buffer_dist + ref_straight

    # ── DROGA DOSTĘPNA NA HAMOWANIE = przebyta - reakcja ────────────────
    # (w fazie reakcji dron NIE hamuje, więc ta droga się nie liczy)
    d_available_for_braking = buffer_dist + ref_straight

    return {
        "valid": True,
        "v_detect": v_detect,
        "d_reaction": d_reaction,
        "d_buffer": buffer_dist,
        "d_to_turn": ref_straight,
        "d_total_traveled": d_total_traveled,
        "d_available_for_braking": d_available_for_braking,
        "d_total_braking": d_total_braking,
        "v_safe_ref": v_safe_ref,
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

    Argumenty:
        results: dict {nazwa_algo: dict_z_analyze_braking_scenario}
        mass, risk_weight, obstacle_pos: parametry symulacji (do nagłówka)
    """
    print()
    print("═" * 100)
    print(f"  ANALIZA HAMOWANIA OD WYKRYCIA ZAGROŻENIA DO PIERWSZEGO ZAKRĘTU")
    print(f"  m = {mass:.0f} kg | W = {risk_weight:.0f} | "
          f"przeszkoda: ({obstacle_pos[0]}, {obstacle_pos[1]})")
    print("═" * 100)

    # Nagłówek tabeli
    print(f"  {'Algorytm':<16}"
          f"{'v_det':>8}"
          f"{'reak':>8}"
          f"{'bufor':>8}"
          f"{'prosta':>8}"
          f"{'PRZEBYTA':>11}"
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

        # PRZEBYTA = bufor + prosta (droga w której dron MA możliwość hamowania)
        # Reakcja jest osobno bo w niej dron NIE hamuje
        d_braking_avail = r["d_available_for_braking"]

        print(f"  {algo_name:<16}"
              f"{r['v_detect']:>8.2f}"
              f"{r['d_reaction']:>8.2f}"
              f"{r['d_buffer']:>8.2f}"
              f"{r['d_to_turn']:>8.2f}"
              f"{d_braking_avail:>11.2f}"
              f"{r['d_total_braking']:>11.2f}"
              f"{r['deficit']:>10.2f}"
              f"{status_text:>15}")

    print("  " + "─" * 98)
    print(f"  Legenda kolumn:")
    print(f"    v_det     — prędkość drona w momencie wykrycia zagrożenia [m/s]")
    print(f"    reak      — droga reakcji (dron leci, NIE hamuje — bezwładność systemu) [m]")
    print(f"    bufor     — zaplanowany odcinek hamowania (część systemu Risk-Aware A*) [m]")
    print(f"    prosta    — odcinek prostoliniowy nowej trasy do pierwszego zakrętu [m]")
    print(f"    PRZEBYTA  — łączna droga dostępna NA HAMOWANIE = bufor + prosta [m]")
    print(f"    WYMAGANA  — pełna droga hamowania od v_det do v_safe pierwszego zakrętu [m]")
    print(f"    deficyt   — WYMAGANA - PRZEBYTA (>0 oznacza, że dron nie zdąży wyhamować)")
    print()

    # Linia z informacją o zakręcie
    for algo_name in ("Dijkstra", "A* Standard", "Risk-Aware A*"):
        r = results.get(algo_name)
        if r and r.get("valid") and "message" in r:
            v_safe = r.get("v_safe_ref", 0.0)
            print(f"    {algo_name:<16} → {r['message']}, v_safe = {v_safe:.2f} m/s")

    print("═" * 100)
    print()
"""
metrics_terminal.py — Porównawczy raport hamowania w trybie dynamicznym.

Wywoływany RAZ po kliknięciu przeszkody (oraz po każdej zmianie suwaków
w trybie po kliknięciu). Drukuje w terminalu krótkie zestawienie 3 algorytmów
z jedną kluczową miarą: czy dron zdąży wyhamować od wykrycia zagrożenia
do bezpiecznej prędkości pozwalającej na DOWOLNY zakręt na siatce.

[METODOLOGIA — PORÓWNANIE SYSTEMÓW NAWIGACYJNYCH]
Praca porównuje KOMPLETNE SYSTEMY NAWIGACYJNE, nie izolowane algorytmy grafowe.
Tytuł pracy ("Optymalizacja tras BSP z uwzględnieniem stref ryzyka") wymaga
oceny pełnego systemu planowania w dynamicznym środowisku miejskim.

System Risk-Aware A* obejmuje:
  - algorytm grafowy z modelem kinematycznym w funkcji kosztu,
  - mechanizm bufora hamowania awaryjnego przed replanowaniem
    (wynika ze świadomości fizyki — algorytm "wie", że dron musi wytracić
    prędkość zanim zacznie nowy manewr).

Systemy klasyczne (Dijkstra, A* Standard) obejmują:
  - algorytm grafowy bez modelu kinematycznego,
  - replanowanie natychmiast po fazie reakcji, bez bufora hamowania
    (algorytm nie posiada wiedzy o fizyce, więc nie wie, że hamowanie
    jest konieczne — to nie jest "krzywda" dla klasyków, lecz konsekwencja
    braku modelu kinematycznego).

[METODOLOGIA — JEDNOLITY PRÓG OCENY]
Wszystkie trzy systemy oceniane są pod identyczne kryterium fizyczne:
zdolność do wyhamowania od prędkości v_detect do bezpiecznej prędkości
NAJGORSZEGO możliwego zakrętu na siatce 8-kierunkowej (135°).

Wybór jednolitego progu (zamiast progu dostosowanego do faktycznego
pierwszego zakrętu każdej trasy) gwarantuje pełną porównywalność:
  - kryterium identyczne dla wszystkich algorytmów,
  - reprezentuje sytuację najtrudniejszą fizycznie (najniższa v_safe),
  - dron zdolny do tego progu jest gotowy na DOWOLNY zakręt na siatce.

Dzięki temu różnice w wynikach (WYKONALNY vs NIEWYKONALNY) wynikają
WYŁĄCZNIE z różnic w wewnętrznej logice algorytmów (obecność/brak bufora
hamowania awaryjnego), a nie z przypadkowego wyboru łagodniejszych
zakrętów przez dany algorytm. Jeśli Risk-Aware A* spełnia surowy próg
135° mimo wybierania ostrzejszych zakrętów na trasie, to znaczy że robi
to ŚWIADOMIE (planuje pod to bufor), a nie szczęśliwym zbiegiem geometrii.

[METODOLOGIA — POJEDYNCZA CIĄGŁA DROGA HAMOWANIA]
Od momentu wykrycia zagrożenia (v_detect) dron hamuje aż do osiągnięcia
bezpiecznej prędkości v_safe(135°). Pełna dostępna droga hamowania
składa się z trzech kolejnych rozłącznych odcinków:

  v_detect
      ↓ faza bezwładności systemu      → długość d_reaction
  v_react_end
      ↓ bufor hamowania awaryjnego     → długość d_buffer    (tylko Risk-Aware A*)
  v_buffer_end
      ↓ prostoliniowy odcinek replan_path → długość d_straight_post_buffer
  v_safe(135°)

Wszystkie trzy fazy to ciągłe hamowanie z tym samym przyspieszeniem a = F/m.
Sumowanie ich do jednej DOSTĘPNEJ drogi jest fizycznie poprawne:

    DOSTĘPNA = d_reaction + d_buffer + d_straight_post_buffer
    WYMAGANA = (v_detect² − v_safe(135°)²) / (2a)
    deficyt  = max(0, WYMAGANA − DOSTĘPNA)
    status   = WYKONALNY gdy deficyt ≈ 0, NIEWYKONALNY w przeciwnym razie.

Dla algorytmów klasycznych d_buffer = 0, więc DOSTĘPNA = d_reaction + prosta.
Dla Risk-Aware A* d_buffer > 0, więc DOSTĘPNA jest powiększona o tę fazę —
co odpowiada faktycznej zdolności systemu do hamowania awaryjnego.

[UWAGA TECHNICZNA — KOLUMNA `prosta`]
Risk-Aware A* dokleja bufor hamowania na początku swojego replan_path,
więc surowy odcinek prostoliniowy "od indeksu 0 replan_path do pierwszego
zakrętu" zawiera w sobie bufor. W kolumnie `prosta` raportujemy WYŁĄCZNIE
część PO buforze (d_straight_post_buffer = d_to_turn − d_buffer), żeby
trzy kolumny `reak`, `bufor`, `prosta` były rozłączne i poprawnie się
sumowały do kolumny DOSTĘPNA.
"""

import math
from typing import List, Tuple, Dict, Any

from algorithms.common import compute_safe_turn_speed
from config import MAX_THRUST_NET_N


# Najgorszy zakręt na siatce 8-kierunkowej: 135° (3π/4 rad).
# Liczone raz na poziomie modułu — referencyjny próg WYMAGANEJ drogi hamowania.
WORST_TURN_ANGLE_RAD = math.radians(135)


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
    """
    Zwraca (kąt_pierwszego_zakrętu, długość_prostoliniowego_odcinka_od_początku_replan).

    Długość liczona jest od indeksu 0 replan_path do węzła pierwszego zakrętu.
    UWAGA: dla Risk-Aware A* ten odcinek zawiera w sobie bufor hamowania,
    który jest doczepiony na początku replan_path. Korekta (odjęcie buffer_dist)
    realizowana jest w analyze_braking_scenario.

    Uwzględnia zakręt wjazdowy: jeśli approach_heading różni się od pierwszego
    kierunku replan_path, to PIERWSZY zakręt następuje na pozycji 0 i prosta
    przed nim ma długość 0.

    Wartość kąta służy WYŁĄCZNIE do raportowania pod tabelą (która kategoria
    zakrętu wystąpiła w trasie danego algorytmu). Do oceny wykonalności
    używany jest jednolity próg WORST_TURN_ANGLE_RAD.

    Brak zakrętu (trasa prosta do celu) → (0.0, długość_całej_replan_path).
    """
    if len(replan_path) < 2:
        return 0.0, 0.0

    first_dir = (replan_path[1][0] - replan_path[0][0],
                 replan_path[1][1] - replan_path[0][1])

    # Zakręt wjazdowy: heading drona ≠ pierwszy kierunek replan_path
    if approach_heading != (0, 0) and first_dir != (0, 0) and first_dir != approach_heading:
        dot = approach_heading[0] * first_dir[0] + approach_heading[1] * first_dir[1]
        m1 = math.hypot(*approach_heading)
        m2 = math.hypot(*first_dir)
        if m1 * m2 > 0:
            cos_t = max(-1.0, min(1.0, dot / (m1 * m2)))
            entry_angle = math.acos(cos_t)
            if entry_angle > 0.01:
                return entry_angle, 0.0

    # Brak zakrętu wjazdowego — szukaj pierwszego zakrętu wewnętrznego
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

    [METODOLOGIA]
    WYMAGANA droga hamowania liczona jest dla najgorszego możliwego zakrętu
    na siatce 8-kierunkowej (135°), niezależnie od tego, jaki konkretnie
    zakręt wygenerował algorytm. Zapewnia to pełną porównywalność między
    algorytmami: ocenie podlega zdolność systemu do bezpiecznego hamowania,
    a nie szczęśliwy zbieg okoliczności geometrii trasy.

    Pełna droga hamowania od v_detect do v_safe(135°) traktowana jest jako
    JEDNA CIĄGŁA droga obejmująca trzy fazy:
      - reakcję (faza bezwładności systemu),
      - bufor hamowania awaryjnego (jeśli istnieje; tylko Risk-Aware A*),
      - prostoliniowy odcinek replan_path PO buforze.
    Wszystkie z tym samym przyspieszeniem a = F/m.

        DOSTĘPNA = d_reaction + d_buffer + d_straight_post_buffer
        WYMAGANA = (v_detect² − v_safe(135°)²) / (2a)
        deficyt  = max(0, WYMAGANA − DOSTĘPNA)

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

    # ── DROGA REAKCJI ───────────────────────────────────────────────────
    # Faza bezwładności systemu: dron HAMUJE zgodnie z v_react_end = v_detect - a·t_proc
    # (zobacz plotter.py:300). Długość = odcinek clean_path[detect_idx..react_idx].
    if detect_idx >= 0 and react_idx > detect_idx and clean_path:
        d_reaction = _path_length(clean_path[detect_idx:react_idx + 1])
    else:
        d_reaction = 0.0

    # ── FAKTYCZNY PIERWSZY ZAKRĘT TRASY REPLANOWANIA ───────────────────
    # d_to_turn_raw — długość prostoliniowej części od indeksu 0 replan_path.
    # Dla R-A A* zawiera w sobie bufor (bufor jest doczepiony na początku).
    # Kąt — TYLKO informacyjnie, do raportowania pod tabelą.
    # Do oceny wykonalności używamy jednolitego progu 135°.
    first_angle, d_to_turn_raw = _first_turn_with_entry(
        replan_path, approach_heading=heading
    )

    # ── PROSTA PO BUFORZE ───────────────────────────────────────────────
    # Wycinamy z d_to_turn_raw tę część, która jest buforem, żeby trzy
    # kolumny reak / bufor / prosta były rozłączne i poprawnie sumowały
    # się do DOSTĘPNEJ.
    d_straight_post_buffer = max(0.0, d_to_turn_raw - buffer_dist)

    # ── JEDNOLITY PRÓG: NAJGORSZY ZAKRĘT NA SIATCE 8-KIERUNKOWEJ ────────
    # Identyczny dla wszystkich trzech algorytmów — zapewnia porównywalność.
    v_safe_uniform = compute_safe_turn_speed(WORST_TURN_ANGLE_RAD)

    # ── WYMAGANA DROGA HAMOWANIA (jednofazowo, od v_detect) ─────────────
    # Pełna droga od momentu detekcji (v_detect) do bezpiecznej prędkości
    # najgorszego zakrętu (v_safe_uniform). Ta sama formuła, ten sam próg,
    # dla wszystkich algorytmów.
    if v_detect > v_safe_uniform:
        d_required = (v_detect ** 2 - v_safe_uniform ** 2) / (2.0 * a)
    else:
        d_required = 0.0

    # ── DOSTĘPNA DROGA HAMOWANIA (suma trzech rozłącznych faz) ──────────
    # Łączny odcinek od momentu detekcji do pierwszego zakrętu na replan_path:
    #   - reakcja: dron HAMUJE w tej fazie (v_detect → v_react_end),
    #   - bufor: zaplanowany odcinek hamowania awaryjnego (Risk-Aware A*),
    #   - prosta: prostoliniowy odcinek replan_path PO buforze.
    d_available = d_reaction + buffer_dist + d_straight_post_buffer

    # ── DEFICYT I STATUS ────────────────────────────────────────────────
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

    Argumenty:
        results: dict {nazwa_algo: dict_z_analyze_braking_scenario}
        mass, risk_weight, obstacle_pos: parametry symulacji (do nagłówka)
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
    print(f"  Legenda kolumn:")
    print(f"    v_det     — prędkość drona w momencie wykrycia zagrożenia [m/s]")
    print(f"    reak      — droga w fazie bezwładności systemu (dron HAMUJE: v_det → v_react_end) [m]")
    print(f"    bufor     — zaplanowany odcinek hamowania awaryjnego (tylko Risk-Aware A*) [m]")
    print(f"    prosta    — prostoliniowy odcinek replan_path PO BUFORZE, przed pierwszym zakrętem [m]")
    print(f"    DOSTĘPNA  — łączna droga na hamowanie = reak + bufor + prosta [m]")
    print(f"    WYMAGANA  — droga hamowania od v_det do v_safe NAJGORSZEGO zakrętu (135°) [m]")
    print(f"    deficyt   — WYMAGANA − DOSTĘPNA (>0 oznacza fizyczną niewykonalność manewru)")
    print()

    # Informacja o faktycznym pierwszym zakręcie każdego algorytmu (informacyjnie)
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
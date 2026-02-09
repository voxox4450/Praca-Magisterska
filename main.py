from environment.grid_map import GridMap
from algorithms.a_star import run_search
from visualization.plotter import plot_simulation
import json
import os
from typing import Dict, Any, Tuple, List


def main() -> None:
    # === 1. KONFIGURACJA ŚRODOWISKA (Zgodnie z metodyką) ===
    SIZE = 100
    # Tworzymy mapę z 8 strefami ryzyka (symulacja zakłóceń/zagęszczenia ludzi)
    env = GridMap(width=SIZE, height=SIZE, risk_zones_count=8)

    start_pos = (5, 5)
    goal_pos = (95, 95)

    print(f"=== ROZPOCZĘCIE BADAŃ SYMULACYJNYCH (Mapa {SIZE}x{SIZE}) ===\n")

    # === SCENARIUSZ 0: BENCHMARK - ALGORYTM DIJKSTRY ===
    # Używamy algorithm_type="dijkstra".
    # Cel: Wykazanie różnicy w liczbie odwiedzonych węzłów względem A*.
    # Dijkstra nie używa heurystyki, więc przeszukuje mapę we wszystkich kierunkach.
    print("--- URUCHAMIANIE DIJKSTRY ---")
    path_dijkstra, stats_dijkstra = run_search(env, start_pos, goal_pos,
                                               algorithm_type="dijkstra",
                                               risk_weight=0.0,
                                               turn_penalty=0.0)
    print_results("SCENARIUSZ 0: Dijkstra (Benchmark)", stats_dijkstra)
    if path_dijkstra:
        plot_simulation(env, path_dijkstra, "Scenariusz 0: Dijkstra (Benchmark)")

    # === SCENARIUSZ 1: TRASA BAZOWA (Najkrótsza) ===
    # Używamy A* bez wag ryzyka i bez kar za skręty.
    # Cel: Punkt odniesienia (Benchmark).
    path_base, stats_base = run_search(env, start_pos, goal_pos,
                                       algorithm_type="astar",
                                       risk_weight=0.0,
                                       turn_penalty=0.0)
    print_results("SCENARIUSZ 1: A* Bazowy (Najkrótszy)", stats_base)
    if path_base:
        plot_simulation(env, path_base, "Scenariusz 1: Trasa Bazowa (Ignoruje Ryzyko)")

    # === SCENARIUSZ 2: TRASA BEZPIECZNA I PŁYNNA (Weryfikacja H1 i H2) ===
    # Używamy Risk-Aware A* z wagą ryzyka (W) i karą za skręty (Turn Penalty).
    # H1: Omija czerwone strefy.
    # H2: Ma mniej zakrętów.
    RISK_W = 20.0
    TURN_P = 2.0
    path_safe, stats_safe = run_search(env, start_pos, goal_pos,
                                       algorithm_type="risk_aware",
                                       risk_weight=RISK_W,
                                       turn_penalty=TURN_P)

    print_results(f"SCENARIUSZ 2: Risk-Aware A* (W={RISK_W}, TurnPenalty={TURN_P})", stats_safe)
    if path_safe:
        plot_simulation(env, path_safe, "Scenariusz 2: Trasa Bezpieczna i Płynna")

    # === SCENARIUSZ 3: REAGOWANIE NA ZAGROŻENIA (Weryfikacja H3) ===
    # Symulacja dynamiczna: Dron leci, pojawia się przeszkoda, dron re-planuje.
    print("\n--- SCENARIUSZ 3: Dynamiczna Reakcja (H3) ---")

    if not path_safe:
        print("Brak trasy bazowej do symulacji dynamicznej.")
        return

    # 1. Dron leci kawałek po bezpiecznej trasie (np. 30% trasy)
    steps_flown = int(len(path_safe) * 0.3)
    current_drone_pos = path_safe[steps_flown]
    print(f"Dron doleciał do punktu: {current_drone_pos}")

    # 2. NAGLE POJAWIA SIĘ NOWA STREFA ZAKAZANA (np. wypadek)
    # Wstawiamy przeszkodę (1.0) dokładnie na dalszej części zaplanowanej trasy
    obstacle_pos = path_safe[int(len(path_safe) * 0.5)]
    print(f"ALERT! Wykryto nowe zagrożenie w punkcie: {obstacle_pos}")

    # Aktualizujemy mapę (dodajemy przeszkodę 10x10)
    for i in range(obstacle_pos[0] - 5, obstacle_pos[0] + 5):
        for j in range(obstacle_pos[1] - 5, obstacle_pos[1] + 5):
            if 0 <= i < SIZE and 0 <= j < SIZE:
                env.grid[i, j] = 1.0  # No-Fly Zone

    # 3. LOKALNA REAKCJA (Re-planowanie)
    # Uruchamiamy A* od obecnej pozycji drona do celu
    print("Inicjowanie manewru omijania...")
    path_dynamic, stats_dyn = run_search(env, current_drone_pos, goal_pos,
                                         algorithm_type="risk_aware",
                                         risk_weight=RISK_W,
                                         turn_penalty=TURN_P)

    if path_dynamic:
        print(f"Manewr udany! Czas reakcji: {stats_dyn['time']:.4f} s")
        # Łączymy trasę przebytą z nową trasą
        full_dynamic_path = path_safe[:steps_flown] + path_dynamic
        plot_simulation(env, full_dynamic_path, "Scenariusz 3: Dynamiczne Omijanie (Reakcja)")
    else:
        print("CRITICAL: Nie udało się znaleźć trasy alternatywnej!")

def print_results(name: str, stats: Dict[str, Any]) -> None:
    if not stats:
        print(f"{name}: Nie znaleziono trasy!")
        return
    print(f"--- {name} ---")
    print(f"  Czas obliczeń:   {stats['time']:.4f} s")
    print(f"  Długość trasy:   {stats['length']:.2f} m")
    print(f"  Skumulowane ryzyko: {stats['total_risk']:.2f} pkt")
    print(f"  Liczba zakrętów: {stats['turns']}")
    print(f"  Odwiedzone węzły:{stats['nodes_expanded']}")
    print("-" * 30)

"""
def save_results_to_json(filename: str, stats: Dict[str, Any]) -> None:
    if not os.path.exists("results"):
        os.makedirs("results")
    path = os.path.join("results", filename)
    with open(path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"(Zapisano wyniki do {path})")
"""

if __name__ == "__main__":
    main()
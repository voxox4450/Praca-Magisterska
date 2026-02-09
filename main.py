import matplotlib

matplotlib.use('TkAgg')  # Wymuszenie okna systemowego dla płynności

from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import plot_simulation, plot_interactive_risk
from typing import Dict, Any, Tuple, List


def print_results(name: str, stats: Dict[str, Any]) -> None:
    print(f"--- {name} ---")
    if stats['found']:
        print(f"  Czas:     {stats['time']:.4f} s")
        print(f"  Długość:  {stats['length']:.2f} m")
        print(f"  Ryzyko:   {stats['risk']:.2f}")
        print(f"  Zakręty:  {stats['turns']}")
        print(f"  Węzły:    {stats['nodes']}")
    else:
        print("  [X] Nie znaleziono trasy!")
    print("-" * 30)


def get_user_difficulty() -> float:
    """Pobiera od użytkownika poziom zagęszczenia przeszkód."""
    print("\nWYBIERZ POZIOM ZAGĘSZCZENIA PRZESZKÓD (NO-FLY ZONES):")
    print("1. Mało (5% mapy)")
    print("2. Średnio (15% mapy)")
    print("3. Dużo (30% mapy )")

    while True:
        choice = input("Twój wybór (1-3): ").strip()
        if choice == '1':
            return 0.05
        elif choice == '2':
            return 0.15
        elif choice == '3':
            return 0.30
        else:
            print("Niepoprawny wybór. Wpisz 1, 2 lub 3.")


def main() -> None:
    # 1. Konfiguracja interaktywna
    print(f"=== SYSTEM SYMULACJI BSP (Konfiguracja) ===")

    SIZE: int = 100
    density: float = get_user_difficulty()  # Wybór użytkownika

    # Tworzenie mapy z wybranym zagęszczeniem
    env: GridMap = GridMap(width=SIZE, height=SIZE, risk_zones_count=8, obstacle_density=density)

    start_pos: Tuple[int, int] = (5, 5)
    goal_pos: Tuple[int, int] = (95, 95)

    print(f"\n=== START SYMULACJI (Zagęszczenie: {density * 100}%) ===\n")

    # === SCENARIUSZ 0: DIJKSTRA ===
    path_d, stats_d = run_dijkstra(env, start_pos, goal_pos)
    print_results("0. Dijkstra (Benchmark)", stats_d)
    if path_d:
        plot_simulation(env, path_d, "Dijkstra", block=False)

    # === SCENARIUSZ 1: A* STANDARD ===
    path_a, stats_a = run_astar(env, start_pos, goal_pos)
    print_results("1. A* Standard", stats_a)
    if path_a:
        plot_simulation(env, path_a, "A* Standard", block=False)

    # === SCENARIUSZ 2: RISK-AWARE A* ===
    path_risk, stats_risk = run_risk_astar(env, start_pos, goal_pos, risk_weight=20.0)
    print_results("2. Risk-Aware A* (W=20)", stats_risk)

    # === SCENARIUSZ INTERAKTYWNY ===
    print("\n--- Otwieranie okna z suwakiem ---")
    plot_interactive_risk(env, start_pos, goal_pos, run_risk_astar)


if __name__ == "__main__":
    main()
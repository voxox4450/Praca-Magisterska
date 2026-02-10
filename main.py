import matplotlib

# TkAgg pozwala na obsługę wielu okien naraz
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import plot_simulation, plot_interactive_risk
from typing import Tuple, List, Dict, Any


def main() -> None:
    print(f"=== SYSTEM SYMULACJI BSP - Tryb Wielookienkowy ===")

    # 1. KONFIGURACJA
    SIZE: int = 100
    density: float = get_user_difficulty()
    env: GridMap = GridMap(width=SIZE, height=SIZE, risk_zones_count=8, obstacle_density=density)

    start_pos: Tuple[int, int] = (5, 5)
    goal_pos: Tuple[int, int] = (95, 95)

    # ETAP 1: OBLICZENIA
    print(f"\n[1] Tabela porównawcza dla H1 - Wpływ wagi ryzyka na długość trasy i ryzyko:")

    # Dijkstra
    path_d, stats_d = run_dijkstra(env, start_pos, goal_pos)
    base_len = stats_d['length'] if stats_d['found'] else 0
    base_risk = stats_d['risk'] if stats_d['found'] else 0

    # A* Standard
    path_a, stats_a = run_astar(env, start_pos, goal_pos)

    # Tabela Risk-Aware (dla dowodu naukowego w konsoli)
    risk_weights = [0.0, 5.0, 20.0, 50.0]
    print("\n" + "=" * 80)
    print(f"{'WYNIKI':^80}")
    print("=" * 80)
    print(f"{'Waga (W)':<10} | {'Dystans':<10} | {'Koszt [%]':<10} | {'Ryzyko':<10} | {'Poprawa [%]':<12}")
    print("-" * 80)

    for w in risk_weights:
        _, stats = run_risk_astar(env, start_pos, goal_pos, risk_weight=w)
        if stats['found'] and base_len > 0:
            len_increase = ((stats['length'] - base_len) / base_len) * 100
            risk_reduction = 0.0
            if base_risk > 0:
                risk_reduction = ((base_risk - stats['risk']) / base_risk) * 100
            print(
                f"{w:<10.1f} | {stats['length']:<10.2f} | +{len_increase:<9.2f} | {stats['risk']:<10.2f} | -{risk_reduction:<11.2f}")
    print("=" * 80)

    # ETAP 2: OTWIERANIE OKIEN

    # 1. Dijkstra
    if path_d:
        plot_simulation(env, path_d, stats_d, "1. Dijkstra", block=False)

    # 2. A* Standard
    if path_a:
        plot_simulation(env, path_a, stats_a, "2. A* Standard", block=False)

    # 3. A* Risk-Aware
    plot_interactive_risk(env, start_pos, goal_pos, run_risk_astar)


def get_user_difficulty() -> float:
    print("\nWYBIERZ POZIOM TRUDNOŚCI:")
    print("1. Mało przeszkód (5%)")
    print("2. Średnio (15%) - ZALECANE")
    print("3. Dużo przeszkód (30%)")
    while True:
        choice = input("Wybór (1-3): ").strip()
        if choice == '1':
            return 0.05
        elif choice == '2':
            return 0.15
        elif choice == '3':
            return 0.30


if __name__ == "__main__":
    main()
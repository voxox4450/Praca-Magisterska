from environment.grid_map import GridMap
from algorithms.a_star import run_search
from visualization.plotter import plot_simulation
import time


def main():
    # 1. Konfiguracja mapy (100x100, 150x150 lub 200x200)
    SIZE = 100
    env = GridMap(width=SIZE, height=SIZE, risk_zones_count=8)

    start_pos = (5, 5)
    goal_pos = (SIZE - 5, SIZE - 5)git

    print(f"--- Symulacja na mapie {SIZE}x{SIZE} ---")

    # 2. Uruchomienie Dijkstry
    print("\n1. Algorytm Dijkstry...")
    t0 = time.time()
    path_dijkstra, nodes_d, cost_d = run_search(env, start_pos, goal_pos, "dijkstra")
    t1 = time.time()
    print(f"   Czas: {t1 - t0:.4f}s | Długość/Koszt: {cost_d:.2f} | Odwiedzone węzły: {nodes_d}")
    if path_dijkstra:
        plot_simulation(env, path_dijkstra, f"Dijkstra (Nodes: {nodes_d})")

    # 3. Uruchomienie A* (Standard)
    print("\n2. Algorytm A* (Standard)...")
    t0 = time.time()
    path_astar, nodes_a, cost_a = run_search(env, start_pos, goal_pos, "astar")
    t1 = time.time()
    print(f"   Czas: {t1 - t0:.4f}s | Długość/Koszt: {cost_a:.2f} | Odwiedzone węzły: {nodes_a}")
    if path_astar:
        plot_simulation(env, path_astar, f"A* Standard (Nodes: {nodes_a})")

    # 4. Uruchomienie A* (Risk Aware - Zoptymalizowany)
    # Tutaj WAGA RYZYKA (risk_weight) jest kluczowa.
    # W=0 -> zwykły A*, W=50 -> bardzo boi się ryzyka
    RISK_WEIGHT = 20.0
    print(f"\n3. Algorytm A* (Risk Aware, W={RISK_WEIGHT})...")
    t0 = time.time()
    path_opt, nodes_opt, cost_opt = run_search(env, start_pos, goal_pos, "risk_aware", risk_weight=RISK_WEIGHT)
    t1 = time.time()
    print(f"   Czas: {t1 - t0:.4f}s | Długość/Koszt: {cost_opt:.2f} | Odwiedzone węzły: {nodes_opt}")
    if path_opt:
        plot_simulation(env, path_opt, f"A* Risk Aware (W={RISK_WEIGHT})")


if __name__ == "__main__":
    main()
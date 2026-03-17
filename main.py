from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import run_online_simulation
import random
import numpy as np

from config import (
    MAP_SIZE, RISK_ZONES_COUNT, RANDOM_SEED,
    START_POS, GOAL_POS,
    COLLISION_RADIUS,
    RISK_WEIGHT, TURN_PENALTY,
    DRONE_MASS_KG, MAX_THRUST_NET_N, V_MAX_MS
)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    while True:
        print(f"=" * 32)
        print(f"   SYSTEM PLANOWANIA TRAS BSP     ")
        print(f"=" * 32)
        print(f"Mapa: {MAP_SIZE}x{MAP_SIZE} | Start: {START_POS} | Cel: {GOAL_POS}")
        print(f"Dron: {DRONE_MASS_KG} kg | V_max: {V_MAX_MS} m/s | W={RISK_WEIGHT} | Kara={TURN_PENALTY}")
        print(f"=" * 32)
        print("WYBIERZ POZIOM TRUDNOŚCI OTOCZENIA:")
        print("1. Mało przeszkód (5%)")
        print("2. Średnio (15%)")
        print("3. Dużo przeszkód (30%)")
        print("Naciśnij q aby zakończyć program")

        choice = input("\nWybór (1-3, q): ").strip()

        if choice == '1':
            density = 0.05
        elif choice == '2':
            density = 0.15
        elif choice == '3':
            density = 0.30
        elif choice in ('q', 'Q'):
            break
        else:
            print("Nieprawidłowy wybór — wpisz 1, 2, 3 lub q.")
            continue

        print(f"\n Uruchomienie trybu dynamicznego (Gęstość: {density * 100:.0f}%)")

        env = GridMap(
            width=MAP_SIZE, height=MAP_SIZE,
            start_pos=START_POS, goal_pos=GOAL_POS,
            risk_zones_count=RISK_ZONES_COUNT,
            obstacle_density=density
        )

        run_online_simulation(env, START_POS, GOAL_POS,
                              search_func=run_risk_astar,
                              collision_radius=COLLISION_RADIUS,
                              func_dijkstra=run_dijkstra,
                              func_astar=run_astar)


if __name__ == "__main__":
    main()
from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import plot_simulation, plot_interactive_risk, run_online_simulation
from algorithms.common import generate_analysis_table
from visualization.plotter import generate_thesis_charts
from typing import Tuple
import random
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# JEDYNE MIEJSCE KONFIGURACJI – zmień tu, zmiana działa wszędzie
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    MAP_SIZE, RISK_ZONES_COUNT, RANDOM_SEED,
    START_POS, GOAL_POS,
    COLLISION_RADIUS,
    RISK_WEIGHT, TURN_PENALTY_CLASSIC, TURN_PENALTY_RISK,
    N_TESTS,
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
        print(f"Dron: {DRONE_MASS_KG} kg | V_max: {V_MAX_MS} m/s | W={RISK_WEIGHT} | Kara klasyczna={TURN_PENALTY_CLASSIC} / Risk={TURN_PENALTY_RISK}")
        print(f"=" * 32)
        print("WYBIERZ TRYB PRACY:")
        print("1. Wersja Statyczna (Offline) ")
        print("2. Wersja Dynamiczna (Online)  ")
        print("3. PEŁNE BADANIE (Automatyczny Benchmark 3 poziomów)")
        print("Naciśnij q aby zakończyć program")

        mode = input("\nTwój wybór 1 - Offline , 2 - Online , q - Opuść: ").strip()

        if mode == '1':
            density = get_user_difficulty()
            run_offline_mode(density=density, interactive=True)
        elif mode == '2':
            run_online_mode()
        elif mode == '3':
            run_batch_benchmark()
        elif mode == 'q' or mode == 'Q':
            break
        else:
            print("Nieprawidłowy wybór")


def get_user_difficulty() -> float:
    """Wspólna funkcja wyboru trudności dla obu trybów."""
    print("\nWYBIERZ POZIOM TRUDNOŚCI OTOCZENIA:")
    print("1. Mało przeszkód (5%)")
    print("2. Średnio (15%)")
    print("3. Dużo przeszkód (30%)")

    while True:
        choice = input("Wybór (1-3): ").strip()
        if choice == '1':
            return 0.05
        elif choice == '2':
            return 0.15
        elif choice == '3':
            return 0.30


def run_offline_mode(density: float, interactive: bool = True, density_label: str = "") -> None:
    if interactive:
        print(f"\n Uruchomienie trybu statycznego (Gęstość: {density * 100:.0f}%) ")

    env = GridMap(
        width=MAP_SIZE, height=MAP_SIZE,
        start_pos=START_POS, goal_pos=GOAL_POS,
        risk_zones_count=RISK_ZONES_COUNT,   # z config – ujednolicone
        obstacle_density=density
    )

    path_d, stats_d = run_dijkstra(env, START_POS, GOAL_POS,
                                   risk_weight=RISK_WEIGHT, turn_penalty=TURN_PENALTY_CLASSIC,
                                   drone_radius=COLLISION_RADIUS)
    path_a, stats_a = run_astar(env, START_POS, GOAL_POS,
                                risk_weight=RISK_WEIGHT, turn_penalty=TURN_PENALTY_CLASSIC,
                                drone_radius=COLLISION_RADIUS)
    path_r, stats_r = run_risk_astar(env, START_POS, GOAL_POS,
                                     risk_weight=RISK_WEIGHT, turn_penalty=TURN_PENALTY_RISK,
                                     drone_radius=COLLISION_RADIUS)

    generate_thesis_charts(
        envs=[env], start=START_POS, goal=GOAL_POS,
        func_dijkstra=run_dijkstra, func_astar=run_astar, func_risk_astar=run_risk_astar,
        collision_radius=COLLISION_RADIUS, density_label=density_label,
        turn_penalty_classic=TURN_PENALTY_CLASSIC, turn_penalty_risk=TURN_PENALTY_RISK
    )

    if interactive:
        print("\n Wizualizacja algorytmów na mapie...")
        if path_d:
            plot_simulation(env, path_d, stats_d, "1. Dijkstra (Referencja W=20)", block=False, use_smoothing=True)
        if path_a:
            plot_simulation(env, path_a, stats_a, "2. A* Standard (Szybki W=20)", block=False, use_smoothing=True)
        plot_interactive_risk(env, START_POS, GOAL_POS, run_risk_astar)


def run_batch_benchmark() -> None:
    print("\n" + "=" * 50)
    print(f" ROZPOCZYNAM BENCHMARK (Monte Carlo, N={N_TESTS} map / scenariusz)")
    print("=" * 50)

    scenarios = {
        "5_procent_malo":    0.05,
        "15_procent_srednio": 0.15,
        "30_procent_duzo":   0.30,
    }

    for label, den in scenarios.items():
        # Reset ziarna dla każdego scenariusza → pełna reprodukowalność
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print(f"\n---> Scenariusz: {label} (Gęstość: {den * 100:.0f}%, Próby: {N_TESTS})")
        print("Generowanie weryfikowanych map (odrzucanie nierozwiązywalnych)...")

        envs = []
        valid_maps_count = 0
        attempts = 0

        while valid_maps_count < N_TESTS:
            attempts += 1
            env = GridMap(
                width=MAP_SIZE, height=MAP_SIZE,
                start_pos=START_POS, goal_pos=GOAL_POS,
                risk_zones_count=RISK_ZONES_COUNT,   # z config – ujednolicone
                obstacle_density=den
            )

            # Walidacja geometryczna (W=0, kara=0): najłatwiejszy wariant dla każdego algorytmu.
            # Jeśli którykolwiek nie znajdzie trasy przy zerowym ryzyku, mapa jest nierozwiązywalna
            # i nie ma sensu jej używać w benchmarku dla żadnej wagi W.
            _, sd = run_dijkstra(env, START_POS, GOAL_POS,
                                 risk_weight=0.0, turn_penalty=0.0, drone_radius=COLLISION_RADIUS)
            _, sa = run_astar(env, START_POS, GOAL_POS,
                              risk_weight=0.0, turn_penalty=0.0, drone_radius=COLLISION_RADIUS)
            _, sr = run_risk_astar(env, START_POS, GOAL_POS,
                                   risk_weight=0.0, turn_penalty=0.0, drone_radius=COLLISION_RADIUS)

            if sd['found'] and sa['found'] and sr['found']:
                envs.append(env)
                valid_maps_count += 1
                print(f"  [+] Mapa {valid_maps_count}/{N_TESTS} (po {attempts} próbach)")

        print(f"\nZakończono: {N_TESTS} map po {attempts} próbach łącznie.")
        print("Renderowanie wykresów...")

        generate_thesis_charts(
            envs, START_POS, GOAL_POS,
            run_dijkstra, run_astar, run_risk_astar,
            COLLISION_RADIUS, density_label=label,
            turn_penalty_classic=TURN_PENALTY_CLASSIC, turn_penalty_risk=TURN_PENALTY_RISK
        )

    print("\n" + "=" * 50)
    print(" PEŁNE BADANIE ZAKOŃCZONE!")
    print(" Wyniki w folderze 'research_results/'.")
    print("=" * 50)


def run_online_mode() -> None:
    print(f"\n Uruchomienie trybu dynamicznego ")
    density = get_user_difficulty()
    env = GridMap(
        width=MAP_SIZE, height=MAP_SIZE,
        start_pos=START_POS, goal_pos=GOAL_POS,
        risk_zones_count=RISK_ZONES_COUNT,
        obstacle_density=density
    )
    run_online_simulation(env, START_POS, GOAL_POS,
                          search_func=run_risk_astar,
                          collision_radius=COLLISION_RADIUS)


if __name__ == "__main__":
    main()
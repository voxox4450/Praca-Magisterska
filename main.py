from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import plot_simulation, plot_interactive_risk, run_online_simulation
from algorithms.common import generate_analysis_table
from typing import Tuple, List, Dict, Any
from visualization.plotter import generate_thesis_charts # Pamiętaj o imporcie!


def main() -> None:
    # Rozmiar mapy
    SIZE: int = 100
    DRONE_RADIUS: float = 1.0  # Promień drona w metrach
    SAFE_MARGIN: float = 2.0  # Dodatkowy margines od budynków

    # TEST_SCENARIOS = [
    #     {"name": "1. Klasyczna Przekątna", "start": (5, 5), "goal": (95, 95)},
    #     {"name": "2. Horyzontalna (Środek)", "start": (5, 50), "goal": (95, 50)},
    #     {"name": "3. Wertykalna (Środek)", "start": (50, 5), "goal": (50, 95)},
    #     {"name": "4. Krótki Dystans (Manewry)", "start": (40, 40), "goal": (60, 60)},
    #     {"name": "5. Z Narożnika do Centrum", "start": (5, 95), "goal": (50, 50)},
    #     {"name": "6. Odwrócona Przekątna", "start": (95, 95), "goal": (5, 5)},
    # ]
    start_pos = (5, 50)
    goal_pos = (95, 50)

    COLLISION_RADIUS = DRONE_RADIUS + SAFE_MARGIN

    while True:
        print(f"=" * 32)
        print(f"   SYSTEM PLANOWANIA TRAS BSP     ")
        print(f"=" * 32)
        print("WYBIERZ TRYB PRACY:")
        print("1. Wersja Statyczna (Offline) ")
        print("2. Wersja Dynamiczna (Online)  ")
        print("Naciśnij q aby zakończyć program")

        mode = input("\nTwój wybór 1 - Offline , 2 - Online , q - Opuść: ").strip()

        if mode == '1':
            run_offline_mode(SIZE, COLLISION_RADIUS, start_pos, goal_pos)
        elif mode == '2':
            run_online_mode(SIZE, COLLISION_RADIUS, start_pos, goal_pos)
        elif mode == 'q' or mode == 'Q':
            break
        else:
            print("Nieprawidłowy wybór")


def get_user_difficulty() -> float:
    """Wspólna funkcja wyboru trudności dla obu trybów."""
    print("\nWYBIERZ POZIOM TRUDNOŚCI OTOCZENIA:")
    print("1. Mało przeszkód (5%)")
    print("2. Średnio (15%)      ")
    print("3. Dużo przeszkód (30%)")

    while True:
        choice = input("Wybór (1-3): ").strip()
        if choice == '1':
            return 0.05
        elif choice == '2':
            return 0.15
        elif choice == '3':
            return 0.30


def run_offline_mode(size: int, collision_radius: float, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> None:
    print(f"\n Uruchomienie trybu statycznego ")

    # Wybór trudności
    density = get_user_difficulty()

    env = GridMap(width=size, height=size,start_pos = start_pos, goal_pos = goal_pos, risk_zones_count=8, obstacle_density=density)

    # 1. Dijkstra
    path_d, stats_d = run_dijkstra(env, start_pos, goal_pos)
    base_len = stats_d['length'] if stats_d['found'] else 0
    base_risk = stats_d['risk'] if stats_d['found'] else 0

    # 2. A* Standard
    path_a, stats_a = run_astar(env, start_pos, goal_pos)

    # 3. Generowanie tabeli analizy dla Risk A*
    generate_analysis_table(
        env=env,
        start_pos=start_pos,
        target_pos=goal_pos,
        search_func=run_risk_astar,
        base_len=base_len,
        base_risk=base_risk,
        collision_radius=collision_radius,
        table_title="ANALIZA TRYBU OFFLINE"
    )
    generate_thesis_charts(env, start_pos, goal_pos, run_risk_astar, collision_radius)

    print("\n Wizualizacja Djikstra, A* Standard oraz interaktywna mapa ryzyka dla Risk A*")

    if path_d:
        plot_simulation(env, path_d, stats_d, "1. Dijkstra (Referencja)", block=False, use_smoothing=False)
    if path_a:
        plot_simulation(env, path_a, stats_a, "2. A* Standard", block=False, use_smoothing=False)

    # Główne okno interaktywne
    plot_interactive_risk(env, start_pos, goal_pos, run_risk_astar)


def run_online_mode(size: int, collision_radius: float,
                    start_pos: Tuple[int, int],
                    goal_pos: Tuple[int, int],) -> None:
    print(f"\n uruchomienie trybu dynamicznego ")

    # Wybór trudności
    density = get_user_difficulty()

    # Tworzymy mapę z wybraną gęstością
    env = GridMap(width=size, height=size,start_pos = start_pos, goal_pos = goal_pos, risk_zones_count=8, obstacle_density=density)

    # Tabela utworzy się po utworzeniu dynamicznej strefy ryzyka.
    run_online_simulation(env, start_pos, goal_pos, search_func=run_risk_astar, collision_radius = collision_radius)


if __name__ == "__main__":
    main()
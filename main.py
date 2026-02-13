from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import plot_simulation, plot_interactive_risk, run_online_simulation
from algorithms.common import generate_analysis_table
from typing import Tuple, List, Dict, Any


def main() -> None:
    print(f"="*32)
    print(f"   SYSTEM PLANOWANIA TRAS BSP     ")
    print(f"="*32)
    print("WYBIERZ TRYB PRACY:")
    print("1. Wersja OFFLINE ")
    print("2. Wersja ONLINE ")

    mode = input("\nTwój wybór (1 lub 2): ").strip()

    # Rozmiar mapy
    SIZE: int = 100
    DRONE_RADIUS: float = 1.0  # Promień drona w metrach
    SAFE_MARGIN: float = 2.0  # Dodatkowy margines od budynków

    COLLISION_RADIUS = DRONE_RADIUS + SAFE_MARGIN

    if mode == '1':
        run_offline_mode(SIZE, COLLISION_RADIUS)
    elif mode == '2':
        run_online_mode(SIZE, COLLISION_RADIUS)
    else:
        print("Nieprawidłowy wybór. Uruchom ponownie.")


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


def run_offline_mode(size: int, collision_radius: float) -> None:
    """Tryb H1: Analiza statyczna (Tabela na start)"""
    print(f"\n=== URUCHAMIANIE TRYBU OFFLINE (H1) ===")

    # Wybór trudności
    density = get_user_difficulty()

    env = GridMap(width=size, height=size, risk_zones_count=8, obstacle_density=density)
    start_pos = (5, 5)
    goal_pos = (95, 95)

    print(f"\n[1] GENEROWANIE DANYCH DO TABELI (OBLICZENIA W TLE)...")

    # 1. Baza (Dijkstra)
    path_d, stats_d = run_dijkstra(env, start_pos, goal_pos)
    base_len = stats_d['length'] if stats_d['found'] else 0
    base_risk = stats_d['risk'] if stats_d['found'] else 0

    # 2. A* Standard
    path_a, stats_a = run_astar(env, start_pos, goal_pos)

    # 3. Generowanie tabeli analizy dla Risk A* (używamy wspólnej funkcji)
    generate_analysis_table(
        env=env,
        start_pos=start_pos,
        target_pos=goal_pos,
        search_func=run_risk_astar,
        base_len=base_len,
        base_risk=base_risk,
        collision_radius=collision_radius,
        table_title="ANALIZA TRYBU OFFLINE (H1)"
    )

    print("\n>>> OTWIERANIE OKIEN WIZUALIZACJI OFFLINE")

    if path_d:
        plot_simulation(env, path_d, stats_d, "1. Dijkstra (Referencja)", block=False, use_smoothing=False)
    if path_a:
        plot_simulation(env, path_a, stats_a, "2. A* Standard", block=False, use_smoothing=False)

    # Główne okno interaktywne
    plot_interactive_risk(env, start_pos, goal_pos, run_risk_astar)


def run_online_mode(size: int, collision_radius: float) -> None:
    """Tryb H3: Symulacja dynamiczna"""
    print(f"\n=== URUCHAMIANIE TRYBU ONLINE (H3) ===")

    # Wybór trudności
    density = get_user_difficulty()

    print("\n[INFO] URUCHAMIANIE WIZUALIZACJI ONLINE")

    # Tworzymy mapę z wybraną gęstością
    env = GridMap(width=size, height=size, risk_zones_count=8, obstacle_density=density)
    start_pos = (5, 5)
    goal_pos = (95, 95)

    # Uruchamiamy symulację (Tabela wygeneruje się wewnątrz tej funkcji po kliknięciu)
    run_online_simulation(env, start_pos, goal_pos, search_func=run_risk_astar, collision_radius = collision_radius)


if __name__ == "__main__":
    main()
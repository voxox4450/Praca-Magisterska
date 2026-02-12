import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar
from visualization.plotter import plot_simulation, plot_interactive_risk, run_online_simulation
from typing import Tuple, List, Dict, Any


def main() -> None:
    print(f"==========================================")
    print(f"   SYSTEM PLANOWANIA TRAS BSP (Mgr)      ")
    print(f"==========================================")
    print("WYBIERZ TRYB PRACY:")
    print("1. Wersja OFFLINE (Hipoteza H1 - Analiza Statyczna)")
    print("   - Generowanie tabeli (Wagi 0-100)")
    print("   - Porównanie wizualne")
    print("2. Wersja ONLINE (Hipoteza H3 - Dynamiczna Reakcja)")
    print("   - Symulacja lotu")
    print("   - Dodawanie zagrożeń kliknięciem")

    mode = input("\nTwój wybór (1 lub 2): ").strip()

    # Rozmiar mapy
    SIZE: int = 100

    if mode == '1':
        run_offline_mode(SIZE)
    elif mode == '2':
        run_online_mode(SIZE)
    else:
        print("Nieprawidłowy wybór. Uruchom ponownie.")


def run_offline_mode(size: int):
    """Tryb H1: Analiza wpływu wagi ryzyka na trasę"""
    print(f"\n=== URUCHAMIANIE TRYBU OFFLINE (H1) ===")

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

    # 3. Risk A* (Seria pomiarowa 0 - 100)
    # POPRAWKA: Zakres od 0 do 100 co 5
    risk_weights = [float(x) for x in range(0, 101, 5)]

    print("-" * 80)
    print(f"{'Waga (W)':<10} | {'Dystans':<10} | {'Koszt [%]':<10} | {'Ryzyko':<10} | {'Poprawa [%]':<12}")
    print("-" * 80)

    for w in risk_weights:
        _, stats = run_risk_astar(env, start_pos, goal_pos, risk_weight=w)

        if stats['found'] and base_len > 0:
            len_inc = ((stats['length'] - base_len) / base_len) * 100

            risk_red = 0.0
            if base_risk > 0:
                risk_red = ((base_risk - stats['risk']) / base_risk) * 100

            print(
                f"{w:<10.1f} | {stats['length']:<10.2f} | +{len_inc:<9.2f} | {stats['risk']:<10.2f} | -{risk_red:<11.2f}")
        else:
            print(f"{w:<10.1f} | BRAK TRASY")

    print("-" * 80)

    print("\n>>> OTWIERANIE OKIEN WIZUALIZACJI...")
    print("1. Dijkstra")
    print("2. A* Standard")
    print("3. Risk A* Interaktywny (Suwak)")
    print("ZAMKNIJ OKNO Z SUWAKIEM, ABY ZAKOŃCZYĆ.")

    if path_d:
        plot_simulation(env, path_d, stats_d, "1. Dijkstra (Referencja)", block=False, use_smoothing=False)
    if path_a:
        plot_simulation(env, path_a, stats_a, "2. A* Standard", block=False, use_smoothing=False)

    # Główne okno interaktywne
    plot_interactive_risk(env, start_pos, goal_pos, run_risk_astar)


def run_online_mode(size: int):
    """Tryb H3: Symulacja dynamiczna z klikaniem"""
    print(f"\n=== URUCHAMIANIE TRYBU ONLINE (H3) ===")
    print("Instrukcja:")
    print("1. Pojawi się okno z planowaną trasą.")
    print("2. KLIKNIJ na trasie (przed dronem), aby wywołać zagrożenie.")
    print("3. Obserwuj reakcję (replanowanie).")

    # Stała gęstość dla łatwiejszych testów online
    env = GridMap(width=size, height=size, risk_zones_count=8, obstacle_density=0.45)
    start_pos = (5, 5)
    goal_pos = (95, 95)

    run_online_simulation(env, start_pos, goal_pos, search_func=run_risk_astar)


def get_user_difficulty() -> float:
    print("\nWYBIERZ POZIOM TRUDNOŚCI (Dla Offline):")
    print("1. Mało przeszkód (5%)")
    print("2. Średnio (15%)")
    print("3. Dużo przeszkód (30%)")
    while True:
        c = input("Wybór: ").strip()
        if c == '1':
            return 0.05
        elif c == '2':
            return 0.15
        elif c == '3':
            return 0.30


if __name__ == "__main__":
    main()
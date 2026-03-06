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
    SIZE: int = 200
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
    start_pos = (5, 5)
    goal_pos = (195, 195)

    COLLISION_RADIUS = DRONE_RADIUS + SAFE_MARGIN

    while True:
        print(f"=" * 32)
        print(f"   SYSTEM PLANOWANIA TRAS BSP     ")
        print(f"=" * 32)
        print("WYBIERZ TRYB PRACY:")
        print("1. Wersja Statyczna (Offline) ")
        print("2. Wersja Dynamiczna (Online)  ")
        print("3. PEŁNE BADANIE (Automatyczny Benchmark 3 poziomów)")
        print("Naciśnij q aby zakończyć program")

        mode = input("\nTwój wybór 1 - Offline , 2 - Online , q - Opuść: ").strip()

        if mode == '1':
            density = get_user_difficulty()  # Najpierw pytamy o poziom trudności
            run_offline_mode(SIZE, COLLISION_RADIUS, start_pos, goal_pos, density=density, interactive=True)
        elif mode == '2':
            run_online_mode(SIZE, COLLISION_RADIUS, start_pos, goal_pos)
        elif mode == '3':
            run_batch_benchmark(SIZE, COLLISION_RADIUS, start_pos, goal_pos)
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


def run_offline_mode(size: int, collision_radius: float, start_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                     density: float, interactive: bool = True, density_label: str = "") -> None:
    if interactive:
        print(f"\n Uruchomienie trybu statycznego (Gęstość: {density * 100}%) ")

    env = GridMap(width=size, height=size, start_pos=start_pos, goal_pos=goal_pos, risk_zones_count=10,
                  obstacle_density=density)

    path_d, stats_d = run_dijkstra(env, start_pos, goal_pos, collision_radius=collision_radius)
    base_len = stats_d['length'] if stats_d['found'] else 0
    base_risk = stats_d['risk'] if stats_d['found'] else 0

    path_a, stats_a = run_astar(env, start_pos, goal_pos, collision_radius=collision_radius)

    generate_analysis_table(
        env=env, start_pos=start_pos, target_pos=goal_pos,
        search_func=run_risk_astar, base_len=base_len, base_risk=base_risk,
        collision_radius=collision_radius, table_title=f"ANALIZA (Gęstość: {density * 100}%)"
    )

    # --- ZMIANA: Przekazujemy env jako listę jednoelementową [env] ---
    generate_thesis_charts([env], start_pos, goal_pos, run_risk_astar, collision_radius, stats_d, stats_a,
                           density_label=density_label)

    if interactive:
        print("\n Wizualizacja Djikstra, A* Standard oraz interaktywna mapa ryzyka dla Risk A*")
        if path_d:
            plot_simulation(env, path_d, stats_d, "1. Dijkstra (Referencja)", block=False, use_smoothing=False)
        if path_a:
            plot_simulation(env, path_a, stats_a, "2. A* Standard", block=False, use_smoothing=False)
        plot_interactive_risk(env, start_pos, goal_pos, run_risk_astar)


def run_batch_benchmark(size: int, collision_radius: float, start_pos: Tuple[int, int],
                        goal_pos: Tuple[int, int]) -> None:
    print("\n" + "=" * 50)
    print(" ROZPOCZYNAM BADANIE BATCH BENCHMARK (MONTE CARLO)")
    print("=" * 50)

    scenarios = {
        "5_procent_malo": 0.05,
        "15_procent_srednio": 0.15,
        "30_procent_duzo": 0.30
    }

    # LICZBA MAP DO WYGENEROWANIA DLA KAŻDEGO SCENARIUSZA!
    # Na czas testów ustaw 5. Do ostatecznej pracy magisterskiej polecam zmienić na 20.
    N_TESTS = 15

    for label, den in scenarios.items():
        print(f"\n---> Przetwarzanie scenariusza: {label} (Gęstość: {den * 100}%, Próby: {N_TESTS})")
        print("Trwa generowanie map i obliczanie średnich dla klasycznych algorytmów (to może chwilę potrwać)...")

        envs = []
        d_sum = {'length': 0, 'risk': 0, 'flight_time': 0, 'turns': 0, 'found': 0}
        a_sum = {'length': 0, 'risk': 0, 'flight_time': 0, 'turns': 0, 'found': 0}

        for _ in range(N_TESTS):
            env = GridMap(width=size, height=size, start_pos=start_pos, goal_pos=goal_pos, risk_zones_count=10,
                          obstacle_density=den)
            envs.append(env)

            # Badanie Dijkstry na wygenerowanej mapie
            _, sd = run_dijkstra(env, start_pos, goal_pos, collision_radius=collision_radius)
            if sd['found']:
                d_sum['length'] += sd['length'];
                d_sum['risk'] += sd['risk']
                d_sum['flight_time'] += sd.get('flight_time', 0.0);
                d_sum['turns'] += sd.get('turns', 0)
                d_sum['found'] += 1

            # Badanie A* Standard na wygenerowanej mapie
            _, sa = run_astar(env, start_pos, goal_pos, collision_radius=collision_radius)
            if sa['found']:
                a_sum['length'] += sa['length'];
                a_sum['risk'] += sa['risk']
                a_sum['flight_time'] += sa.get('flight_time', 0.0);
                a_sum['turns'] += sa.get('turns', 0)
                a_sum['found'] += 1

        # Uśrednianie wyników z N map
        stats_d_avg = None
        if d_sum['found'] > 0:
            f = d_sum['found']
            stats_d_avg = {'length': d_sum['length'] / f, 'risk': d_sum['risk'] / f,
                           'flight_time': d_sum['flight_time'] / f, 'turns': d_sum['turns'] / f}

        stats_a_avg = None
        if a_sum['found'] > 0:
            f = a_sum['found']
            stats_a_avg = {'length': a_sum['length'] / f, 'risk': a_sum['risk'] / f,
                           'flight_time': a_sum['flight_time'] / f, 'turns': a_sum['turns'] / f}

        print("Rozpoczynam badanie Risk-Aware A* i renderowanie wykresów...")
        # Wywołanie generatora przekazując CAŁĄ LISTĘ MAP oraz uśrednione klasyczne algorytmy
        generate_thesis_charts(envs, start_pos, goal_pos, run_risk_astar, collision_radius, stats_d_avg, stats_a_avg,
                               density_label=label)

    print("\n" + "=" * 50)
    print(" PEŁNE BADANIE ZAKOŃCZONE SUKCESEM!")
    print(" Sprawdź uśrednione wykresy w folderze 'research_results/'.")
    print("=" * 50)


def run_online_mode(size: int, collision_radius: float, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> None:
    print(f"\n Uruchomienie trybu dynamicznego ")
    density = get_user_difficulty()
    env = GridMap(width=size, height=size, start_pos=start_pos, goal_pos=goal_pos, risk_zones_count=10,
                  obstacle_density=density)
    run_online_simulation(env, start_pos, goal_pos, search_func=run_risk_astar, collision_radius=collision_radius)


if __name__ == "__main__":
    main()
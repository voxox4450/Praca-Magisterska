import os
import random
import numpy as np
import matplotlib.pyplot as plt

from environment.grid_map import GridMap
from algorithms.dijkstra import run_dijkstra
from algorithms.a_star import run_astar
from algorithms.a_star_risk import run_risk_astar

from config import (
    MAP_SIZE, START_POS, GOAL_POS, RISK_WEIGHT,
    TURN_PENALTY, COLLISION_RADIUS, DRONE_MASS_KG, RANDOM_SEED
)


def create_grouped_bar_chart(data_dijkstra, data_astar, data_risk, labels, ylabel, title, filename, val_fmt='%.0f'):
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, data_dijkstra, width, label='Dijkstra', color='#4472C4')
    rects2 = ax.bar(x, data_astar, width, label='A* Standard', color='#ED7D31')
    rects3 = ax.bar(x + width, data_risk, width, label='Risk-Aware A*', color='#70AD47')

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d * 100)}%" for d in labels], fontsize=11)
    ax.set_xlabel('Gęstość zabudowy na mapie', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(val_fmt % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 punkty pionowego przesunięcia
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig(os.path.join('data', filename), dpi=300)
    plt.close()


def run_performance_benchmark():
    os.makedirs('data', exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    densities = [0.05, 0.15, 0.30]
    iterations = 20  # Liczba map generowanych dla każdego poziomu gęstości

    avg_times_dijkstra, avg_nodes_dijkstra = [], []
    avg_times_astar, avg_nodes_astar = [], []
    avg_times_risk, avg_nodes_risk = [], []

    print("\n" + "=" * 85)
    print(" BENCHMARK WYDAJNOŚCI OBLICZENIOWEJ (Dijkstra vs A* vs Risk-Aware A*)")
    print("=" * 85)
    print(f"{'Gęstość':<8} | {'Algorytm':<18} | {'Śr. Czas [s]':<15} | {'Śr. Liczba Węzłów':<15}")
    print("-" * 85)

    for density in densities:
        total_time_dijkstra, total_nodes_dijkstra = 0.0, 0
        total_time_astar, total_nodes_astar = 0.0, 0
        total_time_risk, total_nodes_risk = 0.0, 0
        valid_iterations = 0

        for i in range(iterations):
            current_seed = RANDOM_SEED + i
            random.seed(current_seed)
            np.random.seed(current_seed)

            env = GridMap(
                width=MAP_SIZE, height=MAP_SIZE,
                start_pos=START_POS, goal_pos=GOAL_POS,
                risk_zones_count=5, obstacle_density=density
            )

            # --- Test 1: Dijkstra ---
            path_d, stats_d = run_dijkstra(
                grid_map=env, start=START_POS, goal=GOAL_POS,
                risk_weight=RISK_WEIGHT, turn_penalty=TURN_PENALTY,
                drone_radius=COLLISION_RADIUS, initial_direction=(0, 0),
                current_speed=0.0, drone_mass=DRONE_MASS_KG
            )

            # --- Test 2: Klasyczny A* ---
            path_a, stats_a = run_astar(
                grid_map=env, start=START_POS, goal=GOAL_POS,
                risk_weight=RISK_WEIGHT, turn_penalty=TURN_PENALTY,
                drone_radius=COLLISION_RADIUS, initial_direction=(0, 0),
                current_speed=0.0, drone_mass=DRONE_MASS_KG
            )

            # --- Test 3: Risk-Aware A* ---
            path_r, stats_r = run_risk_astar(
                grid_map=env, start=START_POS, goal=GOAL_POS,
                risk_weight=RISK_WEIGHT, turn_penalty=TURN_PENALTY,
                drone_radius=COLLISION_RADIUS, initial_direction=(0, 0),
                current_speed=0.0, initial_straight_dist=0.0, drone_mass=DRONE_MASS_KG
            )

            # Bierzemy pod uwagę tylko te mapy, gdzie wszystkie 3 algorytmy znalazły cel
            if (path_d and path_a and path_r and
                    stats_d.get('found') and stats_a.get('found') and stats_r.get('found')):
                total_time_dijkstra += stats_d.get('time', 0)
                total_nodes_dijkstra += stats_d.get('nodes', 0)

                total_time_astar += stats_a.get('time', 0)
                total_nodes_astar += stats_a.get('nodes', 0)

                total_time_risk += stats_r.get('time', 0)
                total_nodes_risk += stats_r.get('nodes', 0)

                valid_iterations += 1

        if valid_iterations > 0:
            avg_times_dijkstra.append(total_time_dijkstra / valid_iterations)
            avg_nodes_dijkstra.append(total_nodes_dijkstra // valid_iterations)

            avg_times_astar.append(total_time_astar / valid_iterations)
            avg_nodes_astar.append(total_nodes_astar // valid_iterations)

            avg_times_risk.append(total_time_risk / valid_iterations)
            avg_nodes_risk.append(total_nodes_risk // valid_iterations)

            print(
                f"{density * 100:>5.0f}%   | {'Dijkstra':<18} | {avg_times_dijkstra[-1]:>13.2f} | {avg_nodes_dijkstra[-1]:>15}")
            print(f"         | {'A* Standard':<18} | {avg_times_astar[-1]:>13.2f} | {avg_nodes_astar[-1]:>15}")
            print(f"         | {'Risk-Aware A*':<18} | {avg_times_risk[-1]:>13.2f} | {avg_nodes_risk[-1]:>15}")
            print("-" * 85)
        else:
            print(f"{density * 100:>5.0f}%   | Zbyt trudna mapa w wyznaczonych iteracjach.")

    print("=" * 85)
    print("Generowanie wykresów do katalogu /data...")

    create_grouped_bar_chart(
        data_dijkstra=avg_times_dijkstra,
        data_astar=avg_times_astar,
        data_risk=avg_times_risk,
        labels=densities,
        ylabel='Czas planowania [s]',
        title='Porównanie czasu planowania trasy w zależności od gęstości zabudowy',
        filename='benchmark_time.png',
        val_fmt='%.2f'
    )

    create_grouped_bar_chart(
        data_dijkstra=avg_nodes_dijkstra,
        data_astar=avg_nodes_astar,
        data_risk=avg_nodes_risk,
        labels=densities,
        ylabel='Liczba odwiedzonych węzłów',
        title='Porównanie liczby eksplorowanych węzłów w zależności od gęstości zabudowy',
        filename='benchmark_nodes.png',
        val_fmt='%d'
    )

    print("Zakończono! Wykresy 'benchmark_time.png' i 'benchmark_nodes.png' zapisane w folderze 'data'.")


if __name__ == "__main__":
    run_performance_benchmark()
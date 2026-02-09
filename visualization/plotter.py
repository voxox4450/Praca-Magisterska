import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # <--- KLUCZOWA POPRAWKA (Import efektów ścieżki)
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Tuple, Callable, Any
from environment.grid_map import GridMap


def get_city_cmap():
    """
    Tworzy paletę dedykowaną dla miasta:
    - 0.0      -> Biały (Bezpiecznie)
    - 0.1..0.9 -> Odcienie Czerwieni (Ryzyko - chodniki, tłum)
    - 1.0      -> Czarny (Budynek/Zakaz lotu)
    """
    colors = [
        (0.0, (1.0, 1.0, 1.0)),  # 0.0 -> Biały
        (0.01, (1.0, 0.9, 0.9)),  # 0.01 -> Bardzo blady czerwony
        (0.99, (0.8, 0.0, 0.0)),  # 0.99 -> Mocny Czerwony (Wysokie ryzyko)
        (0.991, (0.0, 0.0, 0.0)),  # Ostry przeskok na Czarny
        (1.0, (0.0, 0.0, 0.0))  # 1.0 -> Czarny (Ściana)
    ]
    return LinearSegmentedColormap.from_list("CityMap", colors)


def plot_simulation(
        grid_map: GridMap,
        path: List[Tuple[int, int]],
        title: str = "Simulation",
        block: bool = True
) -> None:
    plt.figure(figsize=(10, 8))

    # Rysowanie mapy z paletą miejską
    plt.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    cbar = plt.colorbar(label='Legenda')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko (Ludzie)', 'BUDYNEK (Czarny)'])

    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]

        # Rysowanie trasy z obrysem (używamy zaimportowanego 'pe')
        plt.plot(path_x, path_y, color='cyan', linewidth=2.5, label='Trasa',
                 path_effects=[pe.withStroke(linewidth=4, foreground="blue")])

        plt.scatter([path_x[0]], [path_y[0]], color='lime', marker='o', s=120, label='Start', edgecolors='black')
        plt.scatter([path_x[-1]], [path_y[-1]], color='magenta', marker='X', s=120, label='Cel', edgecolors='black')

    plt.title(title)
    plt.legend(loc='upper right')
    plt.show(block=block)


def plot_interactive_risk(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable
) -> Slider:
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    ax.set_title("Interaktywna analiza: Zmieniaj wagę ryzyka suwakiem")

    # Inicjalizacja pustej linii z efektem obrysu
    line, = ax.plot([], [], color='cyan', linewidth=2.5, label='Trasa',
                    path_effects=[pe.withStroke(linewidth=4, foreground="blue")])

    ax.scatter([start[0]], [start[1]], color='lime', s=120, label='Start', edgecolors='black')
    ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=120, label='Cel', edgecolors='black')
    ax.legend()

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    risk_slider = Slider(
        ax=ax_slider,
        label='Waga Ryzyka (W)',
        valmin=0.0,
        valmax=100.0,
        valinit=20.0,
        valstep=1.0
    )

    def update(val):
        w = risk_slider.val
        path, stats = search_func(grid_map, start, goal, risk_weight=w, turn_penalty=2.0)

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            line.set_data(px, py)
            ax.set_title(f"Waga: {w:.0f} | Długość: {stats['length']:.1f} | Ryzyko: {stats['risk']:.1f}")
        else:
            line.set_data([], [])
            ax.set_title("Brak trasy!")
        fig.canvas.draw_idle()

    risk_slider.on_changed(update)
    update(20.0)  # Pierwsze wywołanie
    plt.show(block=True)
    return risk_slider
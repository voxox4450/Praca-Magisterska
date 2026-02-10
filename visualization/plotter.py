import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from environment.grid_map import GridMap


def get_city_cmap():
    """Paleta: Biały -> Czerwony -> Czarny"""
    colors = [
        (0.0, (1.0, 1.0, 1.0)),
        (0.01, (1.0, 0.9, 0.9)),
        (0.99, (0.8, 0.0, 0.0)),
        (0.991, (0.0, 0.0, 0.0)),
        (1.0, (0.0, 0.0, 0.0))
    ]
    return LinearSegmentedColormap.from_list("CityMap", colors)


def setup_dark_theme(fig, ax):
    """Ustawia ciemny motyw okna i wykresu."""
    fig.patch.set_facecolor('#1e1e1e')  # Ciemnoszare tło okna
    ax.set_facecolor('#1e1e1e')  # Ciemnoszare tło wykresu
    ax.tick_params(colors='white')  # Białe liczby na osiach
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # Ramki wykresu na biało
    for spine in ax.spines.values():
        spine.set_edgecolor('white')


def plot_simulation(
        grid_map: GridMap,
        path: List[Tuple[int, int]],
        stats: Dict[str, Any],
        algo_name: str,
        block: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    # Marginesy: right=0.75 zostawia 25% miejsca z prawej strony
    plt.subplots_adjust(right=0.75, left=0.05, bottom=0.1, top=0.9)
    setup_dark_theme(fig, ax)

    # Mapa
    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # === 1. LEGENDA RYZYKA (COLORBAR) ===
    # shrink=0.8 -> Pasek zajmuje 80% (4/5) dostępnej wysokości
    # anchor=(0.0, 1.0) -> Pasek jest "przyklejony" do górnej krawędzi wykresu
    cbar = fig.colorbar(img, ax=ax, location='right', pad=0.05, shrink=0.8, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])

    # Stylizacja colorbara
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    # Rysowanie trasy
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, color='cyan', linewidth=3, label='Trasa',
                path_effects=[pe.withStroke(linewidth=5, foreground="blue")])
        ax.scatter([path_x[0]], [path_y[0]], color='lime', marker='o', s=150, label='Start', edgecolors='black',
                   zorder=5)
        ax.scatter([path_x[-1]], [path_y[-1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                   zorder=5)

    # === 2. LEGENDA ELEMENTÓW (Dół-Prawo) ===
    # bbox_to_anchor=(1.04, 0.0) -> Ustawia legendę pod spodem, na równi z dolną osią wykresu
    legend = ax.legend(
        loc='lower left',
        bbox_to_anchor=(1.04, 0.0),
        facecolor='#333333',
        edgecolor='white',
        title="Elementy Mapy"
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Tytuł
    title_text = (f"{algo_name}\n"
                  f"Długość: {stats['length']:.2f} m | Ryzyko: {stats['risk']:.2f} | Czas: {stats['time']:.4f} s")
    ax.set_title(title_text, fontsize=14, pad=15)

    plt.show(block=block)


def plot_interactive_risk(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable
) -> Slider:
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.20, right=0.75, left=0.05, top=0.9)
    setup_dark_theme(fig, ax)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # === 1. COLORBAR (4/5 wysokości) ===
    cbar = fig.colorbar(img, ax=ax, location='right', pad=0.05, shrink=0.8, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # Trasa
    line, = ax.plot([], [], color='cyan', linewidth=3, label='Trasa',
                    path_effects=[pe.withStroke(linewidth=5, foreground="blue")])
    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)

    # === 2. LEGENDA ELEMENTÓW ===
    legend = ax.legend(
        loc='lower left',
        bbox_to_anchor=(1.04, 0.0),
        facecolor='#333333',
        edgecolor='white',
        title="Elementy Mapy"
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Suwak
    ax_slider = plt.axes([0.15, 0.05, 0.6, 0.03], facecolor='#333333')
    risk_slider = Slider(
        ax=ax_slider,
        label='Waga Ryzyka (W)',
        valmin=0.0,
        valmax=100.0,
        valinit=20.0,
        valstep=1.0,
        color='cyan'
    )
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    def update(val):
        w = risk_slider.val
        path, stats = search_func(grid_map, start, goal, risk_weight=w, turn_penalty=2.0)

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            line.set_data(px, py)

            title_text = (f"A* Risk-Aware (Interaktywny)\n"
                          f"Waga: {w:.0f} | Długość: {stats['length']:.1f} | Ryzyko: {stats['risk']:.1f} | Czas: {stats['time']:.4f} s")
            ax.set_title(title_text, fontsize=14)
        else:
            line.set_data([], [])
            ax.set_title("Brak trasy!", color='red')
        fig.canvas.draw_idle()

    risk_slider.on_changed(update)
    update(20.0)

    print("Otwieranie okna interaktywnego")
    plt.show(block=True)
    return risk_slider

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.interpolate as interp
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


def smooth_path_bspline(path: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Wygładzanie trasy B-Spline."""
    if len(path) < 3:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        return np.array(px), np.array(py)

    x = [p[0] for p in path]
    y = [p[1] for p in path]

    try:
        tck, u = interp.splprep([x, y], s=3.0, k=3)
        u_new = np.linspace(0, 1, num=len(path) * 10)
        x_smooth, y_smooth = interp.splev(u_new, tck)
        return x_smooth, y_smooth
    except Exception:
        return np.array(x), np.array(y)


def plot_simulation(
        grid_map: GridMap,
        path: List[Tuple[int, int]],
        stats: Dict[str, Any],
        algo_name: str,
        block: bool = True,
        use_smoothing: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(right=0.75, left=0.05, bottom=0.1, top=0.9)
    setup_dark_theme(fig, ax)

    # Mapa
    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # === 1. LEGENDA RYZYKA (PRZYWRÓCONA) ===
    cbar = fig.colorbar(img, ax=ax, location='right', pad=0.05, shrink=0.80, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    # Rysowanie trasy
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]

        if use_smoothing:
            # DLA RISK A*: Dwie linie (szary plan + cyjanowa trajektoria)
            ax.plot(path_x, path_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            smooth_x, smooth_y = smooth_path_bspline(path)
            ax.plot(smooth_x, smooth_y, color='cyan', linewidth=3, label='Trajektoria (Smooth)',
                    path_effects=[pe.withStroke(linewidth=5, foreground="blue")])
        else:
            # DLA DIJKSTRA / STANDARD A*: Tylko jedna, kanciasta linia (Bezpieczna)
            ax.plot(path_x, path_y, color='cyan', linewidth=3, label='Trasa',
                    path_effects=[pe.withStroke(linewidth=5, foreground="blue")])

        # Start i Meta
        ax.scatter([path_x[0]], [path_y[0]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
        ax.scatter([path_x[-1]], [path_y[-1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                   zorder=5)

    # === 2. LEGENDA ELEMENTÓW (PRZYWRÓCONA NA DOLE) ===
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
                  f"Dystans: {stats['length']:.1f} m | Ryzyko: {stats['risk']:.1f}")
    ax.set_title(title_text, fontsize=14, pad=15)

    plt.show(block=block)


def plot_interactive_risk(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable
) -> Slider:
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.20, right=0.65, left=0.05, top=0.9)
    setup_dark_theme(fig, ax)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # === 1. COLORBAR (PRZYWRÓCONY) ===
    cbar = fig.colorbar(img, ax=ax, location='right', pad=0.05, shrink=0.80, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # Trasa
    line_raw, = ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    line_smooth, = ax.plot([], [], color='cyan', linewidth=3, label='Trasa',
                           path_effects=[pe.withStroke(linewidth=4, foreground="blue")])

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)

    # === 2. LEGENDA ELEMENTÓW (PRZYWRÓCONA) ===
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
        # Tutaj wywołujemy algorytm
        path, stats = search_func(grid_map, start, goal, risk_weight=w, turn_penalty=20.0)

        if path:
            # Plan surowy
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            line_raw.set_data(px, py)

            # Wygładzanie
            sx, sy = smooth_path_bspline(path)
            line_smooth.set_data(sx, sy)

            title_text = (f"A* Risk-Aware (Interaktywny)\n"
                          f"Waga: {w:.0f} | Długość: {stats['length']:.1f} | Ryzyko: {stats['risk']:.1f}")
            ax.set_title(title_text, fontsize=14)
        else:
            line_raw.set_data([], [])
            line_smooth.set_data([], [])
            ax.set_title("Brak trasy!", color='red')
        fig.canvas.draw_idle()

    risk_slider.on_changed(update)
    update(20.0)

    plt.show(block=True)
    return risk_slider


def run_online_simulation(
        env: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable
) -> None:
    """
    H3: Symulacja Online.
    Logika Fallback:
    1. Próba ominięcia przeszkody i lotu do CELU.
    2. Jeśli cel nieosiągalny -> Automatyczny powrót do STARTU (RTH).
    3. Jeśli start też nieosiągalny -> Dopiero wtedy Błąd Krytyczny.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    plt.subplots_adjust(right=0.75, left=0.05, bottom=0.1, top=0.9)
    setup_dark_theme(fig, ax)

    ax.set_title("TRYB ONLINE (H3): Kliknij na trasie, aby dodać przeszkodę!", color='yellow', fontsize=14)

    # Mapa
    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.046, pad=0.045, shrink=0.75, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    # Trasa Startowa
    path_global, _ = search_func(env, start, goal, risk_weight=20.0)

    if not path_global:
        print("Błąd: Nie znaleziono trasy startowej.")
        return

    gx_smooth, gy_smooth = smooth_path_bspline(path_global)

    # Elementy wykresu
    line_global, = ax.plot(gx_smooth, gy_smooth, color='black', linestyle='--', linewidth=2.5, alpha=0.8,
                           label='Pierwotny Plan')
    line_flown, = ax.plot([], [], color='lime', linewidth=3, label='Droga Przebyta', zorder=3)
    line_new, = ax.plot([], [], color='cyan', linewidth=3, label='Replanowana Trasa',
                        path_effects=[pe.withStroke(linewidth=5, foreground="blue")], zorder=4)

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    goal_marker = ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                             zorder=5)
    drone_marker, = ax.plot([], [], 'o', color='yellow', markersize=12, label='Wykrycie (5m)', markeredgecolor='black',
                            zorder=6)

    # Legenda
    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.04, 0.0), facecolor='#333333', edgecolor='white',
                       title="Legenda H3")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    state = {"clicked": False}

    def onclick(event):
        if state["clicked"] or event.xdata is None or event.ydata is None:
            return

        click_x, click_y = int(event.xdata), int(event.ydata)
        print(f"\n[ONLINE] Kliknięcie w: ({click_x}, {click_y})")

        # 1. Dodajemy przeszkodę
        OBSTACLE_RADIUS = 8
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        # Logika 5m
        SENSOR_RANGE = 5
        LIMIT_DIST = OBSTACLE_RADIUS + SENSOR_RANGE

        collision_idx = -1
        for i, (px, py) in enumerate(path_global):
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= LIMIT_DIST:
                collision_idx = i
                break

        if collision_idx == -1:
            print("[ONLINE] Zagrożenie daleko. Ignoruję.")
            ax.set_title("Zagrożenie poza zasięgiem (>5m).\nKontynuuję pierwotną trasę.", color='lime', fontsize=14)
            line_flown.set_data(gx_smooth, gy_smooth)
            state["clicked"] = True
            fig.canvas.draw()
            return

        drone_idx = max(0, collision_idx - SENSOR_RANGE)
        current_drone_pos = path_global[drone_idx]

        # Sprawdzenie czy dron żyje
        dist_to_drone = np.sqrt((click_x - current_drone_pos[0]) ** 2 + (click_y - current_drone_pos[1]) ** 2)
        if dist_to_drone <= OBSTACLE_RADIUS:
            print("[ONLINE] ALERT: Dron zniszczony.")
            ax.set_title("POZYCJA ZAGROŻONA! Dron zniszczony.", color='red', fontsize=14)
            drone_marker.set_data([current_drone_pos[0]], [current_drone_pos[1]])
            state["clicked"] = True
            fig.canvas.draw()
            return

        # ======================================================================
        # STRATEGIA REPLANOWANIA (FALLBACK)
        # ======================================================================
        print(f"[ONLINE] Replanowanie z {current_drone_pos} do CELU {goal}")

        # KROK 1: Próba lotu do CELU (Meta)
        path_to_goal, _ = search_func(env, current_drone_pos, goal, risk_weight=50.0)

        if path_to_goal:
            # SUKCES: Omijamy przeszkodę i lecimy do celu
            flown_raw = path_global[:drone_idx + 1]
            if len(flown_raw) > 2:
                fx, fy = smooth_path_bspline(flown_raw)
            else:
                fx = [p[0] for p in flown_raw]
                fy = [p[1] for p in flown_raw]
            line_flown.set_data(fx, fy)

            nx, ny = smooth_path_bspline(path_to_goal)
            line_new.set_data(nx, ny)

            drone_marker.set_data([current_drone_pos[0]], [current_drone_pos[1]])
            ax.set_title("ZAGROŻENIE OMINIĘTE! Lot do celu.", color='lime', fontsize=14)

        else:
            # PORAŻKA KROKU 1: Cel zablokowany -> Próba powrotu do STARTU
            print("[ONLINE] Cel nieosiągalny! Próba powrotu do STARTU...")
            path_rth, _ = search_func(env, current_drone_pos, start, risk_weight=20.0)

            if path_rth:
                # SUKCES RTH: Wracamy do bazy
                flown_raw = path_global[:drone_idx + 1]
                if len(flown_raw) > 2:
                    fx, fy = smooth_path_bspline(flown_raw)
                else:
                    fx = [p[0] for p in flown_raw]
                    fy = [p[1] for p in flown_raw]
                line_flown.set_data(fx, fy)

                nx, ny = smooth_path_bspline(path_rth)
                line_new.set_data(nx, ny)
                line_new.set_color('orange')
                line_new.set_label('Powrót (Awaryjny)')

                # Aktualizacja legendy
                ax.legend(loc='lower left', bbox_to_anchor=(1.04, 0.0), facecolor='#333333', edgecolor='white')

                drone_marker.set_data([current_drone_pos[0]], [current_drone_pos[1]])
                ax.set_title("CEL ZABLOKOWANY! Awaryjny powrót do bazy.", color='orange', fontsize=14)
                goal_marker.set_facecolor('gray')  # Cel nieaktywny
            else:
                # PORAŻKA CAŁKOWITA: Nie da się wrócić (bardzo rzadkie)
                ax.set_title("BŁĄD KRYTYCZNY: Uwięziony!", color='red', fontsize=14)

        state["clicked"] = True
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
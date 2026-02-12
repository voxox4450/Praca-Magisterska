import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.interpolate as interp
import math
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
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')


def smooth_path_bspline(path: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
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


# Funkcje pomocnicze do tabeli w trybie Online
def calculate_segment_risk(path: List[Tuple[int, int]], env: GridMap) -> float:
    total_risk = 0.0
    for (x, y) in path:
        total_risk += env.get_cost(x, y)
    return total_risk


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    length = 0.0
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        length += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length


# Funkcje wizualizacji Offline (zostawiamy bez zmian dla kompatybilności z main.py)
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
    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    cbar = fig.colorbar(img, ax=ax, location='right', pad=0.05, shrink=0.80, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        if use_smoothing:
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

    turns_count = stats.get('turns', 0)

    title_text = (f"{algo_name}\n"
                  f"Dystans: {stats['length']:.1f} m | Ryzyko: {stats['risk']:.1f} | Czas: {stats['time']:.4f} s | Zakręty: {turns_count}")
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
    cbar = fig.colorbar(img, ax=ax, location='right', pad=0.05, shrink=0.80, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
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
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            line_raw.set_data(px, py)
            sx, sy = smooth_path_bspline(path)
            line_smooth.set_data(sx, sy)
            turns_count = stats.get('turns', 0)
            title_text = (f"A* Risk-Aware (W={w:.0f})\n"
                          f"Dyst: {stats['length']:.1f} m | Ryzyko: {stats['risk']:.1f} | "
                          f"Czas: {stats['time']:.4f} s | Zakręty: {turns_count}")
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


# --- GŁÓWNA FUNKCJA DLA H3 (POPRAWIONA) ---
def run_online_simulation(
        env: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable,
        collision_radius: float = 3.0
) -> None:
    """
    H3: Symulacja Online z SUWAKIEM.
    Poprawki:
    1. Precyzyjne wykrywanie (14m dystans).
    2. Tabela ze znakami +/-.
    3. Suwak startuje od 20.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(right=0.75, left=0.05, bottom=0.20, top=0.9)
    setup_dark_theme(fig, ax)

    ax.set_title("TRYB ONLINE: Kliknij na trasie, aby dodać przeszkodę!", color='yellow', fontsize=14)

    # Mapa
    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.046, pad=0.045, shrink=0.70, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    # Trasa Startowa (W=20)
    path_global, _ = search_func(env, start, goal, risk_weight=20.0, turn_penalty=20.0, drone_radius=collision_radius)

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

    # Legenda (Przyklejona do dołu)
    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.04, 0.0), facecolor='#333333', edgecolor='white',
                       title="Legenda H3")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Suwak (Start od 20.0)
    ax_slider = plt.axes([0.20, 0.05, 0.50, 0.03], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W) ', valmin=0.0, valmax=100.0, valinit=20.0, valstep=1.0,
                         color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    sim_state = {"clicked": False, "drone_pos": None, "target_pos": None, "mode": "IDLE"}

    def update_route(val):
        if not sim_state["clicked"] or sim_state["mode"] in ["CRASH", "IGNORE", "IDLE"]: return
        w = risk_slider.val
        path_local, stats = search_func(env, sim_state["drone_pos"], sim_state["target_pos"], risk_weight=w, turn_penalty=20.0, drone_radius=collision_radius)

        if path_local:
            nx, ny = smooth_path_bspline(path_local)
            line_new.set_data(nx, ny)

            # --- ZMIANA: Pełne statystyki w tytule (Dystans, Ryzyko, Czas, Zakręty) ---
            turns_count = stats.get('turns', 0)
            stats_text = (f"Dyst: {stats['length']:.1f}m | Ryzyko: {stats['risk']:.1f} | "
                          f"Czas: {stats['time']:.4f}s | Zakręty: {turns_count}")
            if sim_state["mode"] == "RTH":
                # Dodano \n{stats_text}
                ax.set_title(f"TRYB POWROTU (RTH) | W={w:.0f}\n{stats_text}", color='orange', fontsize=14)
                line_new.set_color('orange')
            else:
                # Dodano \n{stats_text}
                ax.set_title(f"OMIJANIE PRZESZKODY | W={w:.0f}\n{stats_text}", color='lime', fontsize=14)
                line_new.set_color('cyan')
        else:
            ax.set_title(f"DLA W={w:.0f} DRON JEST UWIĘZIONY!", color='red', fontsize=14)
            line_new.set_data([], [])
        fig.canvas.draw_idle()

    risk_slider.on_changed(update_route)

    def onclick(event):
        if event.inaxes != ax: return
        if sim_state["clicked"]: return

        click_x, click_y = int(event.xdata), int(event.ydata)
        print(f"\n[ONLINE] Kliknięcie w: ({click_x}, {click_y})")

        # 1. Dodajemy przeszkodę
        OBSTACLE_RADIUS = 8
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        # ---------------------------------------------------------
        # NOWA LOGIKA WYKRYWANIA (Matematyka: 14 metrów)
        # ---------------------------------------------------------
        DRONE_RADIUS = 1.0  # Promień drona
        SENSOR_RANGE = 5.0  # Zasięg czujnika

        # Dystans graniczny = Promień Przeszkody + Promień Drona + Zasięg Czujnika
        # = 8 + 1 + 5 = 14.0
        LIMIT_DIST = OBSTACLE_RADIUS + DRONE_RADIUS + SENSOR_RANGE

        collision_idx = -1
        # Szukamy PIERWSZEGO punktu, gdzie dron widzi przeszkodę
        for i, (px, py) in enumerate(path_global):
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= LIMIT_DIST:
                collision_idx = i
                break

        if collision_idx == -1:
            print("[ONLINE] Zagrożenie daleko.")
            ax.set_title("Zagrożenie poza zasięgiem (>5m).", color='lime', fontsize=14)
            line_flown.set_data(gx_smooth, gy_smooth)
            sim_state["clicked"] = True
            sim_state["mode"] = "IGNORE"
            fig.canvas.draw()
            return

        drone_idx = collision_idx  # To jest pozycja drona W MOMENCIE wykrycia
        current_drone_pos = path_global[drone_idx]

        # Rysujemy przebytą
        flown_raw = path_global[:drone_idx + 1]
        if len(flown_raw) > 2:
            fx, fy = smooth_path_bspline(flown_raw)
        else:
            fx = [p[0] for p in flown_raw]
            fy = [p[1] for p in flown_raw]
        line_flown.set_data(fx, fy)
        drone_marker.set_data([current_drone_pos[0]], [current_drone_pos[1]])

        # Sprawdzenie zniszczenia (Czy dron już siedzi w ogniu?)
        # Jeśli środek drona jest bliżej niż (Promień przeszkody + Promień drona)
        dist_to_drone = np.sqrt((click_x - current_drone_pos[0]) ** 2 + (click_y - current_drone_pos[1]) ** 2)
        if dist_to_drone <= (OBSTACLE_RADIUS + DRONE_RADIUS):
            ax.set_title("DRON ZNISZCZONY!", color='red', fontsize=14)
            sim_state["clicked"] = True
            sim_state["mode"] = "CRASH"
            fig.canvas.draw()
            return

        # DECYZJA
        path_check, _ = search_func(env, current_drone_pos, goal, risk_weight=20.0, turn_penalty=20.0, drone_radius=collision_radius)

        sim_state["drone_pos"] = current_drone_pos
        sim_state["clicked"] = True

        if path_check:
            print("[ONLINE] Cel osiągalny.")
            sim_state["target_pos"] = goal
            sim_state["mode"] = "NORMAL"
            line_new.set_label('Replanowana Trasa')
        else:
            print("[ONLINE] Cel zablokowany -> RTH.")
            sim_state["target_pos"] = start
            sim_state["mode"] = "RTH"
            goal_marker.set_facecolor('gray')
            line_new.set_label('Powrót (Awaryjny)')
            ax.legend(loc='lower left', bbox_to_anchor=(1.04, 0.0), facecolor='#333333', edgecolor='white')

        update_route(risk_slider.val)

        # Tabela w konsoli
        print("\n--- GENEROWANIE TABELI ANALIZY (W KONSOLI) ---")
        path_remainder = path_global[drone_idx:]
        base_risk = calculate_segment_risk(path_remainder, env)
        base_len = calculate_path_length(path_remainder)

        print("-" * 90)
        print(f"Baza (Bez reakcji): Dystans: {base_len:.2f} | Ryzyko: {base_risk:.2f}")
        print("-" * 90)
        print(f"{'Waga (W)':<10} | {'Dystans':<10} | {'Koszt [%]':<10} | {'Ryzyko':<10} | {'Zmiana Ryzyka [%]':<20}")
        print("-" * 90)

        for w_test in range(0, 101, 5):
            p_test, s_test = search_func(env, current_drone_pos, sim_state["target_pos"], risk_weight=float(w_test), turn_penalty=20.0, drone_radius=collision_radius)
            if s_test['found']:
                # Obliczamy procentową zmianę
                if base_risk > 0:
                    pct_change = ((s_test['risk'] - base_risk) / base_risk) * 100
                else:
                    pct_change = 0.0

                cost_inc = ((s_test['length'] - base_len) / base_len) * 100

                # Używamy formatowania {:+.1f}, które samo doda "+" dla wzrostu i "-" dla spadku
                print(
                    f"{w_test:<10} | {s_test['length']:<10.2f} | +{cost_inc:<9.1f} | {s_test['risk']:<10.1f} | {pct_change:<+19.1f}")
            else:
                print(f"{w_test:<10} | BRAK TRASY")
        print("-" * 90)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
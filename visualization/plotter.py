import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import numpy as np
import scipy.interpolate as interp
import math
from typing import List, Tuple, Callable, Any, Dict
from environment.grid_map import GridMap
from algorithms.common import calculate_segment_risk, calculate_path_length, generate_analysis_table, calculate_kinematic_flight_time
import os


def _plot_benchmark_bars(bench_data: dict, title: str, filename: str, w_label: str) -> None:
    """Funkcja pomocnicza do rysowania ustandaryzowanych 4 wykresów słupkowych."""
    labels = [f'Dijkstra\n({w_label})', f'A* Standard\n({w_label})', f'Risk-Aware A*\n({w_label})']
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    dist_vals = [bench_data['d_len'], bench_data['a_len'], bench_data['r_len']]
    risk_vals = [bench_data['d_risk'], bench_data['a_risk'], bench_data['r_risk']]
    time_vals = [bench_data['d_fl'], bench_data['a_fl'], bench_data['r_fl']]
    turns_vals = [bench_data['d_trn'], bench_data['a_trn'], bench_data['r_trn']]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    axs[0, 0].bar(labels, dist_vals, color=colors, edgecolor='black', alpha=0.9)
    axs[0, 0].set_title("1. Długość Trasy (Geometryczna) [m]", fontsize=13)
    axs[0, 0].set_ylabel("Metry")

    axs[0, 1].bar(labels, risk_vals, color=colors, edgecolor='black', alpha=0.9)
    axs[0, 1].set_title("2. Poziom Ekspozycji na Ryzyko", fontsize=13)
    axs[0, 1].set_ylabel("Wartość Ryzyka")

    axs[1, 0].bar(labels, time_vals, color=colors, edgecolor='black', alpha=0.9)
    axs[1, 0].set_title("3. Fizyczny Czas Przelotu (Kinematyka) [s]", fontsize=13)
    axs[1, 0].set_ylabel("Sekundy")

    axs[1, 1].bar(labels, turns_vals, color=colors, edgecolor='black', alpha=0.9)
    axs[1, 1].set_title("4. Liczba Wykonanych Manewrów (Płynność)", fontsize=13)
    axs[1, 1].set_ylabel("Zakręty")

    # Ujednolicona siatka i wartości nad słupkami
    for ax, vals in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]],
                        [dist_vals, risk_vals, time_vals, turns_vals]):
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        max_v = max(vals) if max(vals) > 0 else 1
        for i, v in enumerate(vals):
            ax.text(i, v + (max_v * 0.02), f"{v:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _setup_ui_colorbars(fig, ax, img, speed_axes_rect: list,
                        risk_fraction: float = 0.046,
                        risk_pad: float = 0.05,
                        risk_shrink: float = 0.80) -> None:
    """Generuje ustandaryzowane paski boczne dla ryzyka i dolne dla prędkości."""
    # Pasek Ryzyka z uwzględnieniem elastycznych marginesów
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=risk_fraction, pad=risk_pad, shrink=risk_shrink, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    # Pasek Prędkości (położenie definiowane przez argument speed_axes_rect)
    cax_speed = fig.add_axes(speed_axes_rect)
    sm = plt.cm.ScalarMappable(cmap=get_speed_cmap(), norm=plt.Normalize(0, 18.0))
    sm.set_array([])
    cbar_speed = fig.colorbar(sm, cax=cax_speed, orientation='horizontal')
    cbar_speed.set_label('Prędkość Kinematyczna [m/s]', color='white', labelpad=10)
    cbar_speed.ax.xaxis.set_tick_params(color='white', labelcolor='white')

def get_city_cmap() -> LinearSegmentedColormap:
    """
    Biały -> Żółty -> Pomarańczowy -> Czerwony -> Czarny
    """
    colors = [
        # WARTOŚĆ | KOLOR (R, G, B)
        (0.0, (1.0, 1.0, 1.0)),  # 0.0  = Biały
        (0.01, (1.0, 1.0, 0.0)),  # 0.01 = Żółty
        (0.4, (1.0, 0.5, 0.0)),  # 0.4  = Pomarańczowy
        (0.8, (1.0, 0.0, 0.0)),  # 0.8  = Czysta Czerwień
        (0.99, (0.5, 0.0, 0.0)),  # 0.99 = Ciemna Czerwień
        (0.991, (0.0, 0.0, 0.0)),  # Odcięcie
        (1.0, (0.0, 0.0, 0.0))  # 1.0  = Czarny
    ]
    return LinearSegmentedColormap.from_list("CityMapOrange", colors)


def get_speed_cmap() -> LinearSegmentedColormap:
    """
    Paleta chłodna z ultrawysoką czułością w środkowym i górnym zakresie prędkości.
    Nawet lekkie zwolnienie z 18 m/s (lime) natychmiast wybija błękit (cyan).
    """
    colors = [
        (0.00, "magenta"),      # 0 m/s     - Zatrzymanie
        (0.20, "mediumblue"),   # 3.6 m/s   - Wolny lot / ostre hamowanie
        (0.50, "dodgerblue"),   # 9.0 m/s   - Środek skali (wyraźny niebieski)
        (0.75, "cyan"),         # 13.5 m/s  - Zauważalne zwolnienie z V_max
        (0.90, "springgreen"),  # 16.2 m/s  - Lekkie hamowanie przed łagodnym łukiem
        (1.00, "lime")          # 18.0 m/s  - Czysty lot z maksymalną prędkością
    ]
    return LinearSegmentedColormap.from_list("SpeedMap_UltraCool", colors)


def setup_dark_theme(fig, ax) -> None:
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


def compute_path_speeds(path: List[Tuple[int, int]], initial_speed: float = 0.0) -> np.ndarray:
    if len(path) < 2:
        return np.array([initial_speed] * len(path))

    v_max = 18.0
    a = 4.0
    max_lat_a = 7.0  # Maksymalne przyspieszenie boczne (bezpieczne dla 30kg)

    speeds = np.zeros(len(path))
    turn_speeds = np.full(len(path), v_max)

    turn_speeds[0] = initial_speed
    turn_speeds[-1] = 0.0

    for i in range(1, len(path) - 1):
        dx1, dy1 = path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]
        dx2, dy2 = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            dot = dx1 * dx2 + dy1 * dy2
            mag1, mag2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
            if mag1 * mag2 > 0:
                cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                angle = math.acos(cos_theta)

                # --- NOWA LOGIKA FIZYCZNA ---
                # 1. Prędkość wynikająca z geometrii (rzut wektora)
                v_geom = v_max * max(0.0, math.cos(angle))

                # 2. Prędkość wynikająca z siły dośrodkowej (żeby nie wypadł z trasy)
                # r_approx to promień łuku na siatce grid
                r_approx = 1.5 / max(0.1, math.sin(angle / 2))
                v_phys = math.sqrt(max_lat_a * r_approx)

                # Wybieramy bezpieczniejszą (mniejszą) wartość
                turn_speeds[i] = max(0.5, min(v_geom, v_phys))

    speeds[0] = initial_speed

    for i in range(1, len(path)):
        dist = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        speeds[i] = min(turn_speeds[i], math.sqrt(speeds[i - 1] ** 2 + 2 * a * dist), v_max)

    for i in range(len(path) - 2, -1, -1):
        dist = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1] ** 2 + 2 * a * dist))

    return speeds

def smooth_path_with_speeds(path: List[Tuple[int, int]], speeds: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    if len(path) < 3:
        return np.array([p[0] for p in path]), np.array([p[1] for p in path]), speeds

    x, y = [p[0] for p in path], [p[1] for p in path]
    try:
        tck, u = interp.splprep([x, y], s=3.0, k=3)
        u_new = np.linspace(0, 1, num=len(path) * 10)
        x_smooth, y_smooth = interp.splev(u_new, tck)
        speeds_smooth = interp.interp1d(u, speeds, kind='linear')(u_new)
        return x_smooth, y_smooth, speeds_smooth
    except Exception:
        return np.array(x), np.array(y), speeds


def plot_simulation(
        grid_map: GridMap,
        path: List[Tuple[int, int]],
        stats: Dict[str, Any],
        algo_name: str,
        block: bool = True,
        use_smoothing: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(right=0.85, left=0.15, bottom=0.12, top=0.9)
    setup_dark_theme(fig, ax)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    _setup_ui_colorbars(fig, ax, img, speed_axes_rect=[0.195, 0.06, 0.59, 0.02])

    if path:
        speeds = compute_path_speeds(path)
        if use_smoothing:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            sx, sy, s_speeds = smooth_path_with_speeds(path, speeds)
        else:
            sx = np.array([p[0] for p in path])
            sy = np.array([p[1] for p in path])
            s_speeds = speeds

        points = np.array([sx, sy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_speeds = (s_speeds[:-1] + s_speeds[1:]) / 2.0

        # --- UJEDNOLICONY STYL (Z trybu Online) ---
        lc_bg = LineCollection(segments, colors='#555555', linewidths=7, alpha=0.4, zorder=3)

        lc = LineCollection(segments, cmap=get_speed_cmap(), norm=plt.Normalize(0, 18.0), zorder=4)
        lc.set_array(segment_speeds)
        lc.set_linewidth(5)

        ax.add_collection(lc_bg)
        ax.add_collection(lc)

        # Start i Meta
        ax.scatter([sx[0]], [sy[0]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
        ax.scatter([sx[-1]], [sy[-1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)

        # Fikcyjny wpis dla legendy dopasowany do nowego stylu
        ax.plot([], [], color='cyan', linewidth=5, label='Trasa',
                path_effects=[pe.withStroke(linewidth=7, foreground="#555555")])

    # Legenda ląduje ładnie pod paskami po prawej
    legend = ax.legend(
        loc='lower left',
        bbox_to_anchor=(1.05, -0.01),
        facecolor='#333333',
        edgecolor='white',
        title="Elementy Mapy"
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    turns_count = stats.get('turns', 0)
    title_text = (f"{algo_name}\n"
                  f"Dystans: {stats['length']:.1f} m | Czas Lotu: {stats.get('flight_time', 0):.1f} s | "
                  f"Ryzyko: {stats['risk']:.1f} | Zakręty: {turns_count}")
    ax.set_title(title_text, fontsize=14, pad=15)

    plt.show(block=block)


def plot_interactive_risk(
        grid_map: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable
) -> Slider:
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.20, right=0.85, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    _setup_ui_colorbars(fig, ax, img, speed_axes_rect=[0.195, 0.13, 0.59, 0.02])

    line_raw, = ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # --- UJEDNOLICONY STYL PUSTEJ KOLEKCJI ---
    lc_smooth_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
    lc_smooth = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, 18.0), zorder=4)
    ax.add_collection(lc_smooth_bg)
    ax.add_collection(lc_smooth)

    # Legenda zgodna ze stylem
    ax.plot([], [], color='cyan', linewidth=5, label='Trasa',
            path_effects=[pe.withStroke(linewidth=7, foreground="#555555")])

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)

    legend = ax.legend(
        loc='lower left',
        bbox_to_anchor=(1.05, -0.01),
        facecolor='#333333',
        edgecolor='white',
        title="Elementy Mapy"
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Suwak dostosowany szerokością do mapy
    ax_slider = plt.axes([0.15, 0.04, 0.70, 0.04], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W)', valmin=0.0, valmax=100.0, valinit=20.0, valstep=1.0,
                         color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    def update(val):
        w = risk_slider.val
        path, stats = search_func(grid_map, start, goal, risk_weight=w, turn_penalty=20.0)

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            line_raw.set_data(px, py)

            speeds = compute_path_speeds(path)
            sx, sy, s_speeds = smooth_path_with_speeds(path, speeds)
            points = np.array([sx, sy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_speeds = (s_speeds[:-1] + s_speeds[1:]) / 2.0

            lc_smooth_bg.set_segments(segments)
            lc_smooth.set_segments(segments)
            lc_smooth.set_array(segment_speeds)

            turns_count = stats.get('turns', 0)
            title_text = (f"A* Risk-Aware (W={w:.0f})\n"
                          f"Dyst: {stats['length']:.1f} m | Czas Lotu: {stats.get('flight_time', 0):.1f} s | "
                          f"Ryzyko: {stats['risk']:.1f} | Zakręty: {turns_count}")
            ax.set_title(title_text, fontsize=14, pad=15)
        else:
            line_raw.set_data([], [])
            lc_smooth_bg.set_segments([])
            lc_smooth.set_segments([])
            ax.set_title(f"BRAK TRASY (W={w:.0f})", fontsize=14, color='red')
        fig.canvas.draw_idle()

    risk_slider.on_changed(update)
    update(20.0)
    plt.show(block=True)
    return risk_slider


def run_online_simulation(
        env: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable,
        collision_radius: float
) -> None:
    path_global, stats_global = search_func(env, start, goal, risk_weight=20.0, turn_penalty=20.0,
                                            drone_radius=collision_radius)

    if not path_global:
        print("Błąd: Nie znaleziono trasy startowej.")
        return False

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.18, right=0.80, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    _setup_ui_colorbars(fig, ax, img, speed_axes_rect=[0.155, 0.13, 0.59, 0.02], risk_fraction=0.048, risk_pad=0.045,
                        risk_shrink=0.72)

    turns = stats_global.get('turns', 0)
    dist_total_start = stats_global['length']
    initial_title = (f"Optymalizacja tras BSP z uwzględnieniem stref ryzyka (Risk-Aware A*, W=20)\n"
                     f"Sumaryczna droga przelotu: {dist_total_start:.1f} m | "
                     f"Czas Lotu: {stats_global.get('flight_time', 0):.1f} s | Ryzyko: {stats_global['risk']:.1f} | Zakręty: {turns}")
    ax.set_title(initial_title, fontsize=14, color='white', pad=25)

    gx_smooth, gy_smooth = smooth_path_bspline(path_global)
    global_speeds = compute_path_speeds(path_global)

    line_global, = ax.plot(gx_smooth, gy_smooth, color='gray', linestyle='--', linewidth=2.5, alpha=0.8,
                           label='Pierwotny Plan')

    lc_flown_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
    lc_flown = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, 18.0), zorder=4)
    ax.add_collection(lc_flown_bg)
    ax.add_collection(lc_flown)
    ax.plot([], [], color='lime', linewidth=5, label='Droga Przebyta', zorder=0)

    line_reaction, = ax.plot([], [], color='orange', linestyle=':', linewidth=4, label='Czas Reakcji (Bezwładność)',
                             zorder=4)

    lc_new_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=4)
    lc_new = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, 18.0), zorder=5)
    ax.add_collection(lc_new_bg)
    ax.add_collection(lc_new)

    line_proxy, = ax.plot([], [], color='cyan', linewidth=5, label='Replanowana Trasa',
                          path_effects=[pe.withStroke(linewidth=7, foreground="#555555")], zorder=4)

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    goal_marker = ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                             zorder=5)
    drone_marker, = ax.plot([], [], 'o', color='yellow', markersize=12, label='Wykrycie (Zasięg)',
                            markeredgecolor='black', zorder=6)

    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.04, -0.01), facecolor='#333333', edgecolor='white',
                       title="Legenda Elementów")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    ax_slider = plt.axes([0.15, 0.04, 0.60, 0.04], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W) ', valmin=0.0, valmax=100.0, valinit=20.0, valstep=1.0,
                         color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    sim_state = {
        "clicked": False, "drone_pos": None, "target_pos": None, "mode": "IDLE",
        "base_dist": stats_global['length'], "base_time": stats_global.get('flight_time', 0),
        "base_risk": stats_global['risk'], "base_turns": stats_global.get('turns', 0),
        "flown_dist": 0.0, "flown_time": 0.0, "flown_risk": 0.0, "flown_turns": 0,
        "buffer_points": []  # Właściwe miejsce dla bufora pędu
    }

    def update_route(val):
        if not sim_state["clicked"] or sim_state["mode"] in ["CRASH", "IGNORE", "IDLE"]: return

        w = 50.0 if sim_state["mode"] == "RTH" else risk_slider.val

        # A* szuka trasy od KOŃCA bufora
        search_start = sim_state["buffer_points"][-1] if sim_state.get("buffer_points") else sim_state["drone_pos"]

        path_local, stats = search_func(env, search_start, sim_state["target_pos"], risk_weight=w,
                                        turn_penalty=20.0, drone_radius=collision_radius,
                                        initial_direction=sim_state.get("heading", (0, 0)),
                                        current_speed=sim_state.get("drone_speed", 0.0))

        if path_local:
            # Doklejamy bufor do nowej trasy
            if sim_state.get("buffer_points"):
                full_new_path = [sim_state["drone_pos"]] + sim_state["buffer_points"] + path_local[1:]
            else:
                full_new_path = path_local

            current_drone_speed = sim_state.get("drone_speed", 0.0)
            speeds = compute_path_speeds(full_new_path, initial_speed=current_drone_speed)

            sx, sy, s_speeds = smooth_path_with_speeds(full_new_path, speeds)
            points = np.array([sx, sy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segment_speeds = (s_speeds[:-1] + s_speeds[1:]) / 2.0

            lc_new_bg.set_segments(segments)
            lc_new.set_segments(segments)
            lc_new.set_array(segment_speeds)

            actual_new_dist = calculate_path_length(full_new_path)
            t_dist = sim_state["flown_dist"] + actual_new_dist

            f_turns = 0
            if len(full_new_path) > 2:
                last_dir = (full_new_path[1][0] - full_new_path[0][0], full_new_path[1][1] - full_new_path[0][1])
                for i in range(2, len(full_new_path)):
                    curr_dir = (full_new_path[i][0] - full_new_path[i - 1][0],
                                full_new_path[i][1] - full_new_path[i - 1][1])
                    if curr_dir != last_dir:
                        f_turns += 1
                        last_dir = curr_dir

            t_time = sim_state["flown_time"] + calculate_kinematic_flight_time(full_new_path)
            t_risk = sim_state["flown_risk"] + (
                calculate_segment_risk(full_new_path[:-1], env) if len(full_new_path) > 1 else 0.0)
            t_turns = sim_state["flown_turns"] + f_turns

            d_dist = t_dist - sim_state["base_dist"]
            d_time = t_time - sim_state["base_time"]
            d_risk = t_risk - sim_state["base_risk"]
            d_turns = t_turns - sim_state["base_turns"]

            def fmt(v):
                return f"+{v:.1f}" if v > 0 else f"{v:.1f}"

            def fmt_i(v):
                return f"+{int(v)}" if v > 0 else f"{int(v)}"

            stats_text = (f"Sumaryczna droga przelotu: {t_dist:.1f} m ({fmt(d_dist)} m nadłożono)\n"
                          f"Czas: {t_time:.1f}s ({fmt(d_time)}) | Ryzyko: {t_risk:.1f} ({fmt(d_risk)}) | Zakręty: {t_turns} ({fmt_i(d_turns)})")

            if sim_state["mode"] == "RTH":
                ax.set_title(f"Tryb Powrotu (W={w:.0f})\n{stats_text}", color='orange', fontsize=14, pad=25)
                lc_new.set_cmap(LinearSegmentedColormap.from_list("Warn", ["orange", "yellow"]))
                line_proxy.set_color('orange')
            else:
                ax.set_title(f"Omijanie Przeszkody (W={w:.0f})\n{stats_text}", color='lime', fontsize=14, pad=25)
                lc_new.set_cmap(get_speed_cmap())
                line_proxy.set_color('cyan')
        else:
            ax.set_title(f"DRON JEST UWIĘZIONY!", color='red', fontsize=14, pad=25)
            lc_new_bg.set_segments([])
            lc_new.set_segments([])
        fig.canvas.draw_idle()

    risk_slider.on_changed(update_route)

    def onclick(event):
        if event.inaxes != ax: return
        if sim_state["clicked"]: return

        click_x, click_y = int(event.xdata), int(event.ydata)
        OBSTACLE_RADIUS = 8
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        DRONE_RADIUS = collision_radius - 2
        SENSOR_RANGE = 50.0
        CRASH_DIST = OBSTACLE_RADIUS + collision_radius

        is_path_blocked = False
        for (px, py) in path_global:
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= CRASH_DIST:
                is_path_blocked = True
                break

        if not is_path_blocked:
            print("\n-> Radar wykrył obiekt, ale nie leży on na kursie kolizyjnym. Dron ignoruje zagrożenie.")
            ax.set_title("Zagrożenie poza kursem lotu. Brak reakcji.", color='lime', fontsize=14, pad=25)

            sx_f, sy_f, ss_f = smooth_path_with_speeds(path_global, global_speeds)
            pts = np.array([sx_f, sy_f]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc_flown_bg.set_segments(segs)
            lc_flown.set_segments(segs)
            lc_flown.set_array((ss_f[:-1] + ss_f[1:]) / 2.0)

            sim_state["clicked"] = True
            sim_state["mode"] = "IGNORE"
            fig.canvas.draw()
            return

        collision_idx = -1
        for i, (px, py) in enumerate(path_global):
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= SENSOR_RANGE:
                collision_idx = i
                break

        if collision_idx == -1: return

        processing_delay = 0.8
        v_max = 18.0
        acceleration = 4.0

        path_to_current = path_global[:collision_idx + 1]
        dist_from_start = calculate_path_length(path_to_current)
        v_accel = np.sqrt(2 * acceleration * dist_from_start) if dist_from_start > 0 else 0.0

        dist_to_next_turn = 0.0
        v_turn = v_max

        if collision_idx < len(path_global) - 2:
            current_dir = (path_global[collision_idx + 1][0] - path_global[collision_idx][0],
                           path_global[collision_idx + 1][1] - path_global[collision_idx][1])
            temp_dist = 0.0
            for i in range(collision_idx + 1, len(path_global) - 1):
                dx = path_global[i + 1][0] - path_global[i][0]
                dy = path_global[i + 1][1] - path_global[i][1]
                next_dir = (dx, dy)
                temp_dist += np.sqrt(
                    (path_global[i][0] - path_global[i - 1][0]) ** 2 + (path_global[i][1] - path_global[i - 1][1]) ** 2)

                if current_dir != next_dir:
                    dist_to_next_turn = temp_dist
                    dot = current_dir[0] * next_dir[0] + current_dir[1] * next_dir[1]
                    mag1 = np.sqrt(current_dir[0] ** 2 + current_dir[1] ** 2)
                    mag2 = np.sqrt(next_dir[0] ** 2 + next_dir[1] ** 2)
                    if mag1 * mag2 > 0:
                        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                        angle = np.arccos(cos_theta)
                        v_turn = max(1.0, v_max * (1.0 - (angle / np.pi)))
                    break

        v_brake = np.sqrt(v_turn ** 2 + 2 * acceleration * dist_to_next_turn) if dist_to_next_turn > 0 else v_max
        v_detect = min(v_max, v_accel, v_brake)
        print(f"\n-> WYKRYTO ZAGROŻENIE! Prędkość początkowa: {v_detect:.1f} m/s")

        t_stop = v_detect / acceleration
        t_react = min(processing_delay, t_stop)

        reaction_distance_meters = (v_detect * t_react) - (0.5 * acceleration * (t_react ** 2))
        reaction_indices = int(np.ceil(reaction_distance_meters))
        v_react_end = max(0.0, v_detect - acceleration * processing_delay)
        print(
            f"-> Po czasie reakcji ({processing_delay}s) dron przeleciał {reaction_distance_meters:.1f}m i zwolnił do: {v_react_end:.1f} m/s")

        drone_detect_idx = collision_idx
        drone_react_idx = min(drone_detect_idx + reaction_indices, len(path_global) - 1)

        # Pobieramy bazową ścieżkę reakcji
        reaction_path = path_global[drone_detect_idx:drone_react_idx + 1]
        current_drone_pos = path_global[drone_react_idx]

        # Wyciągamy kierunek pędu drona
        if len(reaction_path) >= 2:
            last_dx = reaction_path[-1][0] - reaction_path[-2][0]
            last_dy = reaction_path[-1][1] - reaction_path[-2][1]
            flight_heading = (int(np.sign(last_dx)), int(np.sign(last_dy)))
        else:
            flight_heading = (0, 0)

        # RYSOWANIE POMARAŃCZOWEJ LINII - dokładnie tam gdzie pęd reakcji się kończy! (BEZ BUFORA)
        rx = [p[0] for p in reaction_path]
        ry = [p[1] for p in reaction_path]
        line_reaction.set_data(rx, ry)
        drone_marker.set_data([path_global[drone_detect_idx][0]], [path_global[drone_detect_idx][1]])

        # TWORZENIE DYNAMICZNEGO BUFORA PĘDU
        buffer_points = []
        if flight_heading != (0, 0):
            # Dynamiczne wyliczanie wybiegu na podstawie prędkości na końcu hamowania.
            # Dzielimy przez np. 4.5, aby skalować:
            # 18 m/s -> ok. 4 metry wybiegu
            # 9 m/s  -> ok. 2 metry wybiegu
            # < 2 m/s -> tylko 1 metr wybiegu (niezbędne minimum do płynnego połączenia B-spline)
            buffer_steps = max(1, int(round(v_react_end / 4.5)))

            for d in range(1, buffer_steps + 1):
                bp = (current_drone_pos[0] + flight_heading[0] * d,
                      current_drone_pos[1] + flight_heading[1] * d)
                if 0 <= bp[0] < env.width and 0 <= bp[1] < env.height:
                    if env.grid[int(bp[0]), int(bp[1])] < 1.0:
                        buffer_points.append(bp)
                else:
                    break

        # Zapis do sim_state, żeby `update_route` miało do niego dostęp
        sim_state["drone_pos"] = current_drone_pos
        sim_state["buffer_points"] = buffer_points
        sim_state["heading"] = flight_heading
        sim_state["drone_speed"] = v_react_end

        # STATYSTYKI lotu przed wygenerowaniem nowej trasy
        flown_full = path_global[:drone_react_idx + 1]
        sim_state["flown_dist"] = calculate_path_length(flown_full)
        sim_state["flown_time"] = calculate_kinematic_flight_time(flown_full)
        sim_state["flown_risk"] = calculate_segment_risk(flown_full[:-1], env) if len(flown_full) > 1 else 0.0

        f_turns = 0
        if len(flown_full) > 2:
            last_dir = (flown_full[1][0] - flown_full[0][0], flown_full[1][1] - flown_full[0][1])
            for i in range(2, len(flown_full)):
                curr_dir = (flown_full[i][0] - flown_full[i - 1][0], flown_full[i][1] - flown_full[i - 1][1])
                if curr_dir != last_dir:
                    f_turns += 1
                    last_dir = curr_dir
        sim_state["flown_turns"] = f_turns

        # RYSOWANIE DROGI PRZEBYTEJ DO REAKCJI
        flown_raw = path_global[:drone_detect_idx + 1]
        f_speeds = global_speeds[:drone_detect_idx + 1]
        sx_f, sy_f, ss_f = smooth_path_with_speeds(flown_raw, f_speeds)
        points_f = np.array([sx_f, sy_f]).T.reshape(-1, 1, 2)
        segments_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)

        lc_flown_bg.set_segments(segments_f)
        lc_flown.set_segments(segments_f)
        lc_flown.set_array((ss_f[:-1] + ss_f[1:]) / 2.0)

        # SPRAWDZENIE KATASTROFY
        crash = False
        for (px, py) in reaction_path:
            dist_to_drone = np.sqrt((click_x - px) ** 2 + (click_y - py) ** 2)
            if dist_to_drone <= (OBSTACLE_RADIUS + DRONE_RADIUS):
                crash = True
                break

        if crash:
            ax.set_title("KATASTROFA! Zbyt późna reakcja przy dużej prędkości!", color='red', fontsize=15,
                         fontweight='bold')
            sim_state["clicked"] = True
            sim_state["mode"] = "CRASH"
            fig.canvas.draw()
            return

        # URUCHOMIENIE REPLANOWANIA
        sim_state["clicked"] = True

        search_start = sim_state["buffer_points"][-1] if sim_state.get("buffer_points") else sim_state["drone_pos"]
        path_check, _ = search_func(env, search_start, goal, risk_weight=20.0, turn_penalty=20.0,
                                    drone_radius=collision_radius, initial_direction=flight_heading,
                                    current_speed=v_react_end)

        if path_check:
            sim_state["target_pos"] = goal
            sim_state["mode"] = "NORMAL"
            line_proxy.set_label('Replanowana Trasa')
        else:
            sim_state["target_pos"] = start
            sim_state["mode"] = "RTH"
            goal_marker.set_facecolor('gray')
            line_proxy.set_label('Powrót (Awaryjny)')
            risk_slider.set_val(50.0)

        if sim_state["mode"] == "NORMAL": update_route(risk_slider.val)

        # ANALIZA W KONSOLI
        path_remainder = path_global[drone_react_idx:]
        base_risk = calculate_segment_risk(path_remainder, env)
        base_len = calculate_path_length(path_remainder)

        generate_analysis_table(
            env=env, start_pos=sim_state["drone_pos"], target_pos=sim_state["target_pos"],
            search_func=search_func, base_len=base_len, base_risk=base_risk,
            collision_radius=collision_radius, table_title="ANALIZA TRYBU ONLINE (H4)"
        )

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

def generate_thesis_charts(
        envs: List[GridMap],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        func_dijkstra: Callable,
        func_astar: Callable,
        func_risk_astar: Callable,
        collision_radius: float,
        density_label: str = ""
) -> None:
    print("\n--- GENEROWANIE WYKRESÓW DO PRACY DYPLOMOWEJ ---")

    output_dir = "research_results"
    if density_label:
        output_dir = os.path.join(output_dir, f"gestosc_{density_label}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weights = [0, 20, 40, 60, 80, 100]
    print(f"Początkowa liczba map do sprawdzenia: {len(envs)}")

    # --- KROK 1: Wstępna selekcja map (Fair Benchmarking) ---
    # Używamy tylko tych map, które są rozwiązywalne dla KAŻDEJ wagi przez WSZYSTKIE algorytmy
    valid_envs = []
    for i, env in enumerate(envs):
        is_valid = True
        for w in weights:
            _, sd = func_dijkstra(env, start, goal, risk_weight=float(w), turn_penalty=20.0,
                                  drone_radius=collision_radius)
            _, sa = func_astar(env, start, goal, risk_weight=float(w), turn_penalty=20.0, drone_radius=collision_radius)
            _, sr = func_risk_astar(env, start, goal, risk_weight=float(w), turn_penalty=20.0,
                                    drone_radius=collision_radius)

            if not (sd['found'] and sa['found'] and sr['found']):
                is_valid = False
                break

        if is_valid:
            valid_envs.append(env)

    print(f"Wyselekcjonowano map użytecznych dla wszystkich wag: {len(valid_envs)}")

    if len(valid_envs) == 0:
        print("BŁĄD: Żadna mapa nie przetrwała filtrowania. Zmniejsz maksymalną wagę W lub wygeneruj łatwiejsze mapy.")
        return

    # Słowniki na krzywe
    res_d = {'w': [], 'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}
    res_a = {'w': [], 'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}
    res_r = {'w': [], 'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}

    bench_0 = {}
    bench_20 = {}

    # --- KROK 2: Zbieranie danych (teraz już bezpieczne) ---
    f_count = len(valid_envs)  # Zawsze ta sama liczba map!

    for w in weights:
        sum_d = {k: 0.0 for k in ['len', 'risk', 'time', 'nodes', 'flight', 'turns']}
        sum_a = {k: 0.0 for k in ['len', 'risk', 'time', 'nodes', 'flight', 'turns']}
        sum_r = {k: 0.0 for k in ['len', 'risk', 'time', 'nodes', 'flight', 'turns']}

        for env in valid_envs:
            _, sd = func_dijkstra(env, start, goal, risk_weight=float(w), turn_penalty=20.0,
                                  drone_radius=collision_radius)
            _, sa = func_astar(env, start, goal, risk_weight=float(w), turn_penalty=20.0, drone_radius=collision_radius)
            _, sr = func_risk_astar(env, start, goal, risk_weight=float(w), turn_penalty=20.0,
                                    drone_radius=collision_radius)

            sum_d['len'] += sd['length'];
            sum_d['risk'] += sd['risk'];
            sum_d['time'] += sd['time'] * 1000;
            sum_d['nodes'] += sd['nodes'];
            sum_d['flight'] += sd.get('flight_time', 0);
            sum_d['turns'] += sd.get('turns', 0)
            sum_a['len'] += sa['length'];
            sum_a['risk'] += sa['risk'];
            sum_a['time'] += sa['time'] * 1000;
            sum_a['nodes'] += sa['nodes'];
            sum_a['flight'] += sa.get('flight_time', 0);
            sum_a['turns'] += sa.get('turns', 0)
            sum_r['len'] += sr['length'];
            sum_r['risk'] += sr['risk'];
            sum_r['time'] += sr['time'] * 1000;
            sum_r['nodes'] += sr['nodes'];
            sum_r['flight'] += sr.get('flight_time', 0);
            sum_r['turns'] += sr.get('turns', 0)

        # Zapisywanie do krzywych
        if f_count > 0:
            res_d['w'].append(w)
            res_d['len'].append(sum_d['len'] / f_count)
            res_d['risk'].append(sum_d['risk'] / f_count)
            res_d['time'].append(sum_d['time'] / f_count)
            res_d['nodes'].append(sum_d['nodes'] / f_count)
            res_d['flight'].append(sum_d['flight'] / f_count)
            res_d['turns'].append(sum_d['turns'] / f_count)

            res_a['w'].append(w)
            res_a['len'].append(sum_a['len'] / f_count)
            res_a['risk'].append(sum_a['risk'] / f_count)
            res_a['time'].append(sum_a['time'] / f_count)
            res_a['nodes'].append(sum_a['nodes'] / f_count)
            res_a['flight'].append(sum_a['flight'] / f_count)
            res_a['turns'].append(sum_a['turns'] / f_count)

            res_r['w'].append(w)
            res_r['len'].append(sum_r['len'] / f_count)
            res_r['risk'].append(sum_r['risk'] / f_count)
            res_r['time'].append(sum_r['time'] / f_count)
            res_r['nodes'].append(sum_r['nodes'] / f_count)
            res_r['flight'].append(sum_r['flight'] / f_count)
            res_r['turns'].append(sum_r['turns'] / f_count)

        # SPRAWIEDLIWE POBRANIE DANYCH DLA W=0
        if w == 0 and f_count > 0:
            bench_0 = {
                'd_len': sum_d['len'] / f_count, 'd_risk': sum_d['risk'] / f_count,
                'd_fl': sum_d['flight'] / f_count, 'd_trn': sum_d['turns'] / f_count,
                'a_len': sum_a['len'] / f_count, 'a_risk': sum_a['risk'] / f_count,
                'a_fl': sum_a['flight'] / f_count, 'a_trn': sum_a['turns'] / f_count,
                'r_len': sum_r['len'] / f_count, 'r_risk': sum_r['risk'] / f_count,
                'r_fl': sum_r['flight'] / f_count, 'r_trn': sum_r['turns'] / f_count,
            }

        # SPRAWIEDLIWE POBRANIE DANYCH DLA W=20
        if w == 20 and f_count > 0:
            bench_20 = {
                'd_len': sum_d['len'] / f_count, 'd_risk': sum_d['risk'] / f_count,
                'd_fl': sum_d['flight'] / f_count, 'd_trn': sum_d['turns'] / f_count,
                'a_len': sum_a['len'] / f_count, 'a_risk': sum_a['risk'] / f_count,
                'a_fl': sum_a['flight'] / f_count, 'a_trn': sum_a['turns'] / f_count,
                'r_len': sum_r['len'] / f_count, 'r_risk': sum_r['risk'] / f_count,
                'r_fl': sum_r['flight'] / f_count, 'r_trn': sum_r['turns'] / f_count,
            }

    plt.style.use('default')

    # --- WYKRES 1: 3 KRZYWE PARETO ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    if len(res_d['len']) > 0:
        ax1.plot(res_d['len'], res_d['risk'], color='#4472C4', marker='o', linewidth=2.5,
                 label='Dijkstra (Teoretyczna)')
    if len(res_a['len']) > 0:
        ax1.plot(res_a['len'], res_a['risk'], color='#ED7D31', marker='s', linewidth=2.5, label='A* Standard')
    if len(res_r['len']) > 0:
        ax1.plot(res_r['len'], res_r['risk'], color='#70AD47', marker='D', linewidth=3.5,
                 label='Risk-Aware A* (Kinematyczna)')

    ax1.set_title("Odwzorowanie Kompromisu: Długość Trasy vs Ekspozycja na Ryzyko", fontsize=15, pad=15)
    ax1.set_xlabel("Długość Trasy [m]", fontsize=13)
    ax1.set_ylabel("Całkowity Koszt Ryzyka", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_pareto_3_algorithms.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- WYKRES 2: WYDAJNOŚĆ ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    if len(res_d['time']) > 0:
        ax2a.plot(res_d['w'], res_d['time'], color='#4472C4', marker='o', label='Dijkstra')
        ax2b.plot(res_d['w'], res_d['nodes'], color='#4472C4', marker='o', label='Dijkstra')
    if len(res_a['time']) > 0:
        ax2a.plot(res_a['w'], res_a['time'], color='#ED7D31', marker='s', label='A* Standard')
        ax2b.plot(res_a['w'], res_a['nodes'], color='#ED7D31', marker='s', label='A* Standard')
    if len(res_r['time']) > 0:
        ax2a.plot(res_r['w'], res_r['time'], color='#70AD47', marker='D', linewidth=2, label='Risk-Aware A*')
        ax2b.plot(res_r['w'], res_r['nodes'], color='#70AD47', marker='D', linewidth=2, label='Risk-Aware A*')

    ax2a.set_title("Średni Czas Obliczeń", fontsize=13)
    ax2a.set_xlabel("Waga Ryzyka (W)")
    ax2a.set_ylabel("Czas [ms]")
    ax2a.grid(True, linestyle='--', alpha=0.5)
    ax2a.legend()

    ax2b.set_title("Liczba Odwiedzonych Węzłów", fontsize=13)
    ax2b.set_xlabel("Waga Ryzyka (W)")
    ax2b.set_ylabel("Węzły")
    ax2b.grid(True, linestyle='--', alpha=0.5)
    ax2b.legend()

    plt.suptitle("Analiza Wydajności Obliczeniowej", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_performance_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # --- WYKRES 3: SŁUPKI DLA W=0 (IZOLACJA KINEMATYKI) ---
    if bench_0:
        _plot_benchmark_bars(
            bench_data=bench_0,
            title="Izolacja Kinematyki: Porównanie wszystkich algorytmów przy braku uwzględniania ryzyka (W=0)",
            filename=os.path.join(output_dir, "3_algorithm_comparison_fair_benchmark_W0.png"),
            w_label="Waga Ryzyka W=0"
        )

    # --- WYKRES 4: SŁUPKI DLA W=20 (FAIR BENCHMARK DLA RYZYKA) ---
    if bench_20:
        _plot_benchmark_bars(
            bench_data=bench_20,
            title="Fair Benchmarking: Porównanie algorytmów przy optymalnym omijaniu ryzyka (W=20)",
            filename=os.path.join(output_dir, "4_algorithm_comparison_benchmark_W20.png"),
            w_label="Waga Ryzyka W=20"
        )

    print("Gotowe! Wykresy zapisano w:", output_dir)
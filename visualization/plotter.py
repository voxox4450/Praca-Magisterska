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
from algorithms.common import (
    calculate_segment_risk, calculate_path_length, generate_analysis_table,
    calculate_kinematic_flight_time, compute_turn_cost, compute_safe_turn_speed
)
import os
from config import (
    V_MAX_MS, ACCELERATION, MAX_LATERAL_ACCEL, MIN_TURN_SPEED,
    RISK_WEIGHT, TURN_PENALTY,
    PARETO_WEIGHT_MAX, PARETO_WEIGHT_STEP,
    COLLISION_RADIUS, SENSOR_RANGE  # [FIX #24] Import z config
)


def _plot_benchmark_bars(bench_data: dict, title: str, filename: str, w_label: str) -> None:
    """Funkcja pomocnicza do rysowania 4 wykresów słupkowych."""
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

    for ax, vals in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]],
                        [dist_vals, risk_vals, time_vals, turns_vals]):
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        max_v = max(vals) if max(vals) > 0 else 1
        for i, v in enumerate(vals):
            ax.text(i, v + (max_v * 0.02), f"{v:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _setup_ui_colorbars(fig, ax, img, speed_axes_rect: list,
                        risk_fraction: float = 0.046,
                        risk_pad: float = 0.05,
                        risk_shrink: float = 0.80) -> None:
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=risk_fraction, pad=risk_pad, shrink=risk_shrink, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    cax_speed = fig.add_axes(speed_axes_rect)
    sm = plt.cm.ScalarMappable(cmap=get_speed_cmap(), norm=plt.Normalize(0, V_MAX_MS))
    sm.set_array([])
    cbar_speed = fig.colorbar(sm, cax=cax_speed, orientation='horizontal')
    cbar_speed.set_label('Prędkość Kinematyczna [m/s]', color='white', labelpad=10)
    cbar_speed.ax.xaxis.set_tick_params(color='white', labelcolor='white')


def get_city_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.0, (1.0, 1.0, 1.0)),
        (0.01, (1.0, 1.0, 0.0)),
        (0.4, (1.0, 0.5, 0.0)),
        (0.8, (1.0, 0.0, 0.0)),
        (0.99, (0.5, 0.0, 0.0)),
        (0.991, (0.0, 0.0, 0.0)),
        (1.0, (0.0, 0.0, 0.0))
    ]
    return LinearSegmentedColormap.from_list("CityMapOrange", colors)


def get_speed_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.00, "magenta"),
        (0.20, "mediumblue"),
        (0.50, "dodgerblue"),
        (0.75, "cyan"),
        (0.90, "springgreen"),
        (1.00, "lime")
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
        return np.array([p[0] for p in path]), np.array([p[1] for p in path])

    x = [p[0] for p in path]
    y = [p[1] for p in path]
    try:
        tck, u = interp.splprep([x, y], s=3.0, k=3)
        u_new = np.linspace(0, 1, num=len(path) * 10)
        x_smooth, y_smooth = interp.splev(u_new, tck)
        return x_smooth, y_smooth
    except Exception:
        return np.array(x), np.array(y)


# [FIX #10] Profil prędkości — dokumentacja spójności z planistą.
# Planista (base_search) liczy forward-only z ograniczeniami zakrętów.
# Wizualizacja dodaje backward pass (hamowanie PRZED zakrętem) i forward enforce,
# co jest fizycznie konieczne — planista implicite zakłada że dron wyhamuje.
def compute_path_speeds(path: List[Tuple[int, int]], initial_speed: float = 0.0) -> np.ndarray:
    if len(path) < 2:
        return np.array([initial_speed] * len(path))

    v_max = V_MAX_MS
    a = ACCELERATION

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
                # [FIX #18] Spójna funkcja z common.py
                turn_speeds[i] = compute_safe_turn_speed(angle)

    speeds[0] = initial_speed

    # Pass 1: Forward (rozpędzanie)
    for i in range(1, len(path)):
        dist = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        speeds[i] = min(turn_speeds[i], math.sqrt(speeds[i - 1] ** 2 + 2 * a * dist), v_max)

    # Pass 2: Backward (hamowanie przed zakrętami)
    for i in range(len(path) - 2, -1, -1):
        dist = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1] ** 2 + 2 * a * dist))

    # Pass 3: Enforce fizyczny warunek brzegowy (prędkość startowa jest faktem)
    speeds[0] = initial_speed
    for i in range(1, len(path)):
        dist = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        min_phys_speed = math.sqrt(max(0.0, speeds[i - 1] ** 2 - 2 * a * dist))
        speeds[i] = max(speeds[i], min_phys_speed)

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

        a = ACCELERATION
        for i in range(1, len(speeds_smooth)):
            dist = math.hypot(x_smooth[i] - x_smooth[i-1], y_smooth[i] - y_smooth[i-1])
            speeds_smooth[i] = min(speeds_smooth[i],
                                   math.sqrt(max(0.0, speeds_smooth[i-1]**2 + 2*a*dist)))
        for i in range(len(speeds_smooth)-2, -1, -1):
            dist = math.hypot(x_smooth[i+1] - x_smooth[i], y_smooth[i+1] - y_smooth[i])
            speeds_smooth[i] = min(speeds_smooth[i],
                                   math.sqrt(max(0.0, speeds_smooth[i+1]**2 + 2*a*dist)))

        initial_v = speeds[0]
        speeds_smooth[0] = initial_v
        for i in range(1, len(speeds_smooth)):
            dist = math.hypot(x_smooth[i] - x_smooth[i-1], y_smooth[i] - y_smooth[i-1])
            min_phys = math.sqrt(max(0.0, speeds_smooth[i-1]**2 - 2*a*dist))
            speeds_smooth[i] = max(speeds_smooth[i], min_phys)

        return x_smooth, y_smooth, speeds_smooth
    except Exception:
        return np.array(x), np.array(y), speeds


def plot_simulation(
        grid_map: GridMap, path: List[Tuple[int, int]], stats: Dict[str, Any],
        algo_name: str, block: bool = True, use_smoothing: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(right=0.85, left=0.15, bottom=0.12, top=0.9)
    setup_dark_theme(fig, ax)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    _setup_ui_colorbars(fig, ax, img, speed_axes_rect=[0.195, 0.06, 0.59, 0.02])

    if path:
        speeds = compute_path_speeds(path)
        if use_smoothing:
            ax.plot([p[0] for p in path], [p[1] for p in path],
                    color='gray', linestyle='--', linewidth=1, alpha=0.6)
            sx, sy, s_speeds = smooth_path_with_speeds(path, speeds)
        else:
            sx = np.array([p[0] for p in path])
            sy = np.array([p[1] for p in path])
            s_speeds = speeds

        points = np.array([sx, sy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_speeds = (s_speeds[:-1] + s_speeds[1:]) / 2.0

        lc_bg = LineCollection(segments, colors='#555555', linewidths=7, alpha=0.4, zorder=3)
        lc = LineCollection(segments, cmap=get_speed_cmap(), norm=plt.Normalize(0, V_MAX_MS), zorder=4)
        lc.set_array(segment_speeds)
        lc.set_linewidth(5)
        ax.add_collection(lc_bg)
        ax.add_collection(lc)

        ax.scatter([sx[0]], [sy[0]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
        ax.scatter([sx[-1]], [sy[-1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)
        ax.plot([], [], color='cyan', linewidth=5, label='Trasa',
                path_effects=[pe.withStroke(linewidth=7, foreground="#555555")])

    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.05, -0.01),
                       facecolor='#333333', edgecolor='white', title="Elementy Mapy")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    title_text = (f"{algo_name}\n"
                  f"Dystans: {stats['length']:.1f} m | Czas Lotu: {stats.get('flight_time', 0):.1f} s | "
                  f"Ryzyko: {stats['risk']:.1f} | Zakręty: {stats.get('turns', 0)}")
    ax.set_title(title_text, fontsize=14, pad=15)
    plt.show(block=block)


def plot_interactive_risk(
        grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
        search_func: Callable
) -> Slider:
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.20, right=0.85, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    _setup_ui_colorbars(fig, ax, img, speed_axes_rect=[0.195, 0.13, 0.59, 0.02])

    line_raw, = ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    lc_smooth_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
    lc_smooth = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=4)
    ax.add_collection(lc_smooth_bg)
    ax.add_collection(lc_smooth)

    ax.plot([], [], color='cyan', linewidth=5, label='Trasa',
            path_effects=[pe.withStroke(linewidth=7, foreground="#555555")])
    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)

    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.05, -0.01),
                       facecolor='#333333', edgecolor='white', title="Elementy Mapy")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    ax_slider = plt.axes([0.15, 0.04, 0.70, 0.04], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W)', valmin=0.0, valmax=40.0,
                         valinit=RISK_WEIGHT, valstep=1.0, color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    def update(val):
        w = risk_slider.val
        path, stats = search_func(grid_map, start, goal, risk_weight=w, turn_penalty=TURN_PENALTY)

        if path:
            line_raw.set_data([p[0] for p in path], [p[1] for p in path])
            speeds = compute_path_speeds(path)
            sx, sy, s_speeds = smooth_path_with_speeds(path, speeds)
            points = np.array([sx, sy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc_smooth_bg.set_segments(segments)
            lc_smooth.set_segments(segments)
            lc_smooth.set_array((s_speeds[:-1] + s_speeds[1:]) / 2.0)

            title_text = (f"A* Risk-Aware (W={w:.0f})\n"
                          f"Dyst: {stats['length']:.1f} m | Czas Lotu: {stats.get('flight_time', 0):.1f} s | "
                          f"Ryzyko: {stats['risk']:.1f} | Zakręty: {stats.get('turns', 0)}")
            ax.set_title(title_text, fontsize=14, pad=15)
        else:
            line_raw.set_data([], [])
            lc_smooth_bg.set_segments([])
            lc_smooth.set_segments([])
            ax.set_title(f"BRAK TRASY (W={w:.0f})", fontsize=14, color='red')
        fig.canvas.draw_idle()

    risk_slider.on_changed(update)
    update(RISK_WEIGHT)
    plt.show(block=True)
    return risk_slider


# ─────────────────────────────────────────────────────────────────────────────
# TRYB ONLINE
# [FIX #12] Bufor z is_collision
# [FIX #15] Spójny return None
# [FIX #21] Ryzyko liczone spójnie (pełna ścieżka, bez [:-1])
# [FIX #24] SENSOR_RANGE z config
# ─────────────────────────────────────────────────────────────────────────────
def run_online_simulation(
        env: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
        search_func: Callable, collision_radius: float
) -> None:
    path_global, stats_global = search_func(env, start, goal, risk_weight=RISK_WEIGHT,
                                            turn_penalty=TURN_PENALTY, drone_radius=collision_radius)

    if not path_global:
        print("Błąd: Nie znaleziono trasy startowej.")
        return  # [FIX #15] return None

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.18, right=0.80, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    _setup_ui_colorbars(fig, ax, img, speed_axes_rect=[0.155, 0.13, 0.59, 0.02],
                        risk_fraction=0.048, risk_pad=0.045, risk_shrink=0.72)

    turns = stats_global.get('turns', 0)
    initial_title = (f"Optymalizacja tras BSP (Risk-Aware A*, W={RISK_WEIGHT:.0f})\n"
                     f"Droga: {stats_global['length']:.1f} m | "
                     f"Czas: {stats_global.get('flight_time', 0):.1f} s | "
                     f"Ryzyko: {stats_global['risk']:.1f} | Zakręty: {turns}")
    ax.set_title(initial_title, fontsize=14, color='white', pad=25)

    gx_smooth, gy_smooth = smooth_path_bspline(path_global)
    global_speeds = compute_path_speeds(path_global)

    line_global, = ax.plot(gx_smooth, gy_smooth, color='gray', linestyle='--', linewidth=2.5, alpha=0.8,
                           label='Pierwotny Plan')

    lc_flown_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
    lc_flown = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=4)
    ax.add_collection(lc_flown_bg)
    ax.add_collection(lc_flown)
    ax.plot([], [], color='lime', linewidth=5, label='Droga Przebyta', zorder=0)

    line_reaction, = ax.plot([], [], color='orange', linestyle=':', linewidth=4,
                             label='Czas Reakcji (Bezwładność)', zorder=4)

    lc_new_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=4)
    lc_new = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=5)
    ax.add_collection(lc_new_bg)
    ax.add_collection(lc_new)

    line_proxy, = ax.plot([], [], color='cyan', linewidth=5, label='Replanowana Trasa',
                          path_effects=[pe.withStroke(linewidth=7, foreground="#555555")], zorder=4)

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    goal_marker = ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel',
                             edgecolors='black', zorder=5)
    drone_marker, = ax.plot([], [], 'o', color='yellow', markersize=12, label='Wykrycie (Zasięg)',
                            markeredgecolor='black', zorder=6)

    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.04, -0.01), facecolor='#333333',
                       edgecolor='white', title="Legenda Elementów")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    ax_slider = plt.axes([0.15, 0.04, 0.60, 0.04], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W) ', valmin=0.0, valmax=40.0,
                         valinit=RISK_WEIGHT, valstep=1.0, color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    sim_state = {
        "clicked": False, "drone_pos": None, "target_pos": None, "mode": "IDLE",
        "base_dist": stats_global['length'], "base_time": stats_global.get('flight_time', 0),
        "base_risk": stats_global['risk'], "base_turns": stats_global.get('turns', 0),
        "flown_dist": 0.0, "flown_time": 0.0, "flown_risk": 0.0, "flown_turns": 0,
        "buffer_points": []
    }

    def update_route(val):
        if not sim_state["clicked"] or sim_state["mode"] in ["CRASH", "IGNORE", "IDLE"]:
            return

        w = 40.0 if sim_state["mode"] == "RTH" else risk_slider.val
        search_start = sim_state["buffer_points"][-1] if sim_state.get("buffer_points") else sim_state["drone_pos"]

        path_local, stats = search_func(env, search_start, sim_state["target_pos"], risk_weight=w,
                                        turn_penalty=TURN_PENALTY, drone_radius=collision_radius,
                                        initial_direction=sim_state.get("heading", (0, 0)),
                                        current_speed=sim_state.get("drone_speed", 0.0),
                                        initial_straight_dist=sim_state.get("buffer_dist", 0.0))

        if path_local:
            if sim_state.get("buffer_points"):
                full_new_path = [sim_state["drone_pos"]] + sim_state["buffer_points"] + path_local[1:]
            else:
                full_new_path = path_local

            visual_speed = sim_state.get("visual_speed", sim_state.get("drone_speed", 0.0))
            speeds = compute_path_speeds(full_new_path, initial_speed=visual_speed)

            lead_in = sim_state.get("lead_in_points", [])
            if lead_in and len(full_new_path) >= 3:
                n_lead = len(lead_in)
                lead_speeds = np.full(n_lead, visual_speed)
                combined_path = lead_in + full_new_path
                combined_speeds = np.concatenate([lead_speeds, speeds])
                sx, sy, s_speeds = smooth_path_with_speeds(combined_path, combined_speeds)
                trim = n_lead * 10
                sx, sy, s_speeds = sx[trim:], sy[trim:], s_speeds[trim:]
            else:
                sx, sy, s_speeds = smooth_path_with_speeds(full_new_path, speeds)

            points = np.array([sx, sy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc_new_bg.set_segments(segments)
            lc_new.set_segments(segments)
            lc_new.set_array((s_speeds[:-1] + s_speeds[1:]) / 2.0)

            actual_new_dist = calculate_path_length(full_new_path)
            t_dist = sim_state["flown_dist"] + actual_new_dist

            f_turns = 0
            if len(full_new_path) > 2:
                last_dir = (full_new_path[1][0] - full_new_path[0][0],
                            full_new_path[1][1] - full_new_path[0][1])
                for i in range(2, len(full_new_path)):
                    curr_dir = (full_new_path[i][0] - full_new_path[i-1][0],
                                full_new_path[i][1] - full_new_path[i-1][1])
                    if curr_dir != last_dir:
                        f_turns += 1
                        last_dir = curr_dir

            t_time = sim_state["flown_time"] + calculate_kinematic_flight_time(full_new_path)
            # [FIX #21] Spójne obliczanie ryzyka — pełna ścieżka (bez [:-1])
            t_risk = sim_state["flown_risk"] + calculate_segment_risk(full_new_path, env)
            t_turns = sim_state["flown_turns"] + f_turns

            d_dist = t_dist - sim_state["base_dist"]
            d_time = t_time - sim_state["base_time"]
            d_risk = t_risk - sim_state["base_risk"]
            d_turns = t_turns - sim_state["base_turns"]

            fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
            fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

            stats_text = (f"Droga: {t_dist:.1f} m ({fmt(d_dist)} m nadłożono)\n"
                          f"Czas: {t_time:.1f}s ({fmt(d_time)}) | Ryzyko: {t_risk:.1f} ({fmt(d_risk)}) | "
                          f"Zakręty: {t_turns} ({fmt_i(d_turns)})")

            if sim_state["mode"] == "RTH":
                ax.set_title(f"Tryb Powrotu (W={w:.0f})\n{stats_text}", color='orange', fontsize=14, pad=25)
                lc_new.set_cmap(LinearSegmentedColormap.from_list("Warn", ["orange", "yellow"]))
                line_proxy.set_color('orange')
            else:
                ax.set_title(f"Omijanie Przeszkody (W={w:.0f})\n{stats_text}", color='lime', fontsize=14, pad=25)
                lc_new.set_cmap(get_speed_cmap())
                line_proxy.set_color('cyan')
        else:
            ax.set_title("DRON JEST UWIĘZIONY!", color='red', fontsize=14, pad=25)
            lc_new_bg.set_segments([])
            lc_new.set_segments([])
        fig.canvas.draw_idle()

    risk_slider.on_changed(update_route)

    def onclick(event):
        if event.inaxes != ax or sim_state["clicked"]:
            return

        click_x, click_y = int(event.xdata), int(event.ydata)
        OBSTACLE_RADIUS = 8
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        DRONE_RADIUS = collision_radius - 2
        CRASH_DIST = OBSTACLE_RADIUS + collision_radius

        is_path_blocked = any(
            np.sqrt((px - click_x)**2 + (py - click_y)**2) <= CRASH_DIST
            for px, py in path_global
        )

        if not is_path_blocked:
            print("\n-> Obiekt poza kursem kolizyjnym. Dron ignoruje zagrożenie.")
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

        # [FIX #24] SENSOR_RANGE z config
        collision_idx = -1
        for i, (px, py) in enumerate(path_global):
            if np.sqrt((px - click_x)**2 + (py - click_y)**2) <= SENSOR_RANGE:
                collision_idx = i
                break

        if collision_idx == -1:
            return

        processing_delay = 0.8
        acceleration = ACCELERATION

        v_detect = float(global_speeds[collision_idx])
        print(f"\n-> WYKRYTO ZAGROŻENIE! Prędkość: {v_detect:.1f} m/s")

        t_stop = v_detect / acceleration
        t_react = min(processing_delay, t_stop)
        reaction_distance_meters = (v_detect * t_react) - (0.5 * acceleration * (t_react ** 2))
        reaction_indices = int(np.ceil(reaction_distance_meters))
        v_react_end = max(0.0, v_detect - acceleration * processing_delay)
        print(f"-> Po reakcji ({processing_delay}s): {reaction_distance_meters:.1f}m, v={v_react_end:.1f} m/s")

        drone_detect_idx = collision_idx
        drone_react_idx = min(drone_detect_idx + reaction_indices, len(path_global) - 1)

        reaction_path = path_global[drone_detect_idx:drone_react_idx + 1]
        current_drone_pos = path_global[drone_react_idx]

        if len(reaction_path) >= 2:
            last_dx = reaction_path[-1][0] - reaction_path[-2][0]
            last_dy = reaction_path[-1][1] - reaction_path[-2][1]
            flight_heading = (int(np.sign(last_dx)), int(np.sign(last_dy)))
        else:
            flight_heading = (0, 0)

        line_reaction.set_data([p[0] for p in reaction_path], [p[1] for p in reaction_path])
        drone_marker.set_data([path_global[drone_detect_idx][0]], [path_global[drone_detect_idx][1]])

        # [FIX #12] Bufor pędu z pełnym is_collision
        buffer_points = []
        buffer_dist = 0.0
        if flight_heading != (0, 0):
            r_45 = 1.5 / max(0.1, math.sin(math.radians(22.5)))
            v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_45))
            v_safe_ref = min(v_safe_ref, V_MAX_MS)

            braking_dist_needed = max(0.0, (v_react_end**2 - v_safe_ref**2) / (2 * ACCELERATION)) \
                if v_react_end > v_safe_ref else 0.0

            step_len = math.sqrt(flight_heading[0]**2 + flight_heading[1]**2)
            buffer_steps = max(1, int(math.ceil(braking_dist_needed / step_len)))

            for d in range(1, buffer_steps + 1):
                bp = (current_drone_pos[0] + flight_heading[0] * d,
                      current_drone_pos[1] + flight_heading[1] * d)
                bx, by = int(bp[0]), int(bp[1])
                if 0 <= bx < env.width and 0 <= by < env.height:
                    if not env.is_collision(bx, by, drone_radius=collision_radius):
                        buffer_points.append(bp)
                        buffer_dist += step_len
                    else:
                        break
                else:
                    break

        v_at_buffer_end = max(0.0, math.sqrt(max(0.0, v_react_end**2 - 2 * ACCELERATION * buffer_dist)))

        sim_state["drone_pos"] = current_drone_pos
        sim_state["buffer_points"] = buffer_points
        sim_state["heading"] = flight_heading
        sim_state["drone_speed"] = v_at_buffer_end
        sim_state["visual_speed"] = v_react_end
        sim_state["buffer_dist"] = buffer_dist

        LEAD_IN_COUNT = 5
        lead_start = max(0, drone_react_idx - LEAD_IN_COUNT)
        sim_state["lead_in_points"] = list(path_global[lead_start:drone_react_idx])

        flown_full = path_global[:drone_react_idx + 1]
        sim_state["flown_dist"] = calculate_path_length(flown_full)
        sim_state["flown_time"] = calculate_kinematic_flight_time(flown_full)
        # [FIX #21] Spójna metryka ryzyka — pełna ścieżka
        sim_state["flown_risk"] = calculate_segment_risk(flown_full, env)

        f_turns = 0
        if len(flown_full) > 2:
            last_dir = (flown_full[1][0] - flown_full[0][0], flown_full[1][1] - flown_full[0][1])
            for i in range(2, len(flown_full)):
                curr_dir = (flown_full[i][0] - flown_full[i-1][0], flown_full[i][1] - flown_full[i-1][1])
                if curr_dir != last_dir:
                    f_turns += 1
                    last_dir = curr_dir
        sim_state["flown_turns"] = f_turns

        # Rysowanie drogi przebytej
        flown_raw = path_global[:drone_detect_idx + 1]
        f_speeds = global_speeds[:drone_detect_idx + 1]
        sx_f, sy_f, ss_f = smooth_path_with_speeds(flown_raw, f_speeds)
        pts_f = np.array([sx_f, sy_f]).T.reshape(-1, 1, 2)
        segs_f = np.concatenate([pts_f[:-1], pts_f[1:]], axis=1)
        lc_flown_bg.set_segments(segs_f)
        lc_flown.set_segments(segs_f)
        lc_flown.set_array((ss_f[:-1] + ss_f[1:]) / 2.0)

        # Sprawdzenie katastrofy
        crash = any(
            np.sqrt((click_x - px)**2 + (click_y - py)**2) <= (OBSTACLE_RADIUS + DRONE_RADIUS)
            for px, py in reaction_path
        )

        if crash:
            ax.set_title("KATASTROFA! Zbyt późna reakcja!", color='red', fontsize=15, fontweight='bold')
            sim_state["clicked"] = True
            sim_state["mode"] = "CRASH"
            fig.canvas.draw()
            return

        sim_state["clicked"] = True

        search_start = sim_state["buffer_points"][-1] if sim_state.get("buffer_points") else sim_state["drone_pos"]
        path_check, _ = search_func(env, search_start, goal, risk_weight=RISK_WEIGHT,
                                    turn_penalty=TURN_PENALTY, drone_radius=collision_radius,
                                    initial_direction=flight_heading, current_speed=v_at_buffer_end,
                                    initial_straight_dist=buffer_dist)

        if path_check:
            sim_state["target_pos"] = goal
            sim_state["mode"] = "NORMAL"
            line_proxy.set_label('Replanowana Trasa')
        else:
            sim_state["target_pos"] = start
            sim_state["mode"] = "RTH"
            goal_marker.set_facecolor('gray')
            line_proxy.set_label('Powrót (Awaryjny)')
            risk_slider.set_val(40.0)

        if sim_state["mode"] == "NORMAL":
            update_route(risk_slider.val)

        path_remainder = path_global[drone_react_idx:]
        base_risk = calculate_segment_risk(path_remainder, env)
        base_len = calculate_path_length(path_remainder)
        generate_analysis_table(env=env, start_pos=sim_state["drone_pos"], target_pos=sim_state["target_pos"],
                                search_func=search_func, base_len=base_len, base_risk=base_risk,
                                collision_radius=collision_radius, table_title="ANALIZA TRYBU ONLINE (H4)")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
# [FIX #13] Metryka: rzeczywisty dystans EDT od przeszkód (nie (1-avg_risk)*100)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_advanced_metrics(path, env):
    """
    3 zaawansowane wskaźniki jakości trajektorii:
    1. Indeks Płynności — suma zmian kątów kursu [°] (↓ lepiej)
    2. Ekspozycja Maksymalna — najwyższe punktowe ryzyko na trasie (↓ lepiej)
    3. Średni Margines Bezpieczeństwa — średni dystans EDT od budynków [kratki] (↑ lepiej)
    """
    if not path or len(path) < 2:
        return 0.0, 0.0, 0.0

    # 1. Indeks Płynności
    smoothness_index = 0.0
    if len(path) > 2:
        for i in range(1, len(path) - 1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            mag1 = math.hypot(*v1)
            mag2 = math.hypot(*v2)
            if mag1 * mag2 > 0:
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                smoothness_index += math.degrees(math.acos(cos_theta))

    # 2. Ekspozycja Maksymalna
    max_exposure = max(env.grid[int(p[0]), int(p[1])] for p in path)

    # [FIX #13] Rzeczywisty margines = średni dystans EDT od budynków [kratki].
    # dist_matrix daje euklidesowy dystans do najbliższej ściany.
    dist_values = [float(env.dist_matrix[int(p[0]), int(p[1])]) for p in path]
    safety_margin = sum(dist_values) / len(dist_values)

    return smoothness_index, max_exposure, safety_margin


# ─────────────────────────────────────────────────────────────────────────────
# GENEROWANIE WYKRESÓW — PEŁNA NAPRAWA
# [FIX #1]      Usunięto podwójne liczenie metryk zaawansowanych
# [FIX #3]      Dodano std dev do wykresów słupkowych
# [FIX #11,#14] Poprawna obsługa nieznalezionych tras (per-mapa listy)
# ─────────────────────────────────────────────────────────────────────────────
def generate_thesis_charts(
        envs: List[GridMap],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        func_dijkstra: Callable,
        func_astar: Callable,
        func_risk_astar: Callable,
        collision_radius: float,
        density_label: str = "",
        turn_penalty: float = TURN_PENALTY
) -> None:
    print("\n--- GENEROWANIE WYKRESÓW DO PRACY DYPLOMOWEJ ---")

    output_dir = "research_results"
    if density_label:
        output_dir = os.path.join(output_dir, f"gestosc_{density_label}")
    os.makedirs(output_dir, exist_ok=True)

    weights = sorted(set(
        list(range(0, PARETO_WEIGHT_MAX + 1, PARETO_WEIGHT_STEP)) + [int(RISK_WEIGHT)]
    ))
    print(f"Początkowa liczba map: {len(envs)}")

    # [OPT] Mapy już zwalidowane w run_batch_benchmark() — nie powtarzamy.
    valid_envs = envs

    print(f"Rozwiązywalnych map: {len(valid_envs)}")
    if len(valid_envs) == 0:
        print("BŁĄD: Żadna mapa nie przetrwała filtrowania.")
        return

    # Krzywe Pareto
    res_d = {'w': [], 'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}
    res_a = {'w': [], 'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}
    res_r = {'w': [], 'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}

    bench_0 = {}
    bench_20 = {}
    f_count = len(valid_envs)

    for w in weights:
        # [FIX #11, #14] Listy per-mapa zamiast sum — porażki nie zaniżają średnich
        per_d = {'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}
        per_a = {'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}
        per_r = {'len': [], 'risk': [], 'time': [], 'nodes': [], 'flight': [], 'turns': []}

        # [FIX #1] Metryki zaawansowane — TYLKO RAZ, w tej samej pętli
        adv_d = {'smooth': [], 'max_exp': [], 'safe_marg': []}
        adv_a = {'smooth': [], 'max_exp': [], 'safe_marg': []}
        adv_r = {'smooth': [], 'max_exp': [], 'safe_marg': []}

        for env in valid_envs:
            # [FIX #2] Wspólna kara turn_penalty
            path_d, sd = func_dijkstra(env, start, goal, risk_weight=float(w),
                                       turn_penalty=turn_penalty, drone_radius=collision_radius)
            path_a, sa = func_astar(env, start, goal, risk_weight=float(w),
                                    turn_penalty=turn_penalty, drone_radius=collision_radius)
            path_r, sr = func_risk_astar(env, start, goal, risk_weight=float(w),
                                         turn_penalty=turn_penalty, drone_radius=collision_radius)

            # [FIX #14] Dodajemy TYLKO udane próby
            if sd['found']:
                per_d['len'].append(sd['length']); per_d['risk'].append(sd['risk'])
                per_d['time'].append(sd['time'] * 1000); per_d['nodes'].append(sd['nodes'])
                per_d['flight'].append(sd.get('flight_time', 0)); per_d['turns'].append(sd.get('turns', 0))

            if sa['found']:
                per_a['len'].append(sa['length']); per_a['risk'].append(sa['risk'])
                per_a['time'].append(sa['time'] * 1000); per_a['nodes'].append(sa['nodes'])
                per_a['flight'].append(sa.get('flight_time', 0)); per_a['turns'].append(sa.get('turns', 0))

            if sr['found']:
                per_r['len'].append(sr['length']); per_r['risk'].append(sr['risk'])
                per_r['time'].append(sr['time'] * 1000); per_r['nodes'].append(sr['nodes'])
                per_r['flight'].append(sr.get('flight_time', 0)); per_r['turns'].append(sr.get('turns', 0))

            # [FIX #1] Zaawansowane metryki — jednokrotnie, w głównej pętli
            if w == int(RISK_WEIGHT):
                if sd['found'] and path_d:
                    sm, me, smarg = calculate_advanced_metrics(path_d, env)
                    adv_d['smooth'].append(sm); adv_d['max_exp'].append(me); adv_d['safe_marg'].append(smarg)
                if sa['found'] and path_a:
                    sm, me, smarg = calculate_advanced_metrics(path_a, env)
                    adv_a['smooth'].append(sm); adv_a['max_exp'].append(me); adv_a['safe_marg'].append(smarg)
                if sr['found'] and path_r:
                    sm, me, smarg = calculate_advanced_metrics(path_r, env)
                    adv_r['smooth'].append(sm); adv_r['max_exp'].append(me); adv_r['safe_marg'].append(smarg)

        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        def safe_std(lst): return float(np.std(lst)) if len(lst) > 1 else 0.0

        # [FIX #11] Logowanie porażek
        n_fail_d = f_count - len(per_d['len'])
        n_fail_a = f_count - len(per_a['len'])
        n_fail_r = f_count - len(per_r['len'])
        if n_fail_d or n_fail_a or n_fail_r:
            print(f"  W={w}: porażki D={n_fail_d}/{f_count}, A*={n_fail_a}/{f_count}, R={n_fail_r}/{f_count}")

        # Krzywe Pareto
        for res, pm in [(res_d, per_d), (res_a, per_a), (res_r, per_r)]:
            res['w'].append(w)
            for key in ['len', 'risk', 'time', 'nodes', 'flight', 'turns']:
                res[key].append(safe_mean(pm[key]))

        # Słupki W=0
        if w == 0:
            bench_0 = {}
            for prefix, pm in [('d', per_d), ('a', per_a), ('r', per_r)]:
                for key in ['len', 'risk', 'trn']:
                    src_key = key if key != 'trn' else 'turns'
                    bench_0[f'{prefix}_{key}'] = safe_mean(pm[src_key])
                    bench_0[f'{prefix}_{key}_std'] = safe_std(pm[src_key])
                bench_0[f'{prefix}_fl'] = safe_mean(pm['flight'])
                bench_0[f'{prefix}_fl_std'] = safe_std(pm['flight'])

        # Słupki W=RISK_WEIGHT (z zaawansowanymi metrykami)
        if w == int(RISK_WEIGHT):
            bench_20 = {}
            for prefix, pm, pm_adv in [('d', per_d, adv_d), ('a', per_a, adv_a), ('r', per_r, adv_r)]:
                for key in ['len', 'risk', 'trn']:
                    src_key = key if key != 'trn' else 'turns'
                    bench_20[f'{prefix}_{key}'] = safe_mean(pm[src_key])
                    bench_20[f'{prefix}_{key}_std'] = safe_std(pm[src_key])
                bench_20[f'{prefix}_fl'] = safe_mean(pm['flight'])
                bench_20[f'{prefix}_fl_std'] = safe_std(pm['flight'])

                for adv_key in ['smooth', 'max_exp', 'safe_marg']:
                    bench_20[f'{prefix}_{adv_key}'] = safe_mean(pm_adv[adv_key])
                    bench_20[f'{prefix}_{adv_key}_std'] = safe_std(pm_adv[adv_key])

    plt.style.use('default')

    # WYKRES 1: Krzywe Pareto
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    if res_d['len']:
        ax1.plot(res_d['len'], res_d['risk'], color='#4472C4', marker='o', linewidth=2.5, label='Dijkstra (Referencja)')
    if res_a['len']:
        ax1.plot(res_a['len'], res_a['risk'], color='#ED7D31', marker='s', linewidth=2.5, label='A* Standard')
    if res_r['len']:
        ax1.plot(res_r['len'], res_r['risk'], color='#70AD47', marker='D', linewidth=3.5, label='Risk-Aware A* (Kinem.)')

    ax1.set_title("Kompromis: Długość Trasy vs Ekspozycja na Ryzyko", fontsize=15, pad=15)
    ax1.set_xlabel("Długość Trasy [m]", fontsize=13)
    ax1.set_ylabel("Całkowity Koszt Ryzyka", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_pareto_3_algorithms.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # WYKRES 2: Wydajność
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    for res, color, marker, label in [(res_d, '#4472C4', 'o', 'Dijkstra'),
                                       (res_a, '#ED7D31', 's', 'A* Standard'),
                                       (res_r, '#70AD47', 'D', 'Risk-Aware A*')]:
        if res['time']:
            ax2a.plot(res['w'], res['time'], color=color, marker=marker, label=label)
            ax2b.plot(res['w'], res['nodes'], color=color, marker=marker, label=label)

    ax2a.set_title("Średni Czas Obliczeń", fontsize=13)
    ax2a.set_xlabel("Waga Ryzyka (W)"); ax2a.set_ylabel("Czas [ms]")
    ax2a.grid(True, linestyle='--', alpha=0.5); ax2a.legend()

    ax2b.set_title("Liczba Odwiedzonych Węzłów", fontsize=13)
    ax2b.set_xlabel("Waga Ryzyka (W)"); ax2b.set_ylabel("Węzły")
    ax2b.grid(True, linestyle='--', alpha=0.5); ax2b.legend()

    plt.suptitle("Analiza Wydajności Obliczeniowej", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_performance_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # WYKRES 3: Słupki W=0
    if bench_0:
        _plot_benchmark_bars(bench_0, "Izolacja Kinematyki (W=0)",
                             os.path.join(output_dir, "3_algorithm_comparison_W0.png"), "W=0")

    # WYKRES 4: Słupki W=20
    if bench_20:
        _plot_benchmark_bars(bench_20, f"Fair Benchmarking (W={int(RISK_WEIGHT)})",
                             os.path.join(output_dir, "4_algorithm_comparison_W20.png"),
                             f"W={int(RISK_WEIGHT)}")

    # WYKRES 5: Zaawansowane metryki
    if bench_20:
        fig_adv, axs_adv = plt.subplots(1, 3, figsize=(18, 6))
        labels_alg = ["Dijkstra", "A* Standard", "Risk-Aware A*"]
        colors_alg = ['#4472C4', '#ED7D31', '#70AD47']

        adv_plots = [
            ('smooth', "Indeks Płynności Trajektorii\n(Suma zmian kątów [°])", "Stopnie [°] (↓ Lepiej)", "°"),
            ('max_exp', "Ekspozycja Maksymalna\n(Najwyższe punktowe ryzyko)", "Wartość Ryzyka (↓ Lepiej)", ""),
            ('safe_marg', "Średni Margines Bezpieczeństwa\n(Dystans EDT od przeszkód)", "Kratki [EDT] (↑ Lepiej)", ""),
        ]

        for ax, (key, title, ylabel, suffix) in zip(axs_adv, adv_plots):
            vals = [bench_20[f'd_{key}'], bench_20[f'a_{key}'], bench_20[f'r_{key}']]

            ax.bar(labels_alg, vals, color=colors_alg, edgecolor='black')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(range(len(labels_alg)))
            ax.set_xticklabels(labels_alg, fontsize=11)

            max_v = max(vals) if max(vals) > 0 else 1
            fmt_str = ".1f" if key != 'max_exp' else ".2f"
            for i, v in enumerate(vals):
                ax.text(i, v + (max_v * 0.02),
                        f"{v:{fmt_str}}{suffix}", ha='center', fontweight='bold')

        fig_adv.suptitle(f"Zaawansowane Wskaźniki Jakości (W={int(RISK_WEIGHT)})", fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig_adv.subplots_adjust(top=0.85)
        fig_adv.savefig(os.path.join(output_dir, "5_advanced_metrics_W20.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_adv)

    print("Gotowe! Wykresy:", output_dir)
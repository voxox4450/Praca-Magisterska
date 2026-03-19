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
    calculate_segment_risk, calculate_path_length,
    calculate_kinematic_flight_time, compute_turn_cost, compute_safe_turn_speed,
    collision_radius_for_mass, drone_radius_for_mass,
    sensor_range_for_mass, processing_delay_for_mass
)
from config import (
    V_MAX_MS, ACCELERATION, MAX_LATERAL_ACCEL, MIN_TURN_SPEED,
    RISK_WEIGHT, TURN_PENALTY,
    COLLISION_RADIUS,
    DRONE_MASS_KG, MAX_THRUST_NET_N
)


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
    # cbar_speed.set_label('Prędkość Kinematyczna [m/s]', color='white', labelpad=10)
    cax_speed.text(-0.02, 0.5, 'Prędkość [m/s] ',
                   transform=cax_speed.transAxes,
                   va='center', ha='right', color='white')
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
def compute_path_speeds(path: List[Tuple[int, int]], initial_speed: float = 0.0,
                        accel: float = ACCELERATION, stop_at_end: bool = True) -> np.ndarray:
    if len(path) < 2:
        return np.array([initial_speed] * len(path))

    v_max = V_MAX_MS
    a = accel

    speeds = np.zeros(len(path))
    turn_speeds = np.full(len(path), v_max)

    turn_speeds[0] = initial_speed
    if stop_at_end:
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


def smooth_path_with_speeds(path: List[Tuple[int, int]], speeds: np.ndarray,
                            accel: float = ACCELERATION) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    if len(path) < 3:
        return np.array([p[0] for p in path]), np.array([p[1] for p in path]), speeds

    x, y = [p[0] for p in path], [p[1] for p in path]
    try:
        tck, u = interp.splprep([x, y], s=3.0, k=3)
        u_new = np.linspace(0, 1, num=len(path) * 10)
        x_smooth, y_smooth = interp.splev(u_new, tck)
        speeds_smooth = interp.interp1d(u, speeds, kind='linear')(u_new)

        a = accel
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
# Po kliknięciu przeszkody otwiera 3 okna:
#   1. Risk-Aware A* (kinematyka) — hamuje, zwalnia, omija płynnie
#   2. Dijkstra (brak kinematyki) — skręca 90° przy pełnej prędkości → rozpad
#   3. A* Standard (brak kinematyki) — jak Dijkstra, ostry zakręt → rozpad
# ─────────────────────────────────────────────────────────────────────────────
def run_online_simulation(
        env: GridMap, start: Tuple[int, int], goal: Tuple[int, int],
        search_func: Callable, collision_radius: float,
        func_dijkstra: Callable = None,
        func_astar: Callable = None
) -> None:
    """
    Tryb online z porównaniem 3 algorytmów.
    search_func = Risk-Aware A* (główny planista z kinematyką).
    func_dijkstra, func_astar = algorytmy referencyjne (bez kinematyki).
    """
    path_global, stats_global = search_func(env, start, goal, risk_weight=RISK_WEIGHT,
                                            turn_penalty=TURN_PENALTY, drone_radius=collision_radius)

    if not path_global:
        print("Błąd: Nie znaleziono trasy startowej.")
        return

    # ── OKNO GŁÓWNE: planowanie wstępne (Risk-Aware A*) ──────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title("Risk-Aware A*")
    plt.subplots_adjust(bottom=0.18, right=0.80, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
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

    ax_slider = plt.axes([0.15, 0.07, 0.60, 0.03], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka [W] ', valmin=0.0, valmax=40.0,
                         valinit=RISK_WEIGHT, valstep=1.0, color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    ax_mass_slider = plt.axes([0.15, 0.02, 0.60, 0.03], facecolor='#333333')
    mass_slider = Slider(ax=ax_mass_slider, label='Masa Drona [kg] ', valmin=1.0, valmax=50.0,
                         valinit=DRONE_MASS_KG, valstep=1.0, color='orange')
    mass_slider.label.set_color('white')
    mass_slider.valtext.set_color('white')

    sim_state = {
        "clicked": False, "drone_pos": None, "target_pos": None, "mode": "IDLE",
        "base_dist": stats_global['length'], "base_time": stats_global.get('flight_time', 0),
        "base_risk": stats_global['risk'], "base_turns": stats_global.get('turns', 0),
        "flown_dist": 0.0, "flown_time": 0.0, "flown_risk": 0.0, "flown_turns": 0,
        "buffer_points": [],
        "path_global": path_global,
        "path_global_original": list(path_global),
        "global_speeds": global_speeds,
        "obstacle_pos": None
    }

    def update_route(val):
        if sim_state["mode"] == "IDLE" and not sim_state["clicked"]:
            w = risk_slider.val
            m = mass_slider.val
            a_cur = MAX_THRUST_NET_N / m
            col_r = collision_radius_for_mass(m)
            phys_r = drone_radius_for_mass(m)
            env.update_drone_footprint(phys_r, col_r)

            new_path, new_stats = search_func(env, start, goal, risk_weight=w,
                                              turn_penalty=TURN_PENALTY,
                                              drone_radius=col_r,
                                              drone_mass=m)
            if new_path:
                sim_state["path_global"] = new_path
                sim_state["path_global_original"] = list(new_path)
                new_speeds = compute_path_speeds(new_path, accel=a_cur)
                sim_state["global_speeds"] = new_speeds
                sim_state["base_dist"] = new_stats['length']
                sim_state["base_time"] = new_stats.get('flight_time', 0)
                sim_state["base_risk"] = new_stats['risk']
                sim_state["base_turns"] = new_stats.get('turns', 0)

                gx_s, gy_s = smooth_path_bspline(new_path)
                line_global.set_data(gx_s, gy_s)

                turns_g = new_stats.get('turns', 0)
                flight_t = calculate_kinematic_flight_time(new_path, mass=m)
                dr = drone_radius_for_mass(m)
                ax.set_title(
                    f"Optymalizacja tras BSP (Risk-Aware A*, W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a_cur:.1f} m/s²)\n"
                    f"Droga: {new_stats['length']:.1f} m | Czas: {flight_t:.1f} s | "
                    f"Ryzyko: {new_stats['risk']:.1f} | Zakręty: {turns_g}",
                    fontsize=14, color='white', pad=25)
            else:
                ax.set_title("BRAK TRASY GLOBALNEJ!", color='red', fontsize=14, pad=25)
            fig.canvas.draw_idle()
            return

        if sim_state["mode"] == "IGNORE" or sim_state.get("obstacle_pos") is None:
            return

        click_x, click_y = sim_state["obstacle_pos"]
        OBSTACLE_RADIUS = 8

        w = risk_slider.val
        m = mass_slider.val
        a_current = MAX_THRUST_NET_N / m

        current_sensor_range = sensor_range_for_mass(m)
        processing_delay = processing_delay_for_mass(m)

        col_radius = collision_radius_for_mass(m)
        phys_radius = drone_radius_for_mass(m)
        env.update_drone_footprint(phys_radius, col_radius)

        path_global = sim_state["path_global"]

        # ─── ZMIANA: PRZELICZENIE PRĘDKOŚCI DLA NOWEJ MASY ───
        # To gwarantuje, że "zielona droga" oraz szary plan pierwotny "żyją" i wyginają się poprawnie!
        global_speeds = compute_path_speeds(path_global, accel=a_current)
        sim_state["global_speeds"] = global_speeds

        # Aktualizacja samej szarej linii w tle
        gx_s, gy_s, _ = smooth_path_with_speeds(path_global, global_speeds, accel=a_current)
        line_global.set_data(gx_s, gy_s)
        # ──────────────────────────────────────────────────────

        collision_idx = -1
        for i, (px, py) in enumerate(path_global):
            if np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2) <= current_sensor_range:
                collision_idx = i
                break

        if collision_idx == -1:
            ax.set_title("KATASTROFA! Zbyt mały zasięg czujnika!", color='red', fontsize=15, fontweight='bold')
            lc_new_bg.set_segments([])
            lc_new.set_segments([])
            sim_state["mode"] = "CRASH"
            fig.canvas.draw_idle()
            return

        v_detect = float(global_speeds[collision_idx])
        t_stop = v_detect / a_current
        t_react = min(processing_delay, t_stop)
        reaction_distance_meters = (v_detect * t_react) - (0.5 * a_current * (t_react ** 2))
        reaction_indices = int(np.ceil(reaction_distance_meters))
        v_react_end = max(0.0, v_detect - a_current * processing_delay)

        drone_detect_idx = collision_idx
        drone_react_idx = min(drone_detect_idx + reaction_indices, len(path_global) - 1)

        reaction_path = path_global[drone_detect_idx:drone_react_idx + 1]
        drone_pos = path_global[drone_react_idx]

        # ─── 1. PERFEKCYJNE SKLEJENIE ZIELONEJ I POMARAŃCZOWEJ LINII ───
        end_idx_ext = min(drone_react_idx + 1 + 2, len(path_global))
        flown_raw_ext = path_global[:end_idx_ext]
        f_speeds_ext = global_speeds[:end_idx_ext]

        sx_ext, sy_ext, ss_ext = smooth_path_with_speeds(flown_raw_ext, f_speeds_ext, accel=a_current)

        det_px, det_py = path_global[drone_detect_idx]
        idx_det = np.argmin((sx_ext - det_px) ** 2 + (sy_ext - det_py) ** 2)

        react_px, react_py = path_global[drone_react_idx]
        idx_react = np.argmin((sx_ext - react_px) ** 2 + (sy_ext - react_py) ** 2)

        dot_x, dot_y = sx_ext[idx_det], sy_ext[idx_det]
        end_orange_x, end_orange_y = sx_ext[idx_react], sy_ext[idx_react]

        # Zielona linia (przebyta)
        sx_green, sy_green, ss_green = sx_ext[:idx_det + 1], sy_ext[:idx_det + 1], ss_ext[:idx_det + 1]
        if len(sx_green) > 1:
            pts_f = np.array([sx_green, sy_green]).T.reshape(-1, 1, 2)
            segs_f = np.concatenate([pts_f[:-1], pts_f[1:]], axis=1)
            lc_flown_bg.set_segments(segs_f)
            lc_flown.set_segments(segs_f)
            lc_flown.set_array((ss_green[:-1] + ss_green[1:]) / 2.0)
        else:
            lc_flown_bg.set_segments([])
            lc_flown.set_segments([])

        # Kropka i pomarańczowa linia
        drone_marker.set_data([dot_x], [dot_y])
        line_reaction.set_data(sx_ext[idx_det:idx_react + 1], sy_ext[idx_det:idx_react + 1])

        CRASH_DIST = OBSTACLE_RADIUS + col_radius
        crash = any(np.sqrt((click_x - px) ** 2 + (click_y - py) ** 2) <= CRASH_DIST for px, py in reaction_path)

        if crash:
            ax.set_title("KATASTROFA! Zbyt duża bezwładność drona!", color='red', fontsize=15, fontweight='bold')
            sim_state["mode"] = "CRASH"
            lc_new_bg.set_segments([])
            lc_new.set_segments([])
            fig.canvas.draw_idle()
            return

        if len(reaction_path) >= 2:
            last_dx = reaction_path[-1][0] - reaction_path[-2][0]
            last_dy = reaction_path[-1][1] - reaction_path[-2][1]
            heading = (int(np.sign(last_dx)), int(np.sign(last_dy)))
        else:
            heading = (0, 0)

        buffer_points = []
        buffer_dist = 0.0
        if heading != (0, 0) and v_react_end > 0:
            r_45 = 1.5 / max(0.1, math.sin(math.radians(22.5)))
            v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_45))
            v_safe_ref = min(v_safe_ref, V_MAX_MS)

            braking_dist_needed = max(0.0, (v_react_end ** 2 - v_safe_ref ** 2) / (
                    2 * a_current)) if v_react_end > v_safe_ref else 0.0
            step_len = math.sqrt(heading[0] ** 2 + heading[1] ** 2)
            buffer_steps = max(1, int(math.ceil(braking_dist_needed / step_len)))

            for d in range(1, buffer_steps + 1):
                bp = (drone_pos[0] + heading[0] * d, drone_pos[1] + heading[1] * d)
                bx, by = int(bp[0]), int(bp[1])
                if 0 <= bx < env.width and 0 <= by < env.height:
                    if not env.collision_mask[bx, by]:
                        buffer_points.append(bp)
                        buffer_dist += step_len
                    else:
                        break
                else:
                    break

        v_at_buffer_end = max(0.0, math.sqrt(max(0.0, v_react_end ** 2 - 2 * a_current * buffer_dist)))
        search_start = buffer_points[-1] if buffer_points else drone_pos
        sim_state["drone_detect_idx"] = drone_detect_idx

        path_local, stats = search_func(env, search_start, goal, risk_weight=w,
                                        turn_penalty=TURN_PENALTY, drone_radius=col_radius,
                                        initial_direction=heading,
                                        current_speed=v_at_buffer_end,
                                        initial_straight_dist=buffer_dist,
                                        drone_mass=m)

        if path_local and stats['found']:
            sim_state["target_pos"] = goal
            sim_state["mode"] = "NORMAL"
        else:
            path_local, stats = search_func(env, search_start, start, risk_weight=40.0,
                                            turn_penalty=TURN_PENALTY, drone_radius=col_radius,
                                            initial_direction=heading,
                                            current_speed=v_at_buffer_end,
                                            initial_straight_dist=buffer_dist,
                                            drone_mass=m)
            sim_state["target_pos"] = start
            sim_state["mode"] = "RTH"

        if path_local and stats.get('found'):
            if buffer_points:
                full_new_path = [drone_pos] + buffer_points + path_local[1:]
            else:
                full_new_path = path_local

            speeds = compute_path_speeds(full_new_path, initial_speed=v_react_end, accel=a_current)

            artificial_lead_in = []
            if heading != (0, 0):
                p1 = (drone_pos[0] - heading[0] * 2.0, drone_pos[1] - heading[1] * 2.0)
                p2 = (drone_pos[0] - heading[0] * 1.0, drone_pos[1] - heading[1] * 1.0)
                artificial_lead_in = [p1, p2]

            # ─── 2. PERFEKCYJNE SKLEJENIE NOWEJ TRASY Z POMARAŃCZOWĄ LINIĄ ───
            if artificial_lead_in:
                combined_path = artificial_lead_in + full_new_path
                lead_speeds = np.full(len(artificial_lead_in), v_react_end)
                combined_speeds = np.concatenate([lead_speeds, speeds])

                sx, sy, s_speeds = smooth_path_with_speeds(combined_path, combined_speeds, accel=a_current)

                search_limit = min(len(sx), 60)
                distances = (sx[:search_limit] - react_px) ** 2 + (sy[:search_limit] - react_py) ** 2
                best_idx = np.argmin(distances)

                sx = sx[best_idx:]
                sy = sy[best_idx:]
                s_speeds = s_speeds[best_idx:]

                sx[0] = end_orange_x
                sy[0] = end_orange_y
            else:
                sx, sy, s_speeds = smooth_path_with_speeds(full_new_path, speeds, accel=a_current)
                sx[0] = end_orange_x
                sy[0] = end_orange_y

            points = np.array([sx, sy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc_new_bg.set_segments(segments)
            lc_new.set_segments(segments)
            lc_new.set_array((s_speeds[:-1] + s_speeds[1:]) / 2.0)

            # ─── ZMIANA: PRZELICZENIE STATYSTYK DROGI PRZEBYTEJ ───
            flown_path_for_stats = path_global[:drone_detect_idx + 1]
            sim_state["flown_dist"] = calculate_path_length(flown_path_for_stats)
            sim_state["flown_time"] = calculate_kinematic_flight_time(flown_path_for_stats, mass=m)
            sim_state["flown_risk"] = calculate_segment_risk(flown_path_for_stats, env)

            actual_flown_turns = 0
            if len(flown_path_for_stats) > 2:
                last_dir = (flown_path_for_stats[1][0] - flown_path_for_stats[0][0],
                            flown_path_for_stats[1][1] - flown_path_for_stats[0][1])
                for i in range(2, len(flown_path_for_stats)):
                    curr_dir = (flown_path_for_stats[i][0] - flown_path_for_stats[i - 1][0],
                                flown_path_for_stats[i][1] - flown_path_for_stats[i - 1][1])
                    if curr_dir != last_dir:
                        actual_flown_turns += 1
                        last_dir = curr_dir
            sim_state["flown_turns"] = actual_flown_turns
            # ───────────────────────────────────────────────────────

            actual_new_dist = calculate_path_length(full_new_path)
            t_dist = sim_state["flown_dist"] + actual_new_dist

            f_turns = 0
            if len(full_new_path) > 2:
                last_dir = (full_new_path[1][0] - full_new_path[0][0],
                            full_new_path[1][1] - full_new_path[0][1])
                for i in range(2, len(full_new_path)):
                    curr_dir = (full_new_path[i][0] - full_new_path[i - 1][0],
                                full_new_path[i][1] - full_new_path[i - 1][1])
                    if curr_dir != last_dir:
                        f_turns += 1
                        last_dir = curr_dir

            t_time = sim_state["flown_time"] + calculate_kinematic_flight_time(full_new_path, mass=m)
            t_risk = sim_state["flown_risk"] + calculate_segment_risk(full_new_path, env)
            t_turns = sim_state["flown_turns"] + f_turns

            # Przelicz baseline z aktualną masą (oryginalna trasa bez przeszkody)
            orig_path = sim_state["path_global_original"]
            base_dist = calculate_path_length(orig_path)
            base_time = calculate_kinematic_flight_time(orig_path, mass=m)
            base_risk = calculate_segment_risk(orig_path, env)
            base_turns_count = 0
            if len(orig_path) > 2:
                _ld = (orig_path[1][0] - orig_path[0][0], orig_path[1][1] - orig_path[0][1])
                for _i in range(2, len(orig_path)):
                    _cd = (orig_path[_i][0] - orig_path[_i-1][0], orig_path[_i][1] - orig_path[_i-1][1])
                    if _cd != _ld:
                        base_turns_count += 1
                        _ld = _cd

            d_dist = t_dist - base_dist
            d_time = t_time - base_time
            d_risk = t_risk - base_risk
            d_turns = t_turns - base_turns_count

            fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
            fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

            stats_text = (f"Droga: {t_dist:.1f} m ({fmt(d_dist)} m nadłożono)\n"
                          f"Czas: {t_time:.1f}s ({fmt(d_time)}) | Ryzyko: {t_risk:.1f} ({fmt(d_risk)}) | "
                          f"Zakręty: {t_turns} ({fmt_i(d_turns)})")

            if sim_state["mode"] == "RTH":
                dr = drone_radius_for_mass(m)
                ax.set_title(
                    f"Risk-Aware A* — Tryb Powrotu (W=40, m={m:.0f} kg, r={dr:.2f} m, a={a_current:.1f} m/s²)\n{stats_text}",
                    color='orange', fontsize=14, pad=25)
                lc_new.set_cmap(get_speed_cmap())
                line_proxy.set_color('orange')
            else:
                dr = drone_radius_for_mass(m)
                ax.set_title(
                    f"Risk-Aware A* — Omijanie (W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a_current:.1f} m/s²)\n{stats_text}",
                    color='lime', fontsize=14, pad=25)
                lc_new.set_cmap(get_speed_cmap())
                line_proxy.set_color('cyan')
        else:
            ax.set_title("DRON JEST UWIĘZIONY!", color='red', fontsize=14, pad=25)
            lc_new_bg.set_segments([])
            lc_new.set_segments([])
        fig.canvas.draw_idle()

    risk_slider.on_changed(update_route)
    mass_slider.on_changed(update_route)

    def onclick(event):
        if event.inaxes != ax or sim_state["clicked"]:
            return

        path_global = sim_state["path_global"]
        global_speeds = sim_state["global_speeds"]

        click_x, click_y = int(event.xdata), int(event.ydata)
        OBSTACLE_RADIUS = 8
        sim_state["obstacle_pos"] = (click_x, click_y)
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        onclick_mass_early = mass_slider.val
        onclick_col_r_early = collision_radius_for_mass(onclick_mass_early)
        onclick_phys_r_early = drone_radius_for_mass(onclick_mass_early)
        DRONE_RADIUS = onclick_phys_r_early
        CRASH_DIST = OBSTACLE_RADIUS + onclick_col_r_early

        is_path_blocked = any(
            np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2) <= CRASH_DIST
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

        onclick_mass = mass_slider.val
        current_sensor_range = sensor_range_for_mass(onclick_mass)
        processing_delay = processing_delay_for_mass(onclick_mass)
        onclick_accel = MAX_THRUST_NET_N / onclick_mass

        collision_idx = -1
        for i, (px, py) in enumerate(path_global):
            if np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2) <= current_sensor_range:
                collision_idx = i
                break

        if collision_idx == -1:
            return

        v_detect = float(global_speeds[collision_idx])
        print(f"\n-> WYKRYTO ZAGROŻENIE! Prędkość: {v_detect:.1f} m/s")

        t_stop = v_detect / onclick_accel
        t_react = min(processing_delay, t_stop)
        reaction_distance_meters = (v_detect * t_react) - (0.5 * onclick_accel * (t_react ** 2))
        reaction_indices = int(np.ceil(reaction_distance_meters))
        v_react_end = max(0.0, v_detect - onclick_accel * processing_delay)
        print(f"-> Po reakcji ({processing_delay}s): {reaction_distance_meters:.1f}m, v={v_react_end:.1f} m/s")

        drone_detect_idx = collision_idx
        drone_react_idx = min(drone_detect_idx + reaction_indices, len(path_global) - 1)
        reaction_path = path_global[drone_detect_idx:drone_react_idx + 1]
        current_drone_pos = path_global[drone_react_idx]

        # ─── 3. PERFEKCYJNE RYSOWANIE CRASH-U W ONCLICK ───
        end_idx_ext = min(drone_react_idx + 1 + 2, len(path_global))
        flown_raw_ext = path_global[:end_idx_ext]
        f_speeds_ext = global_speeds[:end_idx_ext]

        sx_ext, sy_ext, ss_ext = smooth_path_with_speeds(flown_raw_ext, f_speeds_ext, accel=onclick_accel)

        det_px, det_py = path_global[drone_detect_idx]
        idx_det = np.argmin((sx_ext - det_px) ** 2 + (sy_ext - det_py) ** 2)

        react_px, react_py = path_global[drone_react_idx]
        idx_react = np.argmin((sx_ext - react_px) ** 2 + (sy_ext - react_py) ** 2)

        drone_marker.set_data([sx_ext[idx_det]], [sy_ext[idx_det]])
        line_reaction.set_data(sx_ext[idx_det:idx_react + 1], sy_ext[idx_det:idx_react + 1])

        sx_green, sy_green, ss_green = sx_ext[:idx_det + 1], sy_ext[:idx_det + 1], ss_ext[:idx_det + 1]
        if len(sx_green) > 1:
            pts_f = np.array([sx_green, sy_green]).T.reshape(-1, 1, 2)
            segs_f = np.concatenate([pts_f[:-1], pts_f[1:]], axis=1)
            lc_flown_bg.set_segments(segs_f)
            lc_flown.set_segments(segs_f)
            lc_flown.set_array((ss_green[:-1] + ss_green[1:]) / 2.0)

        if len(reaction_path) >= 2:
            last_dx = reaction_path[-1][0] - reaction_path[-2][0]
            last_dy = reaction_path[-1][1] - reaction_path[-2][1]
            flight_heading = (int(np.sign(last_dx)), int(np.sign(last_dy)))
        else:
            flight_heading = (0, 0)

        onclick_col_r = collision_radius_for_mass(onclick_mass)

        buffer_points = []
        buffer_dist = 0.0
        if flight_heading != (0, 0):
            r_45 = 1.5 / max(0.1, math.sin(math.radians(22.5)))
            v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_45))
            v_safe_ref = min(v_safe_ref, V_MAX_MS)

            braking_dist_needed = max(0.0, (v_react_end ** 2 - v_safe_ref ** 2) / (2 * onclick_accel)) \
                if v_react_end > v_safe_ref else 0.0

            step_len = math.sqrt(flight_heading[0] ** 2 + flight_heading[1] ** 2)
            buffer_steps = max(1, int(math.ceil(braking_dist_needed / step_len)))

            for d in range(1, buffer_steps + 1):
                bp = (current_drone_pos[0] + flight_heading[0] * d,
                      current_drone_pos[1] + flight_heading[1] * d)
                bx, by = int(bp[0]), int(bp[1])
                if 0 <= bx < env.width and 0 <= by < env.height:
                    if not env.is_collision(bx, by, drone_radius=onclick_col_r):
                        buffer_points.append(bp)
                        buffer_dist += step_len
                    else:
                        break
                else:
                    break

        v_at_buffer_end = max(0.0, math.sqrt(max(0.0, v_react_end ** 2 - 2 * onclick_accel * buffer_dist)))

        sim_state["drone_pos"] = current_drone_pos
        sim_state["buffer_points"] = buffer_points
        sim_state["heading"] = flight_heading
        sim_state["drone_speed"] = v_at_buffer_end
        sim_state["visual_speed"] = v_react_end
        sim_state["buffer_dist"] = buffer_dist
        sim_state["drone_detect_idx"] = drone_detect_idx
        sim_state["click_mass"] = mass_slider.val

        flown_full = path_global[:drone_react_idx + 1]
        sim_state["flown_dist"] = calculate_path_length(flown_full)
        sim_state["flown_time"] = calculate_kinematic_flight_time(flown_full, mass=onclick_mass)
        sim_state["flown_risk"] = calculate_segment_risk(flown_full, env)

        f_turns = 0
        if len(flown_full) > 2:
            last_dir = (flown_full[1][0] - flown_full[0][0], flown_full[1][1] - flown_full[0][1])
            for i in range(2, len(flown_full)):
                curr_dir = (flown_full[i][0] - flown_full[i - 1][0], flown_full[i][1] - flown_full[i - 1][1])
                if curr_dir != last_dir:
                    f_turns += 1
                    last_dir = curr_dir
        sim_state["flown_turns"] = f_turns

        crash = any(
            np.sqrt((click_x - px) ** 2 + (click_y - py) ** 2) <= (OBSTACLE_RADIUS + DRONE_RADIUS)
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
        click_m = mass_slider.val
        click_col_r = collision_radius_for_mass(click_m)
        click_phys_r = drone_radius_for_mass(click_m)
        env.update_drone_footprint(click_phys_r, click_col_r)

        path_check, _ = search_func(env, search_start, goal, risk_weight=risk_slider.val,
                                    turn_penalty=TURN_PENALTY, drone_radius=click_col_r,
                                    initial_direction=flight_heading, current_speed=v_at_buffer_end,
                                    initial_straight_dist=buffer_dist,
                                    drone_mass=click_m)

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

        if func_dijkstra is not None and func_astar is not None:
            _open_comparison_windows(
                env=env, start=start, goal=goal,
                path_global=path_global, global_speeds=global_speeds,
                click_x=click_x, click_y=click_y,
                obstacle_radius=OBSTACLE_RADIUS,
                drone_radius=DRONE_RADIUS,
                collision_radius=collision_radius,
                drone_detect_idx=drone_detect_idx,
                drone_react_idx=drone_react_idx,
                reaction_path=reaction_path,
                current_drone_pos=current_drone_pos,
                flight_heading=flight_heading,
                v_detect=v_detect,
                v_react_end=v_react_end,
                func_dijkstra=func_dijkstra,
                func_astar=func_astar,
                sim_state=sim_state,
                init_w=risk_slider.val,
                init_mass=mass_slider.val,
            )

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    print("\n Kliknij na mapę aby dodać przeszkodę dynamiczną.")
    print("   Po kliknięciu otworzą się 3 okna porównawcze.\n")
    plt.show(block=True)


def _open_comparison_windows(
        env, start, goal, path_global, global_speeds,
        click_x, click_y, obstacle_radius, drone_radius,
        collision_radius, drone_detect_idx, drone_react_idx,
        reaction_path, current_drone_pos, flight_heading,
        v_detect, v_react_end,
        func_dijkstra, func_astar,
        sim_state=None,
        init_w=RISK_WEIGHT,
        init_mass=DRONE_MASS_KG
) -> None:
    """
    Otwiera 2 dodatkowe okna: Dijkstra i A* Standard.
    Każde okno ma suwak Wagi Ryzyka (W) i suwak Masy Drona [kg].
    """

    algorithms = [
        ("Dijkstra", func_dijkstra, '#4472C4'),
        ("A* Standard", func_astar, '#ED7D31'),
    ]

    for algo_name, algo_func, color in algorithms:

        fig_cmp, ax_cmp = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.18, right=0.80, left=0.15, top=0.90)
        setup_dark_theme(fig_cmp, ax_cmp)

        img_cmp = ax_cmp.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
        ax_cmp.set_xlim(0, env.width)
        ax_cmp.set_ylim(0, env.height)
        _setup_ui_colorbars(fig_cmp, ax_cmp, img_cmp,
                            speed_axes_rect=[0.155, 0.13, 0.59, 0.02],
                            risk_fraction=0.048, risk_pad=0.045, risk_shrink=0.72)

        # Oryginalna trasa (szara przerywana)
        gx_s, gy_s = smooth_path_bspline(path_global)
        ax_cmp.plot(gx_s, gy_s, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                    label='Pierwotny Plan')

        # Droga przebyta (do wykrycia) — dynamiczna, aktualizowana suwakiem masy
        flown_raw = path_global[:drone_detect_idx + 1]
        f_speeds = global_speeds[:drone_detect_idx + 1]
        lc_f_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
        lc_f = LineCollection([], cmap=get_speed_cmap(), norm=plt.Normalize(0, V_MAX_MS),
                              linewidths=5, zorder=4)
        ax_cmp.add_collection(lc_f_bg)
        ax_cmp.add_collection(lc_f)

        if len(flown_raw) >= 2:
            sx_f, sy_f, ss_f = smooth_path_with_speeds(flown_raw, f_speeds)
            pts_f = np.array([sx_f, sy_f]).T.reshape(-1, 1, 2)
            segs_f = np.concatenate([pts_f[:-1], pts_f[1:]], axis=1)
            lc_f_bg.set_segments(segs_f)
            lc_f.set_segments(segs_f)
            lc_f.set_array((ss_f[:-1] + ss_f[1:]) / 2.0)

        # === POPRAWIONE WCIĘCIE: Kod poniżej "if" wraca na właściwy poziom ===

        # Czas reakcji (pomarańczowa) i Punkt wykrycia przypisujemy do zmiennych
        line_reaction_cmp, = ax_cmp.plot([], [], color='orange', linestyle=':', linewidth=4, label='Czas Reakcji',
                                         zorder=4)
        detect_dot_cmp, = ax_cmp.plot([], [], 'o', color='yellow', markersize=12, markeredgecolor='black', zorder=6,
                                      label='Punkt Wykrycia')

        # Start i cel
        ax_cmp.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
        ax_cmp.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                       zorder=5)

        # Replanowana trasa (dynamiczna)
        lc_n_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=4)
        lc_n = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=5)
        ax_cmp.add_collection(lc_n_bg)
        ax_cmp.add_collection(lc_n)
        ax_cmp.plot([], [], color='cyan', linewidth=5, label='Replanowana Trasa',
                    path_effects=[pe.withStroke(linewidth=7, foreground="#555555")])

        legend_cmp = ax_cmp.legend(loc='lower left', bbox_to_anchor=(1.04, -0.01), facecolor='#333333',
                                   edgecolor='white', title=algo_name)
        plt.setp(legend_cmp.get_texts(), color='white')
        plt.setp(legend_cmp.get_title(), color='white')

        ax_slider_cmp = plt.axes([0.15, 0.07, 0.60, 0.03], facecolor='#333333')
        slider_cmp = Slider(ax=ax_slider_cmp, label='Waga Ryzyka [W] ', valmin=0.0, valmax=40.0,
                            valinit=init_w, valstep=1.0, color=color)
        slider_cmp.label.set_color('white')
        slider_cmp.valtext.set_color('white')

        ax_mass_cmp = plt.axes([0.15, 0.02, 0.60, 0.03], facecolor='#333333')
        mass_slider_cmp = Slider(ax=ax_mass_cmp, label='Masa Drona [kg] ', valmin=1.0, valmax=50.0,
                                 valinit=init_mass, valstep=1.0, color='orange')
        mass_slider_cmp.label.set_color('white')
        mass_slider_cmp.valtext.set_color('white')

        def make_update(ax_ref, lc_bg_ref, lc_ref, lc_f_bg_ref, lc_f_ref,
                        path_global_ref, global_speeds_ref, func_ref, name_ref, clr_ref,
                        w_slider, m_slider, sim_st, react_line, dot_line):
            def update_cmp(val):
                w = w_slider.val
                m = m_slider.val
                a_val = MAX_THRUST_NET_N / m

                if sim_st.get("obstacle_pos") is None: return
                cx, cy = sim_st["obstacle_pos"]
                OBSTACLE_RADIUS = 8

                # Dynamiczna detekcja w oknach pobocznych!
                current_sensor_range = sensor_range_for_mass(m)
                processing_delay = processing_delay_for_mass(m)
                col_r = collision_radius_for_mass(m)
                phys_r = drone_radius_for_mass(m)
                env.update_drone_footprint(phys_r, col_r)

                c_idx = -1
                for i, (px, py) in enumerate(path_global_ref):
                    if np.sqrt((px - cx) ** 2 + (py - cy) ** 2) <= current_sensor_range:
                        c_idx = i
                        break

                if c_idx == -1:
                    ax_ref.set_title("KATASTROFA! Zbyt mały zasięg czujnika!", color='red', fontsize=14, pad=25)
                    lc_bg_ref.set_segments([])
                    lc_ref.set_segments([])
                    react_line.set_data([], [])
                    dot_line.set_data([], [])
                    ax_ref.figure.canvas.draw_idle()
                    return

                v_det = float(global_speeds_ref[c_idx])
                t_stop = v_det / a_val
                t_react = min(processing_delay, t_stop)
                react_dist = (v_det * t_react) - (0.5 * a_val * (t_react ** 2))
                react_idx_count = int(np.ceil(react_dist))
                v_react_end = max(0.0, v_det - a_val * processing_delay)

                det_idx = c_idx
                react_idx = min(det_idx + react_idx_count, len(path_global_ref) - 1)

                react_path = path_global_ref[det_idx:react_idx + 1]
                cur_drone_pos = path_global_ref[react_idx]

                react_line.set_data([p[0] for p in react_path], [p[1] for p in react_path])
                dot_line.set_data([path_global_ref[det_idx][0]], [path_global_ref[det_idx][1]])

                CRASH_DIST = OBSTACLE_RADIUS + col_r
                if any(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) <= CRASH_DIST for px, py in react_path):
                    ax_ref.set_title("KATASTROFA! Zbyt duża bezwładność drona!", color='red', fontsize=14, pad=25)
                    lc_bg_ref.set_segments([])
                    lc_ref.set_segments([])
                    ax_ref.figure.canvas.draw_idle()
                    return

                # Rysowanie ogona z mechanicznym "przyklejaniem"
                if det_idx > 0:
                    full_spd = compute_path_speeds(path_global_ref, accel=a_val)
                    end_idx = min(det_idx + 1 + 2, len(path_global_ref))
                    ext_raw = path_global_ref[:end_idx]
                    ext_spd = full_spd[:end_idx]
                    sx_fl, sy_fl, ss_fl = smooth_path_with_speeds(ext_raw, ext_spd, accel=a_val)

                    detect_px, detect_py = path_global_ref[det_idx]
                    dists_fl = (sx_fl - detect_px) ** 2 + (sy_fl - detect_py) ** 2
                    best_fl_idx = np.argmin(dists_fl)

                    sx_fl = np.concatenate((sx_fl[:best_fl_idx], [detect_px]))
                    sy_fl = np.concatenate((sy_fl[:best_fl_idx], [detect_py]))
                    ss_fl = np.concatenate((ss_fl[:best_fl_idx], [ss_fl[best_fl_idx]]))

                    if len(sx_fl) > 1:
                        pts_fl = np.array([sx_fl, sy_fl]).T.reshape(-1, 1, 2)
                        segs_fl = np.concatenate([pts_fl[:-1], pts_fl[1:]], axis=1)
                        lc_f_bg_ref.set_segments(segs_fl)
                        lc_f_ref.set_segments(segs_fl)
                        lc_f_ref.set_array((ss_fl[:-1] + ss_fl[1:]) / 2.0)
                    else:
                        lc_f_bg_ref.set_segments([])
                        lc_f_ref.set_segments([])

                if len(react_path) >= 2:
                    last_dx = react_path[-1][0] - react_path[-2][0]
                    last_dy = react_path[-1][1] - react_path[-2][1]
                    f_heading = (int(np.sign(last_dx)), int(np.sign(last_dy)))
                else:
                    f_heading = (0, 0)

                path_replan, stats_replan = func_ref(env, cur_drone_pos, goal, risk_weight=w,
                                                     turn_penalty=TURN_PENALTY, drone_radius=col_r,
                                                     initial_direction=f_heading, current_speed=v_react_end,
                                                     drone_mass=m)

                if path_replan and stats_replan['found']:
                    artificial_lead_in = []
                    if f_heading != (0, 0):
                        p1 = (cur_drone_pos[0] - f_heading[0] * 2.0, cur_drone_pos[1] - f_heading[1] * 2.0)
                        p2 = (cur_drone_pos[0] - f_heading[0] * 1.0, cur_drone_pos[1] - f_heading[1] * 1.0)
                        artificial_lead_in = [p1, p2]

                    speeds_r_raw = compute_path_speeds(path_replan, initial_speed=v_react_end, accel=a_val)

                    if artificial_lead_in:
                        combined_path = artificial_lead_in + path_replan
                        lead_speeds = np.full(len(artificial_lead_in), v_react_end)
                        combined_speeds = np.concatenate([lead_speeds, speeds_r_raw])

                        sx_r, sy_r, ss_r = smooth_path_with_speeds(combined_path, combined_speeds, accel=a_val)

                        start_px, start_py = cur_drone_pos
                        search_limit = min(len(sx_r), 60)
                        dists_r = (sx_r[:search_limit] - start_px) ** 2 + (sy_r[:search_limit] - start_py) ** 2
                        best_idx = np.argmin(dists_r)

                        # CZYSTE CIĘCIE ZAMIAST CONCATENATE
                        sx_r = sx_r[best_idx:]
                        sy_r = sy_r[best_idx:]
                        ss_r = ss_r[best_idx:]

                        sx_r[0] = start_px
                        sy_r[0] = start_py
                    else:
                        sx_r, sy_r, ss_r = smooth_path_with_speeds(path_replan, speeds_r_raw, accel=a_val)
                        start_px, start_py = cur_drone_pos
                        sx_r[0] = start_px
                        sy_r[0] = start_py

                    pts_r = np.array([sx_r, sy_r]).T.reshape(-1, 1, 2)
                    segs_r = np.concatenate([pts_r[:-1], pts_r[1:]], axis=1)

                    lc_bg_ref.set_segments(segs_r)
                    lc_ref.set_segments(segs_r)
                    lc_ref.set_array((ss_r[:-1] + ss_r[1:]) / 2.0)

                    # Dynamiczne przeliczanie drogi przebytej (start → react) z aktualną masą
                    flown_path_cmp = list(path_global_ref[:react_idx + 1])
                    flown_dist_cmp = calculate_path_length(flown_path_cmp)
                    flown_time_cmp = calculate_kinematic_flight_time(flown_path_cmp, mass=m)
                    flown_risk_cmp = calculate_segment_risk(flown_path_cmp, env)
                    flown_turns_cmp = 0
                    if len(flown_path_cmp) > 2:
                        _ld = (flown_path_cmp[1][0] - flown_path_cmp[0][0],
                               flown_path_cmp[1][1] - flown_path_cmp[0][1])
                        for _i in range(2, len(flown_path_cmp)):
                            _cd = (flown_path_cmp[_i][0] - flown_path_cmp[_i-1][0],
                                   flown_path_cmp[_i][1] - flown_path_cmp[_i-1][1])
                            if _cd != _ld:
                                flown_turns_cmp += 1
                                _ld = _cd

                    t_dist = flown_dist_cmp + calculate_path_length(path_replan)
                    t_time = flown_time_cmp + calculate_kinematic_flight_time(path_replan, mass=m)
                    t_risk = flown_risk_cmp + calculate_segment_risk(path_replan, env)
                    t_turns = flown_turns_cmp + stats_replan.get('turns', 0)

                    # Przelicz baseline z aktualną masą
                    orig_path = sim_st.get("path_global_original", path_global_ref)
                    b_dist = calculate_path_length(orig_path)
                    b_time = calculate_kinematic_flight_time(orig_path, mass=m)
                    b_risk = calculate_segment_risk(orig_path, env)
                    b_turns = 0
                    if len(orig_path) > 2:
                        _ld2 = (orig_path[1][0] - orig_path[0][0], orig_path[1][1] - orig_path[0][1])
                        for _i2 in range(2, len(orig_path)):
                            _cd2 = (orig_path[_i2][0] - orig_path[_i2-1][0], orig_path[_i2][1] - orig_path[_i2-1][1])
                            if _cd2 != _ld2:
                                b_turns += 1
                                _ld2 = _cd2

                    d_dist = t_dist - b_dist
                    d_time = t_time - b_time
                    d_risk = t_risk - b_risk
                    d_turns = t_turns - b_turns

                    fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
                    fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

                    dr = drone_radius_for_mass(m)
                    ax_ref.set_title(
                        f"{name_ref} — Omijanie (W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a_val:.1f} m/s²)\n"
                        f"Droga: {t_dist:.1f} m ({fmt(d_dist)} m nadłożono)\n"
                        f"Czas: {t_time:.1f}s ({fmt(d_time)}) | Ryzyko: {t_risk:.1f} ({fmt(d_risk)}) | "
                        f"Zakręty: {t_turns} ({fmt_i(d_turns)})",
                        fontsize=14, color=clr_ref, pad=25)
                else:
                    if w_slider.val != 40.0:
                        w_slider.set_val(40.0)
                        return
                    path_rth, stats_rth = func_ref(env, cur_drone_pos, start, risk_weight=40.0,
                                                   turn_penalty=TURN_PENALTY, drone_radius=col_r,
                                                   initial_direction=f_heading, current_speed=v_react_end, drone_mass=m)
                    if path_rth and stats_rth['found']:
                        artificial_lead_in = []
                        if f_heading != (0, 0):
                            p1 = (cur_drone_pos[0] - f_heading[0] * 2.0, cur_drone_pos[1] - f_heading[1] * 2.0)
                            p2 = (cur_drone_pos[0] - f_heading[0] * 1.0, cur_drone_pos[1] - f_heading[1] * 1.0)
                            artificial_lead_in = [p1, p2]

                        speeds_rth_raw = compute_path_speeds(path_rth, initial_speed=v_react_end, accel=a_val)

                        if artificial_lead_in:
                            combined_path = artificial_lead_in + path_rth
                            lead_speeds = np.full(len(artificial_lead_in), v_react_end)
                            combined_speeds = np.concatenate([lead_speeds, speeds_rth_raw])

                            sx_rth, sy_rth, ss_rth = smooth_path_with_speeds(combined_path, combined_speeds,
                                                                             accel=a_val)

                            start_px, start_py = cur_drone_pos
                            search_limit = min(len(sx_rth), 60)
                            dists_rth = (sx_rth[:search_limit] - start_px) ** 2 + (
                                        sy_rth[:search_limit] - start_py) ** 2
                            best_idx = np.argmin(dists_rth)

                            # CZYSTE CIĘCIE
                            sx_rth = sx_rth[best_idx:]
                            sy_rth = sy_rth[best_idx:]
                            ss_rth = ss_rth[best_idx:]

                            # --- DODANE: IDEALNE ZSZYCIE ---
                            sx_rth[0] = start_px
                            sy_rth[0] = start_py
                        else:
                            sx_rth, sy_rth, ss_rth = smooth_path_with_speeds(path_rth, speeds_rth_raw, accel=a_val)
                            # --- DODANE: IDEALNE ZSZYCIE ---
                            start_px, start_py = cur_drone_pos
                            sx_rth[0] = start_px
                            sy_rth[0] = start_py

                        pts_rth = np.array([sx_rth, sy_rth]).T.reshape(-1, 1, 2)
                        segs_rth = np.concatenate([pts_rth[:-1], pts_rth[1:]], axis=1)
                        lc_bg_ref.set_segments(segs_rth)
                        lc_ref.set_segments(segs_rth)
                        lc_ref.set_array((ss_rth[:-1] + ss_rth[1:]) / 2.0)

                        # Dynamiczne przeliczenie statystyk RTH
                        flown_path_rth = list(path_global_ref[:react_idx + 1])
                        flown_dist_rth = calculate_path_length(flown_path_rth)
                        flown_time_rth = calculate_kinematic_flight_time(flown_path_rth, mass=m)
                        flown_risk_rth = calculate_segment_risk(flown_path_rth, env)
                        flown_turns_rth = 0
                        if len(flown_path_rth) > 2:
                            _ld = (flown_path_rth[1][0] - flown_path_rth[0][0],
                                   flown_path_rth[1][1] - flown_path_rth[0][1])
                            for _i in range(2, len(flown_path_rth)):
                                _cd = (flown_path_rth[_i][0] - flown_path_rth[_i-1][0],
                                       flown_path_rth[_i][1] - flown_path_rth[_i-1][1])
                                if _cd != _ld:
                                    flown_turns_rth += 1
                                    _ld = _cd

                        t_dist = flown_dist_rth + calculate_path_length(path_rth)
                        t_time = flown_time_rth + calculate_kinematic_flight_time(path_rth, mass=m)
                        t_risk = flown_risk_rth + calculate_segment_risk(path_rth, env)
                        t_turns = flown_turns_rth + stats_rth.get('turns', 0)

                        orig_path = sim_st.get("path_global_original", path_global_ref)
                        b_dist = calculate_path_length(orig_path)
                        b_time = calculate_kinematic_flight_time(orig_path, mass=m)
                        b_risk = calculate_segment_risk(orig_path, env)
                        b_turns = 0
                        if len(orig_path) > 2:
                            _ld2 = (orig_path[1][0] - orig_path[0][0], orig_path[1][1] - orig_path[0][1])
                            for _i2 in range(2, len(orig_path)):
                                _cd2 = (orig_path[_i2][0] - orig_path[_i2-1][0], orig_path[_i2][1] - orig_path[_i2-1][1])
                                if _cd2 != _ld2:
                                    b_turns += 1
                                    _ld2 = _cd2

                        fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
                        fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

                        dr = drone_radius_for_mass(m)
                        ax_ref.set_title(
                            f"{name_ref} — Tryb Powrotu (W=40, m={m:.0f} kg, r={dr:.2f} m, a={a_val:.1f} m/s²)\n"
                            f"Droga: {t_dist:.1f} m ({fmt(t_dist - b_dist)} m nadłożono)\n"
                            f"Czas: {t_time:.1f}s ({fmt(t_time - b_time)}) | Ryzyko: {t_risk:.1f} ({fmt(t_risk - b_risk)}) | "
                            f"Zakręty: {t_turns} ({fmt_i(t_turns - b_turns)})",
                            fontsize=14, color='orange', pad=25)
                    else:
                        lc_bg_ref.set_segments([])
                        lc_ref.set_segments([])
                        ax_ref.set_title(f"{name_ref} — BRAK TRASY (W={w:.0f})", fontsize=14, color='red', pad=25)

                ax_ref.figure.canvas.draw_idle()
            return update_cmp

        update_fn = make_update(ax_cmp, lc_n_bg, lc_n, lc_f_bg, lc_f,
                                path_global, global_speeds, algo_func, algo_name, color,
                                slider_cmp, mass_slider_cmp, sim_state,
                                line_reaction_cmp, detect_dot_cmp)
        slider_cmp.on_changed(update_fn)
        mass_slider_cmp.on_changed(update_fn)

        update_fn(RISK_WEIGHT)

        fig_cmp.canvas.manager.set_window_title(algo_name)
        fig_cmp._slider_ref = slider_cmp
        fig_cmp._mass_slider_ref = mass_slider_cmp
        plt.show(block=False)
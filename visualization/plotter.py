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
    collision_radius_for_mass, drone_radius_for_mass
)
from config import (
    V_MAX_MS, ACCELERATION, MAX_LATERAL_ACCEL, MIN_TURN_SPEED,
    RISK_WEIGHT, TURN_PENALTY,
    COLLISION_RADIUS, SENSOR_RANGE,
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
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W) ', valmin=0.0, valmax=40.0,
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
        "global_speeds": global_speeds
    }

    def update_route(val):
        # ── TRYB PRZED KLIKNIĘCIEM: przelicz trasę globalną ──────────
        if sim_state["mode"] == "IDLE" and not sim_state["clicked"]:
            w = risk_slider.val
            m = mass_slider.val
            a_cur = MAX_THRUST_NET_N / m
            col_r = collision_radius_for_mass(m)
            env._recompute_collision_mask(col_r)

            new_path, new_stats = search_func(env, start, goal, risk_weight=w,
                                              turn_penalty=TURN_PENALTY,
                                              drone_radius=col_r,
                                              drone_mass=m)
            if new_path:
                sim_state["path_global"] = new_path
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

        # ── TRYB PO KLIKNIĘCIU: replanowanie ─────────────────────────
        if sim_state["mode"] in ["CRASH", "IGNORE"]:
            return

        w = risk_slider.val
        m = mass_slider.val
        a_current = MAX_THRUST_NET_N / m

        # Promień kolizji zależny od masy
        col_radius = collision_radius_for_mass(m)

        # Przeliczyć maskę kolizji dla nowego promienia drona
        env._recompute_collision_mask(col_radius)

        # Przelicz bufor pędu na podstawie aktualnej masy
        drone_pos = sim_state["drone_pos"]
        heading = sim_state.get("heading", (0, 0))
        v_react = sim_state.get("visual_speed", 0.0)

        buffer_points = []
        buffer_dist = 0.0
        if heading != (0, 0) and v_react > 0:
            r_45 = 1.5 / max(0.1, math.sin(math.radians(22.5)))
            v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_45))
            v_safe_ref = min(v_safe_ref, V_MAX_MS)

            braking_dist_needed = max(0.0, (v_react**2 - v_safe_ref**2) / (2 * a_current)) \
                if v_react > v_safe_ref else 0.0

            step_len = math.sqrt(heading[0]**2 + heading[1]**2)
            buffer_steps = max(1, int(math.ceil(braking_dist_needed / step_len)))

            for d in range(1, buffer_steps + 1):
                bp = (drone_pos[0] + heading[0] * d,
                      drone_pos[1] + heading[1] * d)
                bx, by = int(bp[0]), int(bp[1])
                if 0 <= bx < env.width and 0 <= by < env.height:
                    if not env.is_collision(bx, by, drone_radius=col_radius):
                        buffer_points.append(bp)
                        buffer_dist += step_len
                    else:
                        break
                else:
                    break

        v_at_buffer_end = max(0.0, math.sqrt(max(0.0, v_react**2 - 2 * a_current * buffer_dist)))

        search_start = buffer_points[-1] if buffer_points else drone_pos

        # Zawsze próbuj cel najpierw, potem RTH
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
            # Fallback: powrót do startu
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

            speeds = compute_path_speeds(full_new_path, initial_speed=v_react, accel=a_current)

            lead_in = sim_state.get("lead_in_points", [])
            if lead_in and len(full_new_path) >= 3:
                n_lead = len(lead_in)
                lead_speeds = np.full(n_lead, v_react)
                combined_path = lead_in + full_new_path
                combined_speeds = np.concatenate([lead_speeds, speeds])
                sx, sy, s_speeds = smooth_path_with_speeds(combined_path, combined_speeds, accel=a_current)
                trim = n_lead * 10
                sx, sy, s_speeds = sx[trim:], sy[trim:], s_speeds[trim:]
            else:
                sx, sy, s_speeds = smooth_path_with_speeds(full_new_path, speeds, accel=a_current)

            points = np.array([sx, sy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc_new_bg.set_segments(segments)
            lc_new.set_segments(segments)
            lc_new.set_array((s_speeds[:-1] + s_speeds[1:]) / 2.0)

            # Droga przebyta — prędkości z PEŁNEJ trasy, potem slice do detect_idx
            # Lead-out: dodaj kilka punktów ZA detect_idx żeby spline płynnie kontynuował
            detect_idx = sim_state.get("drone_detect_idx", 0)
            p_global = sim_state["path_global"]
            if detect_idx > 0 and len(p_global) > detect_idx:
                full_spd = compute_path_speeds(p_global, accel=a_current)
                LEAD_OUT = 2
                end_idx = min(detect_idx + 1 + LEAD_OUT, len(p_global))
                extended_raw = p_global[:end_idx]
                extended_spd = full_spd[:end_idx]
                sx_fl, sy_fl, ss_fl = smooth_path_with_speeds(extended_raw, extended_spd, accel=a_current)
                # Odetnij lead-out z wyniku (proporcjonalnie do interpolacji ×10)
                trim_end = (detect_idx + 1) * 10
                if trim_end < len(sx_fl):
                    sx_fl, sy_fl, ss_fl = sx_fl[:trim_end], sy_fl[:trim_end], ss_fl[:trim_end]
                pts_fl = np.array([sx_fl, sy_fl]).T.reshape(-1, 1, 2)
                segs_fl = np.concatenate([pts_fl[:-1], pts_fl[1:]], axis=1)
                lc_flown_bg.set_segments(segs_fl)
                lc_flown.set_segments(segs_fl)
                lc_flown.set_array((ss_fl[:-1] + ss_fl[1:]) / 2.0)

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

            t_time = sim_state["flown_time"] + calculate_kinematic_flight_time(full_new_path, mass=m)
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
                dr = drone_radius_for_mass(m)
                ax.set_title(f"Risk-Aware A* — Tryb Powrotu (W=40, m={m:.0f} kg, r={dr:.2f} m, a={a_current:.1f} m/s²)\n{stats_text}",
                             color='orange', fontsize=14, pad=25)
                lc_new.set_cmap(get_speed_cmap())
                line_proxy.set_color('orange')
            else:
                dr = drone_radius_for_mass(m)
                ax.set_title(f"Risk-Aware A* — Omijanie (W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a_current:.1f} m/s²)\n{stats_text}",
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

        # Użyj aktualnej trasy (mogła być przeliczona suwakiem masy)
        path_global = sim_state["path_global"]
        global_speeds = sim_state["global_speeds"]

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

        # Bufor pędu (hamowanie) — uwzględnia aktualną masę z suwaka
        onclick_mass = mass_slider.val
        onclick_accel = MAX_THRUST_NET_N / onclick_mass
        onclick_col_r = collision_radius_for_mass(onclick_mass)

        buffer_points = []
        buffer_dist = 0.0
        if flight_heading != (0, 0):
            r_45 = 1.5 / max(0.1, math.sin(math.radians(22.5)))
            v_safe_ref = max(MIN_TURN_SPEED, math.sqrt(MAX_LATERAL_ACCEL * r_45))
            v_safe_ref = min(v_safe_ref, V_MAX_MS)

            braking_dist_needed = max(0.0, (v_react_end**2 - v_safe_ref**2) / (2 * onclick_accel)) \
                if v_react_end > v_safe_ref else 0.0

            step_len = math.sqrt(flight_heading[0]**2 + flight_heading[1]**2)
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

        v_at_buffer_end = max(0.0, math.sqrt(max(0.0, v_react_end**2 - 2 * onclick_accel * buffer_dist)))

        sim_state["drone_pos"] = current_drone_pos
        sim_state["buffer_points"] = buffer_points
        sim_state["heading"] = flight_heading
        sim_state["drone_speed"] = v_at_buffer_end
        sim_state["visual_speed"] = v_react_end
        sim_state["buffer_dist"] = buffer_dist
        sim_state["drone_detect_idx"] = drone_detect_idx
        sim_state["click_mass"] = mass_slider.val

        LEAD_IN_COUNT = 5
        lead_start = max(0, drone_react_idx - LEAD_IN_COUNT)
        sim_state["lead_in_points"] = list(path_global[lead_start:drone_react_idx])

        flown_full = path_global[:drone_react_idx + 1]
        sim_state["flown_dist"] = calculate_path_length(flown_full)
        sim_state["flown_time"] = calculate_kinematic_flight_time(flown_full)
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

        # Rysowanie drogi przebytej na oknie głównym
        flown_raw = path_global[:drone_detect_idx + 1]
        f_speeds = global_speeds[:drone_detect_idx + 1]
        sx_f, sy_f, ss_f = smooth_path_with_speeds(flown_raw, f_speeds)
        pts_f = np.array([sx_f, sy_f]).T.reshape(-1, 1, 2)
        segs_f = np.concatenate([pts_f[:-1], pts_f[1:]], axis=1)
        lc_flown_bg.set_segments(segs_f)
        lc_flown.set_segments(segs_f)
        lc_flown.set_array((ss_f[:-1] + ss_f[1:]) / 2.0)

        # Sprawdzenie katastrofy (na ścieżce reakcji)
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

        # ── Risk-Aware A* — replanowanie na oknie głównym ────────────────
        search_start = sim_state["buffer_points"][-1] if sim_state.get("buffer_points") else sim_state["drone_pos"]
        click_m = mass_slider.val
        click_col_r = collision_radius_for_mass(click_m)
        env._recompute_collision_mask(click_col_r)
        path_check, _ = search_func(env, search_start, goal, risk_weight=RISK_WEIGHT,
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

        # ══════════════════════════════════════════════════════════════════
        # OTWARCIE 2 DODATKOWYCH OKIEN: Dijkstra i A* Standard
        # Te algorytmy NIE mają kinematyki — nie wiedzą o prędkości,
        # nie hamują przed zakrętem. Efekt: ostry zakręt → rozpad drona.
        # ══════════════════════════════════════════════════════════════════
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
        sim_state=None
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
        plt.subplots_adjust(bottom=0.18, right=0.80, left=0.10, top=0.88)
        setup_dark_theme(fig_cmp, ax_cmp)

        img_cmp = ax_cmp.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
        ax_cmp.set_xlim(0, env.width)
        ax_cmp.set_ylim(0, env.height)
        _setup_ui_colorbars(fig_cmp, ax_cmp, img_cmp,
                            speed_axes_rect=[0.115, 0.13, 0.62, 0.02],
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

        # Czas reakcji (pomarańczowa)
        ax_cmp.plot([p[0] for p in reaction_path], [p[1] for p in reaction_path],
                    color='orange', linestyle=':', linewidth=4, label='Czas Reakcji', zorder=4)

        # Punkt wykrycia
        ax_cmp.plot([path_global[drone_detect_idx][0]], [path_global[drone_detect_idx][1]],
                    'o', color='yellow', markersize=12, markeredgecolor='black', zorder=6,
                    label='Punkt Wykrycia')

        # Start i cel
        ax_cmp.scatter([start[0]], [start[1]], color='lime', s=150, label='Start',
                       edgecolors='black', zorder=5)
        ax_cmp.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel',
                       edgecolors='black', zorder=5)

        # Replanowana trasa (dynamiczna)
        lc_n_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=4)
        lc_n = LineCollection([], cmap=get_speed_cmap(), linewidths=5,
                              norm=plt.Normalize(0, V_MAX_MS), zorder=5)
        ax_cmp.add_collection(lc_n_bg)
        ax_cmp.add_collection(lc_n)

        ax_cmp.plot([], [], color='cyan', linewidth=5, label='Replanowana Trasa',
                    path_effects=[pe.withStroke(linewidth=7, foreground="#555555")])

        legend_cmp = ax_cmp.legend(loc='lower left', bbox_to_anchor=(1.04, -0.01),
                                    facecolor='#333333', edgecolor='white',
                                    title=algo_name)
        plt.setp(legend_cmp.get_texts(), color='white')
        plt.setp(legend_cmp.get_title(), color='white')

        # Suwak Wagi Ryzyka
        ax_slider_cmp = plt.axes([0.10, 0.07, 0.65, 0.03], facecolor='#333333')
        slider_cmp = Slider(ax=ax_slider_cmp, label='Waga Ryzyka (W) ',
                            valmin=0.0, valmax=40.0, valinit=RISK_WEIGHT, valstep=1.0,
                            color=color)
        slider_cmp.label.set_color('white')
        slider_cmp.valtext.set_color('white')

        # Suwak Masy Drona
        ax_mass_cmp = plt.axes([0.10, 0.02, 0.65, 0.03], facecolor='#333333')
        mass_slider_cmp = Slider(ax=ax_mass_cmp, label='Masa Drona [kg] ',
                                 valmin=1.0, valmax=50.0, valinit=DRONE_MASS_KG, valstep=1.0,
                                 color='orange')
        mass_slider_cmp.label.set_color('white')
        mass_slider_cmp.valtext.set_color('white')

        def make_update(ax_ref, lc_bg_ref, lc_ref, lc_f_bg_ref, lc_f_ref,
                        flown_raw_ref, path_global_ref, func_ref, name_ref, clr_ref,
                        w_slider, m_slider, sim_st):
            def update_cmp(val):
                w = w_slider.val
                m = m_slider.val
                a_val = MAX_THRUST_NET_N / m

                # Przerysuj drogę przebytą — lead-out dla płynnego przejścia
                if len(flown_raw_ref) >= 2 and len(path_global_ref) > len(flown_raw_ref):
                    full_spd = compute_path_speeds(path_global_ref, accel=a_val)
                    LEAD_OUT = 2
                    end_idx = min(len(flown_raw_ref) + LEAD_OUT, len(path_global_ref))
                    extended_raw = list(path_global_ref[:end_idx])
                    extended_spd = full_spd[:end_idx]
                    sx_fl, sy_fl, ss_fl = smooth_path_with_speeds(extended_raw, extended_spd, accel=a_val)
                    trim_end = len(flown_raw_ref) * 10
                    if trim_end < len(sx_fl):
                        sx_fl, sy_fl, ss_fl = sx_fl[:trim_end], sy_fl[:trim_end], ss_fl[:trim_end]
                    pts_fl = np.array([sx_fl, sy_fl]).T.reshape(-1, 1, 2)
                    segs_fl = np.concatenate([pts_fl[:-1], pts_fl[1:]], axis=1)
                    lc_f_bg_ref.set_segments(segs_fl)
                    lc_f_ref.set_segments(segs_fl)
                    lc_f_ref.set_array((ss_fl[:-1] + ss_fl[1:]) / 2.0)

                col_r = collision_radius_for_mass(m)
                env._recompute_collision_mask(col_r)

                path_replan, stats_replan = func_ref(
                    env, current_drone_pos, goal,
                    risk_weight=w,
                    turn_penalty=TURN_PENALTY,
                    drone_radius=col_r,
                    initial_direction=flight_heading,
                    current_speed=0.0,
                    drone_mass=m
                )

                if path_replan and stats_replan['found']:
                    speeds_r = compute_path_speeds(path_replan, initial_speed=v_react_end, accel=a_val)
                    sx_r, sy_r, ss_r = smooth_path_with_speeds(path_replan, speeds_r, accel=a_val)
                    pts_r = np.array([sx_r, sy_r]).T.reshape(-1, 1, 2)
                    segs_r = np.concatenate([pts_r[:-1], pts_r[1:]], axis=1)

                    lc_bg_ref.set_segments(segs_r)
                    lc_ref.set_segments(segs_r)
                    lc_ref.set_array((ss_r[:-1] + ss_r[1:]) / 2.0)

                    # Totale: przebyta + replan (jak Risk-Aware)
                    flown_dist = sim_st.get("flown_dist", 0.0)
                    flown_time = sim_st.get("flown_time", 0.0)
                    flown_risk = sim_st.get("flown_risk", 0.0)
                    flown_turns = sim_st.get("flown_turns", 0)

                    replan_dist = calculate_path_length(path_replan)
                    replan_time = calculate_kinematic_flight_time(path_replan, mass=m)
                    replan_risk = calculate_segment_risk(path_replan, env)

                    t_dist = flown_dist + replan_dist
                    t_time = flown_time + replan_time
                    t_risk = flown_risk + replan_risk
                    t_turns = flown_turns + stats_replan.get('turns', 0)

                    base_dist = sim_st.get("base_dist", 0.0)
                    base_time = sim_st.get("base_time", 0.0)
                    base_risk = sim_st.get("base_risk", 0.0)
                    base_turns = sim_st.get("base_turns", 0)

                    d_dist = t_dist - base_dist
                    d_time = t_time - base_time
                    d_risk = t_risk - base_risk
                    d_turns = t_turns - base_turns

                    fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
                    fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

                    dr = drone_radius_for_mass(m)
                    title = (
                        f"{name_ref} — Omijanie (W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a_val:.1f} m/s²)\n"
                        f"Droga: {t_dist:.1f} m ({fmt(d_dist)} m nadłożono)\n"
                        f"Czas: {t_time:.1f}s ({fmt(d_time)}) | Ryzyko: {t_risk:.1f} ({fmt(d_risk)}) | "
                        f"Zakręty: {t_turns} ({fmt_i(d_turns)})"
                    )
                    ax_ref.set_title(title, fontsize=13, color=clr_ref, pad=20)
                else:
                    # Próba powrotu do startu (RTH) — W=40 jak Risk-Aware
                    if w_slider.val != 40.0:
                        w_slider.set_val(40.0)
                        return  # set_val wywoła update_cmp ponownie
                    path_rth, stats_rth = func_ref(
                        env, current_drone_pos, start,
                        risk_weight=40.0,
                        turn_penalty=TURN_PENALTY,
                        drone_radius=col_r,
                        initial_direction=flight_heading,
                        current_speed=0.0,
                        drone_mass=m
                    )
                    if path_rth and stats_rth['found']:
                        speeds_rth = compute_path_speeds(path_rth, initial_speed=v_react_end, accel=a_val)
                        sx_rth, sy_rth, ss_rth = smooth_path_with_speeds(path_rth, speeds_rth, accel=a_val)
                        pts_rth = np.array([sx_rth, sy_rth]).T.reshape(-1, 1, 2)
                        segs_rth = np.concatenate([pts_rth[:-1], pts_rth[1:]], axis=1)
                        lc_bg_ref.set_segments(segs_rth)
                        lc_ref.set_segments(segs_rth)
                        lc_ref.set_array((ss_rth[:-1] + ss_rth[1:]) / 2.0)

                        flown_dist = sim_st.get("flown_dist", 0.0)
                        flown_time = sim_st.get("flown_time", 0.0)
                        flown_risk = sim_st.get("flown_risk", 0.0)
                        flown_turns = sim_st.get("flown_turns", 0)

                        rth_dist = calculate_path_length(path_rth)
                        rth_time = calculate_kinematic_flight_time(path_rth, mass=m)
                        rth_risk = calculate_segment_risk(path_rth, env)

                        t_dist = flown_dist + rth_dist
                        t_time = flown_time + rth_time
                        t_risk = flown_risk + rth_risk
                        t_turns = flown_turns + stats_rth.get('turns', 0)

                        base_dist = sim_st.get("base_dist", 0.0)
                        base_time = sim_st.get("base_time", 0.0)
                        base_risk = sim_st.get("base_risk", 0.0)
                        base_turns = sim_st.get("base_turns", 0)

                        fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
                        fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

                        dr = drone_radius_for_mass(m)
                        ax_ref.set_title(
                            f"{name_ref} — Tryb Powrotu (W=40, m={m:.0f} kg, r={dr:.2f} m, a={a_val:.1f} m/s²)\n"
                            f"Droga: {t_dist:.1f} m ({fmt(t_dist - base_dist)} m nadłożono)\n"
                            f"Czas: {t_time:.1f}s ({fmt(t_time - base_time)}) | Ryzyko: {t_risk:.1f} ({fmt(t_risk - base_risk)}) | "
                            f"Zakręty: {t_turns} ({fmt_i(t_turns - base_turns)})",
                            fontsize=13, color='orange', pad=20)
                    else:
                        lc_bg_ref.set_segments([])
                        lc_ref.set_segments([])
                        ax_ref.set_title(f"{name_ref} — BRAK TRASY (W={w:.0f})",
                                         fontsize=13, color='red', pad=20)

                ax_ref.figure.canvas.draw_idle()
            return update_cmp

        update_fn = make_update(ax_cmp, lc_n_bg, lc_n, lc_f_bg, lc_f,
                                flown_raw, path_global, algo_func, algo_name, color,
                                slider_cmp, mass_slider_cmp, sim_state)
        slider_cmp.on_changed(update_fn)
        mass_slider_cmp.on_changed(update_fn)

        # Początkowe rysowanie
        update_fn(RISK_WEIGHT)

        fig_cmp.canvas.manager.set_window_title(algo_name)
        fig_cmp._slider_ref = slider_cmp
        fig_cmp._mass_slider_ref = mass_slider_cmp
        plt.show(block=False)
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
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
    V_MAX_MS, ACCELERATION,
    RISK_WEIGHT, TURN_PENALTY,
    COLLISION_RADIUS, OBSTACLE_RADIUS,
    DRONE_MASS_KG, MAX_THRUST_NET_N
)
from visualization.metrics_terminal import (
    analyze_braking_scenario, print_braking_comparison
)


def _build_legend_handles():
    """Jednolita legenda — identyczna kolejność dla wszystkich trzech okien."""
    return [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5,
               label='Pierwotny Plan'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=4,
               label='Czas Reakcji'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
               markersize=10, linestyle='None', markeredgecolor='black',
               label='Punkt Wykrycia'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
               markersize=10, linestyle='None', markeredgecolor='black',
               label='Start'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='magenta',
               markersize=10, linestyle='None', markeredgecolor='black',
               label='Cel'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#000000',
               markersize=10, linestyle='None', markeredgecolor='black',
               label='Twarda Strefa Ryzyka'),
        Line2D([0], [0], color='dodgerblue', linewidth=5, label='Trasa'),
    ]


def _setup_ui_colorbars(fig, ax, img, speed_axes_rect: list,
                        risk_fraction: float = 0.046,
                        risk_pad: float = 0.05,
                        risk_shrink: float = 0.80) -> None:
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=risk_fraction, pad=risk_pad, shrink=risk_shrink,
                        anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    cax_speed = fig.add_axes(speed_axes_rect)
    sm = plt.cm.ScalarMappable(cmap=get_speed_cmap(), norm=plt.Normalize(0, V_MAX_MS))
    sm.set_array([])
    cbar_speed = fig.colorbar(sm, cax=cax_speed, orientation='horizontal')
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
            dist = math.hypot(x_smooth[i] - x_smooth[i - 1], y_smooth[i] - y_smooth[i - 1])
            speeds_smooth[i] = min(speeds_smooth[i],
                                   math.sqrt(max(0.0, speeds_smooth[i - 1] ** 2 + 2 * a * dist)))
        for i in range(len(speeds_smooth) - 2, -1, -1):
            dist = math.hypot(x_smooth[i + 1] - x_smooth[i], y_smooth[i + 1] - y_smooth[i])
            speeds_smooth[i] = min(speeds_smooth[i],
                                   math.sqrt(max(0.0, speeds_smooth[i + 1] ** 2 + 2 * a * dist)))

        initial_v = speeds[0]
        speeds_smooth[0] = initial_v
        for i in range(1, len(speeds_smooth)):
            dist = math.hypot(x_smooth[i] - x_smooth[i - 1], y_smooth[i] - y_smooth[i - 1])
            min_phys = math.sqrt(max(0.0, speeds_smooth[i - 1] ** 2 - 2 * a * dist))
            speeds_smooth[i] = max(speeds_smooth[i], min_phys)

        return x_smooth, y_smooth, speeds_smooth
    except Exception:
        return np.array(x), np.array(y), speeds


# ─────────────────────────────────────────────────────────────────────────────
# WSPÓLNA FUNKCJA OBLICZENIOWA
# Oblicza pełny scenariusz dla dowolnego algorytmu: trasa czysta → detekcja →
# reakcja → replan. Zwraca dict z danymi do wizualizacji i statystykami.
# Używana przez update_route (Risk-Aware) i update_cmp (Dijkstra/A*).
# ─────────────────────────────────────────────────────────────────────────────
def _compute_full_scenario(env, start, goal, algo_func, w, m, click_pos, grid_before):
    """
    Oblicza pełny scenariusz omijania przeszkody.

    [METODOLOGIA] Wszystkie trzy systemy nawigacyjne otrzymują identyczne
    wartości wejściowe (mapa, masa, prędkość, kierunek, parametry fizyczne)
    i są wywoływane przez identyczny kod symulacyjny. Różnice w wynikach
    wynikają WYŁĄCZNIE z wewnętrznej logiki algorytmu:
      - Risk-Aware A*: planuje bufor hamowania awaryjnego wewnątrz siebie
        (zob. _plan_braking_buffer w a_star_risk.py), bo posiada model
        kinematyczny pozwalający obliczyć wymaganą drogę hamowania.
      - Dijkstra / A*: nie planują bufora, bo pojęcie "bufora hamowania"
        jest bez modelu kinematycznego nieokreślone.
    Symulator (ta funkcja) nie zna tych różnic i nie podejmuje za algorytm
    żadnych decyzji kinematycznych.
    """
    a = MAX_THRUST_NET_N / m
    col_r = collision_radius_for_mass(m)
    phys_r = drone_radius_for_mass(m)
    cx, cy = click_pos

    result = {}

    # ── KROK 1: Planuj na czystej mapie ──────────────────────────────────
    grid_dirty = env.grid.copy()
    risk_dirty = env.risk_grid.copy()
    dist_dirty = env.dist_matrix.copy()
    mask_dirty = env.collision_mask.copy()
    dyn_backup = list(env.dynamic_obstacles)

    env.grid = grid_before.copy()
    env.dynamic_obstacles = []
    env._recompute_dist_matrix()
    env.update_drone_footprint(phys_r, col_r)

    clean_path, clean_stats = algo_func(env, start, goal, risk_weight=w,
                                        turn_penalty=TURN_PENALTY,
                                        drone_radius=col_r, drone_mass=m)

    # Przywróć brudną mapę
    env.grid = grid_dirty
    env.risk_grid = risk_dirty
    env.dist_matrix = dist_dirty
    env.collision_mask = mask_dirty
    env.dynamic_obstacles = dyn_backup

    if not clean_path or not clean_stats.get('found'):
        result['mode'] = 'NO_PATH'
        return result

    clean_speeds = compute_path_speeds(clean_path, accel=a)
    result['clean_path'] = clean_path
    result['clean_speeds'] = clean_speeds
    result['clean_stats'] = clean_stats
    result['baseline'] = (clean_stats['length'],
                          clean_stats.get('flight_time', 0),
                          clean_stats['risk'],
                          clean_stats.get('turns', 0))

    # ── KROK 2: Sprawdź czy przeszkoda blokuje trasę ─────────────────────
    CRASH_DIST = OBSTACLE_RADIUS + col_r
    is_blocked = any(
        math.sqrt((px - cx) ** 2 + (py - cy) ** 2) <= CRASH_DIST
        for px, py in clean_path
    )

    if not is_blocked:
        result['mode'] = 'CLEAR'
        result['blocked'] = False
        return result

    result['blocked'] = True

    # ── KROK 3: Detekcja i reakcja ───────────────────────────────────────
    sensor_range = sensor_range_for_mass(m)
    proc_delay = processing_delay_for_mass(m)

    detect_idx = -1
    for i, (px, py) in enumerate(clean_path):
        if math.sqrt((px - cx) ** 2 + (py - cy) ** 2) <= sensor_range:
            detect_idx = i
            break

    if detect_idx == -1:
        result['mode'] = 'NO_SENSOR'
        return result

    v_detect = float(clean_speeds[detect_idx])
    t_stop = v_detect / a
    t_react = min(proc_delay, t_stop)
    react_dist = (v_detect * t_react) - (0.5 * a * (t_react ** 2))

    # [POPRAWKA 1] Przywrócone wyliczenie prędkości na końcu fazy reakcji
    v_react_end = max(0.0, v_detect - a * proc_delay)

    # Iteruj po rzeczywistym dystansie ścieżki
    accumulated_dist = 0.0
    react_idx = detect_idx
    for i in range(detect_idx, len(clean_path) - 1):
        dx = clean_path[i + 1][0] - clean_path[i][0]
        dy = clean_path[i + 1][1] - clean_path[i][1]
        step_len = math.hypot(dx, dy)
        accumulated_dist += step_len
        react_idx = i + 1
        if accumulated_dist >= react_dist:
            break

    # [POPRAWKA 2] Usunięta błędna linijka wykorzystująca `react_indices`

    reaction_path = clean_path[detect_idx:react_idx + 1]
    drone_pos = clean_path[react_idx]

    result['detect_idx'] = detect_idx
    result['react_idx'] = react_idx
    result['v_detect'] = v_detect
    result['v_react_end'] = v_react_end
    result['reaction_path'] = reaction_path
    result['drone_pos'] = drone_pos

    # Heading z reakcji
    if len(reaction_path) >= 2:
        ldx = reaction_path[-1][0] - reaction_path[-2][0]
        ldy = reaction_path[-1][1] - reaction_path[-2][1]
        heading = (int(np.sign(ldx)), int(np.sign(ldy)))
    else:
        heading = (0, 0)
    result['heading'] = heading

    # Crash check
    crash = any(
        math.sqrt((cx - px) ** 2 + (cy - py) ** 2) <= (OBSTACLE_RADIUS + phys_r)
        for px, py in reaction_path
    )
    if crash:
        result['mode'] = 'CRASH'
        result['crash'] = True
        return result
    result['crash'] = False

    # ── KROK 4: Replanowanie (algorytm sam decyduje o buforze) ───────────
    env.update_drone_footprint(phys_r, col_r)

    replan_path, replan_stats = algo_func(env, drone_pos, goal, risk_weight=w,
                                          turn_penalty=TURN_PENALTY, drone_radius=col_r,
                                          initial_direction=heading,
                                          current_speed=v_react_end,
                                          drone_mass=m)

    if not replan_path or not replan_stats.get('found'):
        # RTH: powrót do startu
        replan_path, replan_stats = algo_func(env, drone_pos, start, risk_weight=40.0,
                                              turn_penalty=TURN_PENALTY, drone_radius=col_r,
                                              initial_direction=heading,
                                              current_speed=v_react_end,
                                              drone_mass=m)
        if replan_path and replan_stats.get('found'):
            result['mode'] = 'RTH'
        else:
            result['mode'] = 'TRAPPED'
            return result
    else:
        result['mode'] = 'NORMAL'

    # Wyciągnij informacje o buforze ze stats algorytmu
    buffer_points = replan_stats.get('buffer_points', [])
    buffer_dist = replan_stats.get('buffer_dist', 0.0)
    v_buf_end = replan_stats.get('v_after_buffer', v_react_end)

    result['buffer_points'] = buffer_points
    result['buffer_dist'] = buffer_dist
    result['v_buffer_end'] = v_buf_end

    # Pełna trasa od pozycji drona
    full_new = replan_path

    result['replan_path'] = replan_path
    result['replan_stats'] = replan_stats
    result['full_new_path'] = full_new

    # ── KROK 6: Statystyki ───────────────────────────────────────────────
    flown_path = clean_path[:react_idx + 1]
    flown_dist = calculate_path_length(flown_path)
    flown_time = calculate_kinematic_flight_time(flown_path, mass=m)
    flown_risk = calculate_segment_risk(flown_path, env)
    flown_turns = 0
    if len(flown_path) > 2:
        ld = (flown_path[1][0] - flown_path[0][0], flown_path[1][1] - flown_path[0][1])
        for ii in range(2, len(flown_path)):
            cd = (flown_path[ii][0] - flown_path[ii - 1][0], flown_path[ii][1] - flown_path[ii - 1][1])
            if cd != ld:
                flown_turns += 1
                ld = cd

    new_dist = calculate_path_length(full_new)
    new_time = calculate_kinematic_flight_time(full_new, mass=m)
    new_risk = calculate_segment_risk(full_new, env)
    new_turns = 0
    if len(full_new) > 2:
        ld = (full_new[1][0] - full_new[0][0], full_new[1][1] - full_new[0][1])
        for ii in range(2, len(full_new)):
            cd = (full_new[ii][0] - full_new[ii - 1][0], full_new[ii][1] - full_new[ii - 1][1])
            if cd != ld:
                new_turns += 1
                ld = cd

    result['total'] = (flown_dist + new_dist,
                       flown_time + new_time,
                       flown_risk + new_risk,
                       flown_turns + new_turns)

    return result


def _draw_scenario(result, ax, lc_flown_bg, lc_flown, line_reaction, drone_marker,
                   lc_new_bg, lc_new, line_global, algo_name, title_color, w, m):
    """Rysuje wynik _compute_full_scenario na osi matplotlib."""
    a = MAX_THRUST_NET_N / m
    dr = drone_radius_for_mass(m)
    fmt = lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
    fmt_i = lambda v: f"+{int(v)}" if v > 0 else f"{int(v)}"

    mode = result.get('mode', 'NO_PATH')

    # Szara linia — trasa na czystej mapie
    if 'clean_path' in result:
        gx, gy = smooth_path_bspline(result['clean_path'])
        line_global.set_data(gx, gy)

    # Wyczyść wszystko
    lc_flown_bg.set_segments([])
    lc_flown.set_segments([])
    line_reaction.set_data([], [])
    drone_marker.set_data([], [])
    lc_new_bg.set_segments([])
    lc_new.set_segments([])

    if mode == 'NO_PATH':
        ax.set_title(f"{algo_name} — BRAK TRASY (W={w:.0f}, m={m:.0f} kg)",
                     color='red', fontsize=14, pad=25)
        return

    if mode == 'CLEAR':
        # Trasa nie blokowana — pokaż pełną trasę z prędkościami
        cp = result['clean_path']
        cs = result['clean_speeds']
        sx, sy, ss = smooth_path_with_speeds(cp, cs, accel=a)
        pts = np.array([sx, sy]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc_flown_bg.set_segments(segs)
        lc_flown.set_segments(segs)
        lc_flown.set_array((ss[:-1] + ss[1:]) / 2.0)

        bd, bt, br, btr = result['baseline']
        ax.set_title(
            f"{algo_name} — Zagrożenie poza kursem (W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a:.1f} m/s²)\n"
            f"Droga: {bd:.1f} m | Czas: {bt:.1f}s | Ryzyko: {br:.1f} | Zakręty: {btr}",
            fontsize=14, color='lime', pad=25)
        return

    if mode == 'NO_SENSOR':
        ax.set_title(f"{algo_name} — KATASTROFA! Zbyt mały zasięg czujnika!",
                     color='red', fontsize=14, pad=25)
        return

    if mode == 'CRASH':
        ax.set_title(f"{algo_name} — KATASTROFA! Zbyt duża bezwładność drona!",
                     color='red', fontsize=14, pad=25)
        return

    if mode == 'TRAPPED':
        ax.set_title(f"{algo_name} — DRON UWIĘZIONY! (W={w:.0f}, m={m:.0f} kg)",
                     color='red', fontsize=14, pad=25)
        return

    # mode == 'NORMAL' or 'RTH' — rysuj pełny scenariusz omijania
    cp = result['clean_path']
    cs = result['clean_speeds']
    det_idx = result['detect_idx']
    react_idx = result['react_idx']

    # Wygładź rozszerzoną trasę (droga + reakcja + lead-out)
    end_ext = min(react_idx + 1 + 2, len(cp))
    ext_path = cp[:end_ext]
    ext_speeds = cs[:end_ext]
    sx_ext, sy_ext, ss_ext = smooth_path_with_speeds(ext_path, ext_speeds, accel=a)

    # Znajdź punkt detekcji i reakcji na wygładzonej krzywej
    det_px, det_py = cp[det_idx]
    idx_det = int(np.argmin((sx_ext - det_px) ** 2 + (sy_ext - det_py) ** 2))
    react_px, react_py = cp[react_idx]
    idx_react = int(np.argmin((sx_ext - react_px) ** 2 + (sy_ext - react_py) ** 2))

    # Droga przebyta (start → detekcja) z kolorami prędkości
    if idx_det > 1:
        sx_g, sy_g, ss_g = sx_ext[:idx_det + 1], sy_ext[:idx_det + 1], ss_ext[:idx_det + 1]
        pts_g = np.array([sx_g, sy_g]).T.reshape(-1, 1, 2)
        segs_g = np.concatenate([pts_g[:-1], pts_g[1:]], axis=1)
        lc_flown_bg.set_segments(segs_g)
        lc_flown.set_segments(segs_g)
        lc_flown.set_array((ss_g[:-1] + ss_g[1:]) / 2.0)

    # Punkt wykrycia i linia reakcji (pomarańczowa przerywana)
    drone_marker.set_data([sx_ext[idx_det]], [sy_ext[idx_det]])
    line_reaction.set_data(sx_ext[idx_det:idx_react + 1], sy_ext[idx_det:idx_react + 1])

    # Replanowana trasa z kolorami prędkości
    full_new = result['full_new_path']
    v_react_end = result['v_react_end']
    speeds_new = compute_path_speeds(full_new, initial_speed=v_react_end, accel=a)

    # Lead-in dla płynnego sklejenia
    heading = result['heading']
    drone_pos = result['drone_pos']
    if heading != (0, 0) and len(full_new) >= 3:
        p1 = (drone_pos[0] - heading[0] * 2.0, drone_pos[1] - heading[1] * 2.0)
        p2 = (drone_pos[0] - heading[0] * 1.0, drone_pos[1] - heading[1] * 1.0)
        combined = [p1, p2] + full_new
        lead_spd = np.full(2, v_react_end)
        combined_spd = np.concatenate([lead_spd, speeds_new])
        sx_n, sy_n, ss_n = smooth_path_with_speeds(combined, combined_spd, accel=a)
        # Trim lead-in
        search_lim = min(len(sx_n), 60)
        dists = (sx_n[:search_lim] - react_px) ** 2 + (sy_n[:search_lim] - react_py) ** 2
        best = int(np.argmin(dists))
        sx_n, sy_n, ss_n = sx_n[best:], sy_n[best:], ss_n[best:]
        if len(sx_n) > 0:
            sx_n[0] = sx_ext[idx_react]
            sy_n[0] = sy_ext[idx_react]
    else:
        sx_n, sy_n, ss_n = smooth_path_with_speeds(full_new, speeds_new, accel=a)
        if len(sx_n) > 0:
            sx_n[0] = sx_ext[idx_react]
            sy_n[0] = sy_ext[idx_react]

    if len(sx_n) > 1:
        pts_n = np.array([sx_n, sy_n]).T.reshape(-1, 1, 2)
        segs_n = np.concatenate([pts_n[:-1], pts_n[1:]], axis=1)
        lc_new_bg.set_segments(segs_n)
        lc_new.set_segments(segs_n)
        lc_new.set_array((ss_n[:-1] + ss_n[1:]) / 2.0)

    # Statystyki
    td, tt, tr, ttr = result['total']
    bd, bt, br, btr = result['baseline']
    dd, dt, dri, dtu = td - bd, tt - bt, tr - br, ttr - btr

    mode_label = "Omijanie" if mode == 'NORMAL' else "Tryb Powrotu"
    mode_w = f"W={w:.0f}" if mode == 'NORMAL' else "W=40"
    tc = title_color if mode == 'NORMAL' else 'orange'

    ax.set_title(
        f"{algo_name} — {mode_label} ({mode_w}, m={m:.0f} kg, r={dr:.2f} m, a={a:.1f} m/s²)\n"
        f"Droga: {td:.1f} m ({fmt(dd)} m nadłożono)\n"
        f"Czas: {tt:.1f}s ({fmt(dt)}) | Ryzyko: {tr:.1f} ({fmt(dri)}) | "
        f"Zakręty: {ttr} ({fmt_i(dtu)})",
        fontsize=14, color=tc, pad=25)


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
    dr_init = drone_radius_for_mass(DRONE_MASS_KG)
    a_init = MAX_THRUST_NET_N / DRONE_MASS_KG
    initial_title = (
        f"Optymalizacja tras BSP (Risk-Aware A*, W={RISK_WEIGHT:.0f}, m={DRONE_MASS_KG:.0f} kg, r={dr_init:.2f} m, a={a_init:.1f} m/s²)\n"
        f"Droga: {stats_global['length']:.1f} m | "
        f"Czas: {stats_global.get('flight_time', 0):.1f} s | "
        f"Ryzyko: {stats_global['risk']:.1f} | Zakręty: {turns}")
    ax.set_title(initial_title, fontsize=14, color='white', pad=25)

    gx_smooth, gy_smooth = smooth_path_bspline(path_global)
    initial_accel = MAX_THRUST_NET_N / DRONE_MASS_KG
    global_speeds = compute_path_speeds(path_global, accel=initial_accel)

    # Trasa pierwotna (szara przerywana) — bez etykiety, legenda budowana ręcznie
    line_global, = ax.plot(gx_smooth, gy_smooth, color='gray', linestyle='--', linewidth=2.5, alpha=0.8)

    lc_flown_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
    lc_flown = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=4)
    ax.add_collection(lc_flown_bg)
    ax.add_collection(lc_flown)

    line_reaction, = ax.plot([], [], color='orange', linestyle=':', linewidth=4, zorder=4)

    lc_new_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=4)
    lc_new = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=5)
    ax.add_collection(lc_new_bg)
    ax.add_collection(lc_new)

    ax.scatter([start[0]], [start[1]], color='lime', s=150, edgecolors='black', zorder=5)
    goal_marker = ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150,
                             edgecolors='black', zorder=5)
    drone_marker, = ax.plot([], [], 'o', color='yellow', markersize=12,
                            markeredgecolor='black', zorder=6)

    legend = ax.legend(handles=_build_legend_handles(),
                       loc='lower left', bbox_to_anchor=(1.04, -0.01), facecolor='#333333',
                       edgecolor='white', title="Risk-Aware A*")
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
        # ── TRYB IDLE: przed kliknięciem ──────────────────────────────────
        if sim_state["mode"] == "IDLE" and not sim_state["clicked"]:
            w = risk_slider.val
            m = mass_slider.val
            a_cur = MAX_THRUST_NET_N / m
            col_r = collision_radius_for_mass(m)
            phys_r = drone_radius_for_mass(m)
            env.update_drone_footprint(phys_r, col_r)

            new_path, new_stats = search_func(env, start, goal, risk_weight=w,
                                              turn_penalty=TURN_PENALTY,
                                              drone_radius=col_r, drone_mass=m)
            if new_path:
                sim_state["path_global"] = new_path
                sim_state["path_global_original"] = list(new_path)
                new_speeds = compute_path_speeds(new_path, accel=a_cur)
                sim_state["global_speeds"] = new_speeds

                gx_s, gy_s = smooth_path_bspline(new_path)
                line_global.set_data(gx_s, gy_s)

                dr = drone_radius_for_mass(m)
                flight_t = calculate_kinematic_flight_time(new_path, mass=m)
                ax.set_title(
                    f"Optymalizacja tras BSP (Risk-Aware A*, W={w:.0f}, m={m:.0f} kg, r={dr:.2f} m, a={a_cur:.1f} m/s²)\n"
                    f"Droga: {new_stats['length']:.1f} m | Czas: {flight_t:.1f} s | "
                    f"Ryzyko: {new_stats['risk']:.1f} | Zakręty: {new_stats.get('turns', 0)}",
                    fontsize=14, color='white', pad=25)
            else:
                ax.set_title("BRAK TRASY GLOBALNEJ!", color='red', fontsize=14, pad=25)
            fig.canvas.draw_idle()
            return

        # ── TRYB PO KLIKNIĘCIU: pełna symulacja ──────────────────────────
        if sim_state.get("obstacle_pos") is None:
            return

        w = risk_slider.val
        m = mass_slider.val
        grid_before = sim_state.get("grid_before_obstacle")
        if grid_before is None:
            return

        result = _compute_full_scenario(env, start, goal, search_func, w, m,
                                        sim_state["obstacle_pos"], grid_before)

        _draw_scenario(result, ax, lc_flown_bg, lc_flown, line_reaction, drone_marker,
                       lc_new_bg, lc_new, line_global, "Risk-Aware A*", 'lime', w, m)

        # ── RAPORT METRYK HAMOWANIA: 3 algorytmy w terminalu ──────────────
        if func_dijkstra is not None and func_astar is not None:
            result_dij = _compute_full_scenario(env, start, goal, func_dijkstra,
                                                w, m, sim_state["obstacle_pos"],
                                                grid_before)
            result_ast = _compute_full_scenario(env, start, goal, func_astar,
                                                w, m, sim_state["obstacle_pos"],
                                                grid_before)

            print_braking_comparison(
                results={
                    "Dijkstra":      analyze_braking_scenario(result_dij, m),
                    "A* Standard":   analyze_braking_scenario(result_ast, m),
                    "Risk-Aware A*": analyze_braking_scenario(result, m),
                },
                mass=m,
                risk_weight=w,
                obstacle_pos=sim_state["obstacle_pos"],
            )

        fig.canvas.draw_idle()

    risk_slider.on_changed(update_route)
    mass_slider.on_changed(update_route)

    def onclick(event):
        if event.inaxes != ax or sim_state["clicked"]:
            return

        click_x, click_y = int(event.xdata), int(event.ydata)
        sim_state["obstacle_pos"] = (click_x, click_y)
        sim_state["grid_before_obstacle"] = env.grid.copy()
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        sim_state["clicked"] = True

        # Pierwsze rysowanie Risk-Aware
        update_route(risk_slider.val)

        # Otwórz okna Dijkstra i A*
        if func_dijkstra is not None and func_astar is not None:
            _open_comparison_windows(
                env=env, start=start, goal=goal,
                sim_state=sim_state,
                func_dijkstra=func_dijkstra,
                func_astar=func_astar,
                init_w=risk_slider.val,
                init_mass=mass_slider.val,
            )

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    print("\n Kliknij na mapę aby dodać przeszkodę dynamiczną.")
    print("   Po kliknięciu otworzą się 3 okna porównawcze.\n")
    plt.show(block=True)


def _open_comparison_windows(
        env, start, goal,
        sim_state,
        func_dijkstra, func_astar,
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

        # Oryginalna trasa (szara przerywana) — bez etykiety
        gx_s, gy_s = smooth_path_bspline(sim_state["path_global"])
        global_line_cmp, = ax_cmp.plot(gx_s, gy_s, color='gray', linestyle='--', linewidth=2, alpha=0.5)

        # Droga przebyta — dynamiczna
        lc_f_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=3)
        lc_f = LineCollection([], cmap=get_speed_cmap(), norm=plt.Normalize(0, V_MAX_MS),
                              linewidths=5, zorder=4)
        ax_cmp.add_collection(lc_f_bg)
        ax_cmp.add_collection(lc_f)

        # Czas reakcji (pomarańczowa) i Punkt wykrycia — bez etykiet
        line_reaction_cmp, = ax_cmp.plot([], [], color='orange', linestyle=':', linewidth=4, zorder=4)
        detect_dot_cmp, = ax_cmp.plot([], [], 'o', color='yellow', markersize=12, markeredgecolor='black', zorder=6)

        # Start i cel — bez etykiet
        ax_cmp.scatter([start[0]], [start[1]], color='lime', s=150, edgecolors='black', zorder=5)
        ax_cmp.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, edgecolors='black', zorder=5)

        # Replanowana trasa (dynamiczna)
        lc_n_bg = LineCollection([], colors='#555555', linewidths=7, alpha=0.4, zorder=4)
        lc_n = LineCollection([], cmap=get_speed_cmap(), linewidths=5, norm=plt.Normalize(0, V_MAX_MS), zorder=5)
        ax_cmp.add_collection(lc_n_bg)
        ax_cmp.add_collection(lc_n)

        legend_cmp = ax_cmp.legend(handles=_build_legend_handles(),
                                   loc='lower left', bbox_to_anchor=(1.04, -0.01), facecolor='#333333',
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
                        func_ref, name_ref, clr_ref,
                        w_slider, m_slider, sim_st, react_line, dot_line, gline_ref):
            def update_cmp(val):
                w = w_slider.val
                m = m_slider.val
                if sim_st.get("obstacle_pos") is None:
                    return
                grid_before = sim_st.get("grid_before_obstacle")
                if grid_before is None:
                    return

                result = _compute_full_scenario(env, start, goal, func_ref, w, m,
                                                sim_st["obstacle_pos"], grid_before)

                _draw_scenario(result, ax_ref, lc_f_bg_ref, lc_f_ref,
                               react_line, dot_line, lc_bg_ref, lc_ref,
                               gline_ref, name_ref, clr_ref, w, m)
                ax_ref.figure.canvas.draw_idle()

            return update_cmp

        update_fn = make_update(ax_cmp, lc_n_bg, lc_n, lc_f_bg, lc_f,
                                algo_func, algo_name, color,
                                slider_cmp, mass_slider_cmp, sim_state,
                                line_reaction_cmp, detect_dot_cmp, global_line_cmp)
        slider_cmp.on_changed(update_fn)
        mass_slider_cmp.on_changed(update_fn)

        update_fn(RISK_WEIGHT)

        fig_cmp.canvas.manager.set_window_title(algo_name)
        fig_cmp._slider_ref = slider_cmp
        fig_cmp._mass_slider_ref = mass_slider_cmp
        plt.show(block=False)
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.interpolate as interp
from typing import List, Tuple, Callable, Any, Dict
from environment.grid_map import GridMap
from algorithms.common import calculate_segment_risk, calculate_path_length, generate_analysis_table
import os


def get_city_cmap() -> LinearSegmentedColormap:
    """
    Biały -> Żółty -> Pomarańczowy -> Czerwony -> Czarny
    """
    colors = [
        # WARTOŚĆ | KOLOR (R, G, B)
        (0.0,   (1.0, 1.0, 1.0)),   # 0.0  = Biały
        (0.01,  (1.0, 1.0, 0.0)),   # 0.01 = Żółty
        (0.4,   (1.0, 0.5, 0.0)),   # 0.4  = Pomarańczowy
        (0.8,   (1.0, 0.0, 0.0)),   # 0.8  = Czysta Czerwień
        (0.99,  (0.5, 0.0, 0.0)),   # 0.99 = Ciemna Czerwień
        (0.991, (0.0, 0.0, 0.0)),   # Odcięcie
        (1.0,   (0.0, 0.0, 0.0))    # 1.0  = Czarny
    ]
    return LinearSegmentedColormap.from_list("CityMapOrange", colors)


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
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.046, pad=0.05, shrink=0.80, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        if use_smoothing:
            ax.plot(path_x, path_y, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            smooth_x, smooth_y = smooth_path_bspline(path)
            ax.plot(smooth_x, smooth_y, color='cyan', linewidth=3, label='Trajektoria (Smooth)',
                    path_effects=[pe.withStroke(linewidth=5, foreground="blue")])
        else:
            ax.plot(path_x, path_y, color='cyan', linewidth=3, label='Trasa',
                    path_effects=[pe.withStroke(linewidth=5, foreground="blue")])

        # Start i Meta
        ax.scatter([path_x[0]], [path_y[0]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
        ax.scatter([path_x[-1]], [path_y[-1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                   zorder=5)

    #LEGENDA ELEMENTÓW
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
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.12, right=0.85, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)
    img = ax.imshow(grid_map.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.046, pad=0.05, shrink=0.80, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)
    line_raw, = ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    line_smooth, = ax.plot([], [], color='cyan', linewidth=3, label='Trasa',
                           path_effects=[pe.withStroke(linewidth=4, foreground="blue")])

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black', zorder=5)

    # LEGENDA ELEMENTÓW
    legend = ax.legend(
        loc='lower left',
        bbox_to_anchor=(1.05, -0.01),
        facecolor='#333333',
        edgecolor='white',
        title="Elementy Mapy"
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Suwak
    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.04], facecolor='#333333')
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
        path, stats = search_func(grid_map, start, goal, risk_weight=w, turn_penalty=20.0)

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            line_raw.set_data(px, py)
            sx, sy = smooth_path_bspline(path)
            line_smooth.set_data(sx, sy)
            turns_count = stats.get('turns', 0)
            title_text = (f"A* Risk-Aware (W={w:.0f})\n"
                          f"Dyst: {stats['length']:.1f} m | Czas Lotu: {stats.get('flight_time', 0):.1f} s | "
                          f"Ryzyko: {stats['risk']:.1f} | Zakręty: {turns_count}")
            ax.set_title(title_text, fontsize=14, pad=15)
        else:
            line_raw.set_data([], [])
            line_smooth.set_data([], [])
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
    # Trasa Startowa (W=20)
    path_global, stats_global = search_func(env, start, goal, risk_weight=20.0, turn_penalty=20.0,
                                            drone_radius=collision_radius)

    if not path_global:
        print("Błąd: Nie znaleziono trasy startowej.")
        return False

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.12, right=0.80, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    # Mapa
    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.048, pad=0.045, shrink=0.72, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    turns = stats_global.get('turns', 0)

    # --- POPRAWKA TYTUŁU: CZAS LOTU ZAMIAST CZASU OBLICZEŃ ---
    initial_title = (f"A* Risk-Aware (W=20) planowana trasa przelotu\n"
                     f"Dyst: {stats_global['length']:.1f} m | Czas Lotu: {stats_global.get('flight_time', 0):.1f} s | "
                     f"Ryzyko: {stats_global['risk']:.1f} | Zakręty: {turns}")

    ax.set_title(initial_title, fontsize=14, color='white', pad=15)

    gx_smooth, gy_smooth = smooth_path_bspline(path_global)

    # Elementy wykresu
    line_global, = ax.plot(gx_smooth, gy_smooth, color='black', linestyle='--', linewidth=2.5, alpha=0.8,
                           label='Pierwotny Plan')
    line_flown, = ax.plot([], [], color='lime', linewidth=3, label='Droga Przebyta', zorder=3)

    # NOWOŚĆ: Linia pokazująca drogę przebytą w czasie reakcji (bezwładność)
    line_reaction, = ax.plot([], [], color='orange', linestyle=':', linewidth=4, label='Czas Reakcji (Bezwładność)',
                             zorder=4)

    line_new, = ax.plot([], [], color='cyan', linewidth=3, label='Replanowana Trasa',
                        path_effects=[pe.withStroke(linewidth=5, foreground="blue")], zorder=4)

    ax.scatter([start[0]], [start[1]], color='lime', s=150, label='Start', edgecolors='black', zorder=5)
    goal_marker = ax.scatter([goal[0]], [goal[1]], color='magenta', marker='X', s=150, label='Cel', edgecolors='black',
                             zorder=5)
    drone_marker, = ax.plot([], [], 'o', color='yellow', markersize=12, label='Wykrycie (Zasięg)',
                            markeredgecolor='black', zorder=6)

    # Legenda
    legend = ax.legend(loc='lower left', bbox_to_anchor=(1.04, -0.01), facecolor='#333333', edgecolor='white',
                       title="Legenda Elementów")
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Suwak
    ax_slider = plt.axes([0.151, 0.04, 0.77, 0.04], facecolor='#333333')
    risk_slider = Slider(ax=ax_slider, label='Waga Ryzyka (W) ', valmin=0.0, valmax=100.0, valinit=20.0, valstep=1.0,
                         color='cyan')
    risk_slider.label.set_color('white')
    risk_slider.valtext.set_color('white')

    sim_state = {"clicked": False, "drone_pos": None, "target_pos": None, "mode": "IDLE"}

    def update_route(val):
        if not sim_state["clicked"] or sim_state["mode"] in ["CRASH", "IGNORE", "IDLE"]: return

        if sim_state["mode"] == "RTH":
            w = 50.0
        else:
            w = risk_slider.val

        path_local, stats = search_func(env, sim_state["drone_pos"], sim_state["target_pos"], risk_weight=w,
                                        turn_penalty=20.0,
                                        drone_radius=collision_radius,
                                        initial_direction=sim_state.get("heading", (0, 0)),
                                        current_speed=sim_state.get("drone_speed", 0.0))

        if path_local:
            nx, ny = smooth_path_bspline(path_local)
            line_new.set_data(nx, ny)

            turns_count = stats.get('turns', 0)

            # --- POPRAWKA TYTUŁU: CZAS LOTU ZAMIAST CZASU OBLICZEŃ ---
            stats_text = (f"Dyst: {stats['length']:.1f}m | Czas Lotu: {stats.get('flight_time', 0):.1f} s | "
                          f"Ryzyko: {stats['risk']:.1f} | Zakręty: {turns_count}")

            if sim_state["mode"] == "RTH":
                ax.set_title(f"Tryb Powrotu | W={w:.0f}\n{stats_text}", color='orange', fontsize=14, pad=15)
                line_new.set_color('orange')
            else:
                ax.set_title(f"OMIJANIE PRZESZKODY | W={w:.0f}\n{stats_text}", color='lime', fontsize=14, pad=15)
                line_new.set_color('cyan')
        else:
            ax.set_title(f"DRON JEST UWIĘZIONY!", color='red', fontsize=14, pad=15)
            line_new.set_data([], [])
        fig.canvas.draw_idle()

    risk_slider.on_changed(update_route)

    def onclick(event):
        if event.inaxes != ax: return
        if sim_state["clicked"]: return

        click_x, click_y = int(event.xdata), int(event.ydata)
        OBSTACLE_RADIUS = 8
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        # --- PARAMETRY ŚRODOWISKA ---
        DRONE_RADIUS = collision_radius - 2  # <--- Tutaj była zguba!
        SENSOR_RANGE = 50.0
        CRASH_DIST = OBSTACLE_RADIUS + collision_radius

        # --- ETAP 1: Czy przeszkoda w ogóle przecina naszą zaplanowaną trasę? ---
        is_path_blocked = False
        for (px, py) in path_global:
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= CRASH_DIST:
                is_path_blocked = True
                break

        if not is_path_blocked:
            print("\n-> Radar wykrył obiekt, ale nie leży on na kursie kolizyjnym. Dron ignoruje zagrożenie.")
            ax.set_title("Zagrożenie poza kursem lotu. Brak reakcji.", color='lime', fontsize=14, pad=15)
            line_flown.set_data(gx_smooth, gy_smooth)
            sim_state["clicked"] = True
            sim_state["mode"] = "IGNORE"
            fig.canvas.draw()
            return

        # --- ETAP 2: Szukamy momentu, w którym radar zasygnalizuje problem (60m). ---
        collision_idx = -1
        for i, (px, py) in enumerate(path_global):
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= SENSOR_RANGE:
                collision_idx = i
                break

        if collision_idx == -1:
            return

            # --- DYNAMICZNY MODEL KINEMATYCZNY CHWILOWEJ PRĘDKOŚCI I HAMOWANIA ---
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

        # Obliczenia hamowania w czasie zwłoki układu nawigacji
        t_stop = v_detect / acceleration
        t_react = min(processing_delay, t_stop)

        reaction_distance_meters = (v_detect * t_react) - (0.5 * acceleration * (t_react ** 2))
        reaction_indices = int(np.ceil(reaction_distance_meters))

        v_react_end = max(0.0, v_detect - acceleration * processing_delay)
        print(
            f"-> Po czasie reakcji ({processing_delay}s) dron przeleciał {reaction_distance_meters:.1f}m i zwolnił do: {v_react_end:.1f} m/s")

        drone_detect_idx = collision_idx
        drone_react_idx = min(drone_detect_idx + reaction_indices, len(path_global) - 1)
        reaction_path = path_global[drone_detect_idx:drone_react_idx + 1]

        # Wektor uderzeniowy (pęd) do algorytmu
        if len(reaction_path) >= 2:
            last_dx = reaction_path[-1][0] - reaction_path[-2][0]
            last_dy = reaction_path[-1][1] - reaction_path[-2][1]
            flight_heading = (int(np.sign(last_dx)), int(np.sign(last_dy)))
        else:
            flight_heading = (0, 0)

        flown_raw = path_global[:drone_detect_idx + 1]
        fx, fy = smooth_path_bspline(flown_raw) if len(flown_raw) > 2 else ([p[0] for p in flown_raw],
                                                                            [p[1] for p in flown_raw])
        line_flown.set_data(fx, fy)

        rx, ry = smooth_path_bspline(reaction_path) if len(reaction_path) > 2 else ([p[0] for p in reaction_path],
                                                                                    [p[1] for p in reaction_path])
        line_reaction.set_data(rx, ry)

        drone_marker.set_data([path_global[drone_detect_idx][0]], [path_global[drone_detect_idx][1]])

        # WERYFIKACJA KATASTROFY PRZY ZBYT DUŻEJ PRĘDKOŚCI
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

        current_drone_pos = path_global[drone_react_idx]

        sim_state["drone_pos"] = current_drone_pos
        sim_state["clicked"] = True
        sim_state["heading"] = flight_heading
        sim_state["drone_speed"] = v_react_end

        # --- DECYZJA O NOWEJ TRASIE Z UŻYCIEM PĘDU FIZYCZNEGO ---
        path_check, _ = search_func(env, current_drone_pos, goal, risk_weight=20.0, turn_penalty=20.0,
                                    drone_radius=collision_radius,
                                    initial_direction=flight_heading,
                                    current_speed=v_react_end)

        if path_check:
            sim_state["target_pos"] = goal
            sim_state["mode"] = "NORMAL"
            line_new.set_label('Replanowana Trasa')
        else:
            sim_state["target_pos"] = start
            sim_state["mode"] = "RTH"
            goal_marker.set_facecolor('gray')
            line_new.set_label('Powrót (Awaryjny)')
            risk_slider.set_val(50.0)

        if sim_state["mode"] == "NORMAL":
            update_route(risk_slider.val)

        path_remainder = path_global[drone_react_idx:]
        base_risk = calculate_segment_risk(path_remainder, env)
        base_len = calculate_path_length(path_remainder)

        generate_analysis_table(
            env=env,
            start_pos=current_drone_pos,
            target_pos=sim_state["target_pos"],
            search_func=search_func,
            base_len=base_len,
            base_risk=base_risk,
            collision_radius=collision_radius,
            table_title="ANALIZA TRYBU ONLINE (H4)"
        )

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)


def generate_thesis_charts(
        envs: List[GridMap],  # <--- ZMIANA: Przyjmujemy LISTĘ map!
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable,
        collision_radius: float,
        stats_d: Dict[str, Any] = None,
        stats_a: Dict[str, Any] = None,
        density_label: str = ""
) -> None:
    print("\n--- GENEROWANIE WYKRESÓW DO PRACY DYPLOMOWEJ ---")

    output_dir = "research_results"
    if density_label:
        output_dir = os.path.join(output_dir, f"gestosc_{density_label}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Utworzono katalog: {output_dir}")

    print(f"Zbieranie danych dla wag W od 0 do 100 (Przetwarzanie {len(envs)} map)...")

    weights = range(0, 105, 5)
    data_w, data_len, data_risk, data_time, data_nodes = [], [], [], [], []
    data_flight_time, data_turns = [], []

    # --- ZMIANA: Pętla uśredniająca wyniki z wielu map (Monte Carlo) ---
    for w in weights:
        w_len, w_risk, w_time, w_nodes, w_flight, w_turns = 0, 0, 0, 0, 0, 0
        found_count = 0

        for env in envs:
            path, stats = search_func(env, start, goal, risk_weight=float(w), turn_penalty=20.0,
                                      drone_radius=collision_radius)
            if stats['found']:
                w_len += stats['length']
                w_risk += stats['risk']
                w_time += stats['time'] * 1000
                w_nodes += stats['nodes']
                w_flight += stats.get('flight_time', 0.0)
                w_turns += stats.get('turns', 0)
                found_count += 1

        if found_count > 0:
            data_w.append(w)
            data_len.append(w_len / found_count)
            data_risk.append(w_risk / found_count)
            data_time.append(w_time / found_count)
            data_nodes.append(w_nodes / found_count)
            data_flight_time.append(w_flight / found_count)
            data_turns.append(w_turns / found_count)
        else:
            print(f"Ostrzeżenie: Dla W={w} nie znaleziono trasy na żadnej z próbkowanych map.")

    plt.style.use('default')

    # --- WYKRES 1: KOMPROMIS (Pareto) ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter = ax1.scatter(data_len, data_risk, c=data_w, cmap='viridis', s=100, zorder=2, edgecolor='black')
    ax1.plot(data_len, data_risk, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax1.set_title("Optymalizacja Wielokryterialna: Ryzyko vs Dystans", fontsize=14, pad=15)
    ax1.set_xlabel("Długość Trasy [m]", fontsize=12)
    ax1.set_ylabel("Całkowity Koszt Ryzyka (bezjednostkowy)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Waga Ryzyka (W)', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_pareto_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- WYKRES 2: KOSZT OBLICZENIOWY ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    ax2a.bar(data_w, data_time, width=3, color='purple', alpha=0.7)
    ax2a.set_title("Czas Obliczeń", fontsize=12)
    ax2a.set_xlabel("Waga Ryzyka (W)")
    ax2a.set_ylabel("Czas [ms]")
    ax2a.grid(axis='y', linestyle='--', alpha=0.5)

    ax2b.plot(data_w, data_nodes, marker='o', color='green')
    ax2b.set_title("Złożoność (Odwiedzone Węzły)", fontsize=12)
    ax2b.set_xlabel("Waga Ryzyka (W)")
    ax2b.set_ylabel("Liczba Węzłów")
    ax2b.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Analiza Wydajności Algorytmu", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_performance_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # =========================================================================
    # --- WYKRES 3: GŁÓWNY WYKRES PORÓWNAWCZY (BENCHMARK) DLA PROFESORA ---
    # =========================================================================
    if stats_d and stats_a and 20 in data_w:
        idx_20 = data_w.index(20)  # Bierzemy do porównania W=20 (zbilansowane ryzyko)

        labels = ['Dijkstra\n(Matematyczna)', 'A* Standard\n(Szybka)', 'Risk-Aware A*\n(Kinematyczna)']
        colors = ['#888888', '#33bbee', '#33ee33']

        # Pobranie danych dla wszystkich algorytmów
        dist_vals = [stats_d.get('length', 0), stats_a.get('length', 0), data_len[idx_20]]
        risk_vals = [stats_d.get('risk', 0), stats_a.get('risk', 0), data_risk[idx_20]]
        time_vals = [stats_d.get('flight_time', 0), stats_a.get('flight_time', 0), data_flight_time[idx_20]]
        turns_vals = [stats_d.get('turns', 0), stats_a.get('turns', 0), data_turns[idx_20]]

        fig3, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig3.suptitle("Benchmarking Algorytmów Planowania Lotu (DJI FlyCart 30)", fontsize=18, fontweight='bold')

        # 1. Dystans (Dijkstra wygrywa)
        axs[0, 0].bar(labels, dist_vals, color=colors, edgecolor='black', alpha=0.9)
        axs[0, 0].set_title("1. Długość Trasy (Geometryczna) [m]", fontsize=13)
        axs[0, 0].set_ylabel("Metry")
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

        # 2. Ryzyko (Twój algorytm deklasuje resztę)
        axs[0, 1].bar(labels, risk_vals, color=colors, edgecolor='black', alpha=0.9)
        axs[0, 1].set_title("2. Poziom Ekspozycji na Ryzyko", fontsize=13)
        axs[0, 1].set_ylabel("Wartość Ryzyka")
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

        # 3. Czas lotu (Twój algorytm wygrywa mimo dłuższej trasy!)
        axs[1, 0].bar(labels, time_vals, color=colors, edgecolor='black', alpha=0.9)
        axs[1, 0].set_title("3. Fizyczny Czas Przelotu (Kinematyka) [s]", fontsize=13)
        axs[1, 0].set_ylabel("Sekundy")
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

        # 4. Płynność / Zakręty (Twój algorytm wygrywa)
        axs[1, 1].bar(labels, turns_vals, color=colors, edgecolor='black', alpha=0.9)
        axs[1, 1].set_title("4. Liczba Wykonanych Manewrów (Płynność)", fontsize=13)
        axs[1, 1].set_ylabel("Zakręty")
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

        # Dodanie cyfrowych wartości na szczycie słupków
        for ax, vals in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]],
                            [dist_vals, risk_vals, time_vals, turns_vals]):
            max_v = max(vals) if max(vals) > 0 else 1
            for i, v in enumerate(vals):
                ax.text(i, v + (max_v * 0.02), f"{v:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename3 = os.path.join(output_dir, "3_algorithm_comparison_benchmark.png")
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        print(f"Zapisano: {filename3}")
        plt.close(fig3)

    print("\nGotowe! Wszystkie wykresy zapisano w folderze 'research_results'.")
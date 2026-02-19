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
                          f"Dyst: {stats['length']:.1f} m | Ryzyko: {stats['risk']:.1f} | "
                          f"Czas: {stats['time']:.4f} s | Zakręty: {turns_count}")
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
    path_global, stats_global = search_func(env, start, goal, risk_weight=20.0, turn_penalty=20.0, drone_radius=collision_radius)

    if not path_global:
        print("Błąd: Nie znaleziono trasy startowej.")
        return False

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(bottom=0.12, right=0.80, left=0.15, top=0.90)
    setup_dark_theme(fig, ax)

    # Mapa
    img = ax.imshow(env.grid.T, origin='lower', cmap=get_city_cmap(), vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, location='right', fraction=0.048, pad=0.045, shrink=0.72, anchor=(0.0, 1.0))
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Bezpiecznie', 'Ryzyko', 'BUDYNEK'])
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label('Poziom Ryzyka', color='white', labelpad=10)

    turns = stats_global.get('turns', 0)
    initial_title = (f"A* Risk-Aware (W=20) planowana trasa przelotu\n"
                     f"Dyst: {stats_global['length']:.1f} m | Ryzyko: {stats_global['risk']:.1f} | "
                     f"Czas: {stats_global['time']:.4f} s | Zakręty: {turns}")

    ax.set_title(initial_title, fontsize=14, color='white', pad=15)

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
            w = 50.0  # Sztywna wysoka waga ryzyka dla trybu powrotu
        else:
            w = risk_slider.val # Dynamiczna waga ryzyka dla trybu omijania przeszkody
        path_local, stats = search_func(env, sim_state["drone_pos"], sim_state["target_pos"], risk_weight=w, turn_penalty=20.0, drone_radius=collision_radius)

        if path_local:
            nx, ny = smooth_path_bspline(path_local)
            line_new.set_data(nx, ny)

            turns_count = stats.get('turns', 0)
            stats_text = (f"Dyst: {stats['length']:.1f}m | Ryzyko: {stats['risk']:.1f} | "
                          f"Czas: {stats['time']:.4f}s | Zakręty: {turns_count}")
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
        print(f"\n Kliknięcie w: ({click_x}, {click_y})")
        OBSTACLE_RADIUS = 8
        env.add_dynamic_risk_zone(click_x, click_y, radius=OBSTACLE_RADIUS)
        img.set_data(env.grid.T)

        DRONE_RADIUS = collision_radius - 2 # Promień drona
        SENSOR_RANGE = 5.0  # Zasięg czujnika

        # Dystans graniczny = Promień Przeszkody + Promień Drona + Zasięg Czujnika
        # = 8 + 1 + 5 = 14.0
        LIMIT_DIST = OBSTACLE_RADIUS + DRONE_RADIUS + SENSOR_RANGE

        collision_idx = 0
        for i, (px, py) in enumerate(path_global):
            dist_to_center = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)
            if dist_to_center <= LIMIT_DIST:
                collision_idx = i
                break

        if collision_idx == -1:
            print("Zagrożenie daleko.")
            ax.set_title("Zagrożenie poza zasięgiem (>5m).", color='lime', fontsize=14)
            line_flown.set_data(gx_smooth, gy_smooth)
            sim_state["clicked"] = True
            sim_state["mode"] = "IGNORE"
            fig.canvas.draw()
            return

        drone_idx = collision_idx
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

        # Jeśli środek drona jest bliżej niż (Promień przeszkody + Promień drona)
        dist_to_drone = np.sqrt((click_x - current_drone_pos[0]) ** 2 + (click_y - current_drone_pos[1]) ** 2)
        if dist_to_drone <= (OBSTACLE_RADIUS + DRONE_RADIUS):
            ax.set_title("Dron nie wystartował!", color='red', fontsize=14)
            sim_state["clicked"] = True
            sim_state["mode"] = "CRASH"
            fig.canvas.draw()
            return

        # DECYZJA
        path_check, _ = search_func(env, current_drone_pos, goal, risk_weight=20.0, turn_penalty=20.0, drone_radius=collision_radius)

        sim_state["drone_pos"] = current_drone_pos
        sim_state["clicked"] = True

        if path_check:
            print("Cel osiągalny.")
            sim_state["target_pos"] = goal
            sim_state["mode"] = "NORMAL"
            line_new.set_label('Replanowana Trasa')
        else:
            print("Cel zablokowany")
            sim_state["target_pos"] = start
            sim_state["mode"] = "RTH"
            goal_marker.set_facecolor('gray')
            line_new.set_label('Powrót (Awaryjny)')

            risk_slider.set_val(50.0)

        if sim_state["mode"] == "NORMAL":
            update_route(risk_slider.val)

        print("\nGENEROWANIE TABELI ANALIZY")
        path_remainder = path_global[drone_idx:]
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
            table_title="ANALIZA TRYBU ONLINE (H3)"
        )

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)


def generate_thesis_charts(
        env: GridMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        search_func: Callable,
        collision_radius: float
) -> None:
    print("\n--- GENEROWANIE WYKRESÓW DO PRACY DYPLOMOWEJ ---")

    # 1. Tworzenie katalogu na wyniki
    output_dir = "research_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Utworzono katalog: {output_dir}")

    print("Zbieranie danych dla wag W od 0 do 100...")

    weights = range(0, 105, 5)
    data_w = []
    data_len = []
    data_risk = []
    data_time = []
    data_nodes = []

    # 2. Zbieranie danych
    for w in weights:
        path, stats = search_func(env, start, goal, risk_weight=float(w), turn_penalty=20.0,
                                  drone_radius=collision_radius)

        if stats['found']:
            data_w.append(w)
            data_len.append(stats['length'])
            data_risk.append(stats['risk'])
            data_time.append(stats['time'] * 1000)  # ms
            data_nodes.append(stats['nodes'])
        else:
            print(f"Ostrzeżenie: Dla W={w} nie znaleziono trasy.")

    # Ustawiamy styl na domyślny (białe tło, dobre do druku)
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

    for i, w in enumerate(data_w):
        if w in [0, 20, 50, 100]:
            ax1.annotate(f"W={w}", (data_len[i], data_risk[i]), xytext=(5, 5), textcoords='offset points', fontsize=9,
                         fontweight='bold')

    plt.tight_layout()
    # ZAPIS
    filename1 = os.path.join(output_dir, "1_pareto_tradeoff.png")
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {filename1}")
    plt.close(fig1)  # Zamykamy, żeby nie wisiało w pamięci

    # --- WYKRES 2: ANALIZA WRAŻLIWOŚCI ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    color_len = 'tab:blue'
    ax2.set_xlabel('Współczynnik Wagi Ryzyka (W)', fontsize=12)
    ax2.set_ylabel('Długość Trasy [m]', color=color_len, fontsize=12)
    ax2.plot(data_w, data_len, color=color_len, marker='o', linewidth=2, label='Długość')
    ax2.tick_params(axis='y', labelcolor=color_len)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2_twin = ax2.twinx()
    color_risk = 'tab:red'
    ax2_twin.set_ylabel('Całkowite Ryzyko', color=color_risk, fontsize=12)
    ax2_twin.plot(data_w, data_risk, color=color_risk, marker='s', linewidth=2, linestyle='--', label='Ryzyko')
    ax2_twin.tick_params(axis='y', labelcolor=color_risk)

    plt.title("Wpływ Wagi (W) na parametry trasy", fontsize=14, pad=15)
    fig2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=2)

    plt.tight_layout()
    # ZAPIS
    filename2 = os.path.join(output_dir, "2_sensitivity_analysis.png")
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {filename2}")
    plt.close(fig2)

    # --- WYKRES 3: KOSZT OBLICZENIOWY ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    # Czas
    ax3a.bar(data_w, data_time, width=3, color='purple', alpha=0.7)
    ax3a.set_title("Czas Obliczeń", fontsize=12)
    ax3a.set_xlabel("Waga Ryzyka (W)")
    ax3a.set_ylabel("Czas [ms]")
    ax3a.grid(axis='y', linestyle='--', alpha=0.5)

    # Węzły
    ax3b.plot(data_w, data_nodes, marker='o', color='green')
    ax3b.set_title("Złożoność (Odwiedzone Węzły)", fontsize=12)
    ax3b.set_xlabel("Waga Ryzyka (W)")
    ax3b.set_ylabel("Liczba Węzłów")
    ax3b.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Analiza Wydajności Algorytmu", fontsize=14)
    plt.tight_layout()
    # ZAPIS
    filename3 = os.path.join(output_dir, "3_performance_metrics.png")
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {filename3}")
    plt.close(fig3)

    print("\nGotowe! Wszystkie wykresy zapisano w folderze 'research_results'.")
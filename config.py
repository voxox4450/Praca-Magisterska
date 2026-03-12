# =============================================================================
# config.py – CENTRALNA KONFIGURACJA SYSTEMU PLANOWANIA TRAS BSP
# Wszystkie parametry fizyczne, algorytmiczne i środowiskowe w jednym miejscu.
# Zmiana tutaj = zmiana wszędzie automatycznie.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ŚRODOWISKO / MAPA
# ─────────────────────────────────────────────────────────────────────────────
MAP_SIZE: int = 200                  # Rozmiar mapy [kratki]
RISK_ZONES_COUNT: int = 5           # Liczba stref ryzyka (ujednolicona dla wszystkich trybów)
RANDOM_SEED: int = 42                # Ziarno losowości (dla reprodukowalności)

START_POS = (5, 5)                   # Pozycja startowa drona
GOAL_POS  = (195, 195)               # Pozycja docelowa drona

# [FIX #17] Jawna definicja przelicznika kratka → metr.
# CAŁA kinematyka (V_MAX, ACCELERATION, dist_cost, braking_dist) zakłada tę równość.
# Zmiana CELL_SIZE_M wymaga przeliczenia WSZYSTKICH parametrów fizycznych.
CELL_SIZE_M: float = 1.0             # 1 kratka = 1 metr

# Ochrona startu/celu przed budynkami [kratki]
BUILDING_SAFE_MARGIN: float = 4.0

# Parametry gradientu ryzyka wokół budynków
GRADIENT_RANGE: int   = 15           # Zasięg gradientu [kratki]
GRADIENT_DECAY: float = 6.0          # Stała zaniku eksponencjalnego (exp(-d / DECAY))

# [FIX #6] Jawne progi kolizji i ryzyka:
#   grid == 1.0                          → budynek (fizyczna przeszkoda)
#   COLLISION_GRID_THRESHOLD ≤ grid < 1  → strefa kolizji (jak budynek w is_collision)
#   0.0 < grid < COLLISION_GRID_THRESHOLD → strefa ryzyka (przelotowa, kosztowna)
#   grid == 0.0                          → przestrzeń wolna
COLLISION_GRID_THRESHOLD: float = 0.90
SOFT_RISK_CAP: float = 0.85          # Maksymalna wartość miękkiej strefy ryzyka

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRY FIZYCZNE DRONA
# ─────────────────────────────────────────────────────────────────────────────
DRONE_RADIUS_M: float  = 1.0         # Fizyczny promień drona [m]
SAFE_MARGIN_M: float   = 2.0         # Margines bezpieczeństwa od budynków [m]

# COLLISION_RADIUS = fizyczny promień + margines [m = kratki przy CELL_SIZE_M=1.0]
COLLISION_RADIUS: float = DRONE_RADIUS_M + SAFE_MARGIN_M   # = 3.0

DRONE_MASS_KG: float       = 30.0    # Masa drona [kg]
MAX_THRUST_NET_N: float    = 120.0   # Ciąg netto (po odjęciu ciężaru) [N]

# Kinematyka liniowa
V_MAX_MS: float        = 18.0        # Maksymalna prędkość przelotowa [m/s]
ACCELERATION: float    = MAX_THRUST_NET_N / DRONE_MASS_KG  # = 4.0 [m/s²]

# Kinematyka zakrętów
MAX_LATERAL_ACCEL: float = 7.0       # Maks. przyspieszenie dośrodkowe [m/s²] (~0.7g)
MIN_TURN_SPEED: float    = 0.5       # Minimalna prędkość w zakręcie [m/s]

# Formuła: r_turn = TURN_RADIUS_CONST / sin(angle/2)
# Pochodzenie: dla łuku kołowego wpisanego między dwa odcinki o długości d
# spotykające się pod kątem zwrotu θ: r ≈ d / (2·sin(θ/2)).
# Dla siatki 8-kierunkowej średnia długość kroku ≈ (1.0 + 1.41)/2 ≈ 1.2,
# d/2 ≈ 0.6 — zaokrąglone w górę do 1.5 jako margines bezpieczeństwa,
# uwzględniający dyskretyzację i wygładzanie B-spline.
TURN_RADIUS_CONST: float = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRY ALGORYTMÓW
# ─────────────────────────────────────────────────────────────────────────────
RISK_WEIGHT: float    = 20.0         # Domyślna waga kosztu ryzyka (W)

# Wszystkie 3 algorytmy (Dijkstra, A*, Risk-Aware A*) operują na tym samym
# modelu drona → ta sama formuła kary za zakręt:
#   turn_cost = TURN_PENALTY × (angle / (π/2))
# Różnica między algorytmami leży WYŁĄCZNIE w:
#   - heurystyce (Dijkstra: brak, A*/Risk: euklidesowa)
#   - kinematyce (tylko Risk-Aware: profil prędkości + hamowanie)
TURN_PENALTY: float = 20.0

# Mnożnik > 1.0 → Weighted A* (ε = multiplier − 1.0).
# Gwarancja: koszt rozwiązania ≤ (1+ε) × optimum.
# Ustawienie 1.0 → czysta dopuszczalność (gwarancja optymalności, wolniejsze).
HEURISTIC_MULT_ASTAR: float = 1.001  # ε = 0.001 (~0.1% ponad optimum)
HEURISTIC_MULT_RISK:  float = 1.001  # ε = 0.001

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────
N_TESTS: int = 50                # Liczba prób Monte Carlo

# Wagi ryzyka w sweep Pareto
PARETO_WEIGHT_STEP: int = 5
PARETO_WEIGHT_MAX:  int = 50

# ─────────────────────────────────────────────────────────────────────────────
# PLANOWANIE KINEMATYCZNE / SYMULACJA ONLINE
# ─────────────────────────────────────────────────────────────────────────────
# [FIX #24] Zasięg sensora — importowany z config wszędzie (nie hardcoded)
SENSOR_RANGE: int = 60               # Zasięg sensora drona [kratki]
# ─────────────────────────────────────────────────────────────────────────────
# ŚRODOWISKO / MAPA
# ─────────────────────────────────────────────────────────────────────────────
MAP_SIZE: int = 200                  # Rozmiar mapy [kratki]
RISK_ZONES_COUNT: int = 5           # Liczba stref ryzyka
RANDOM_SEED: int = 42                # Ziarno losowości

START_POS = (5, 5)                   # Pozycja startowa drona
GOAL_POS  = (195, 195)               # Pozycja docelowa drona

CELL_SIZE_M: float = 1.0             # 1 kratka = 1 metr

# Ochrona startu/celu przed budynkami [kratki]
BUILDING_SAFE_MARGIN: float = 4.0

# Parametry gradientu ryzyka wokół budynków
GRADIENT_RANGE: int   = 15           # Zasięg gradientu [kratki]
GRADIENT_DECAY: float = 6.0          # Stała zaniku eksponencjalnego (exp(-d / DECAY))

#   grid == 1.0 budynek (fizyczna przeszkoda)
#   COLLISION_GRID_THRESHOLD ≤ grid < 1 strefa kolizji
#   0.0 < grid < COLLISION_GRID_THRESHOLD strefa ryzyka (przelotowa, kosztowna)
#   grid == 0.0 przestrzeń wolna
COLLISION_GRID_THRESHOLD: float = 0.90
SOFT_RISK_CAP: float = 0.85          # Maksymalna wartość miękkiej strefy ryzyka

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRY DRONA
# ─────────────────────────────────────────────────────────────────────────────
DRONE_RADIUS_M: float  = 1.0         # Fizyczny promień drona [m]
SAFE_MARGIN_M: float   = 2.0         # Margines bezpieczeństwa od budynków [m]

# COLLISION_RADIUS = fizyczny promień + margines [m = kratki przy CELL_SIZE_M=1.0]
COLLISION_RADIUS: float = DRONE_RADIUS_M + SAFE_MARGIN_M   # = 3.0

DRONE_MASS_KG: float       = 30.0    # Masa drona [kg]
MAX_THRUST_NET_N: float    = 120.0   # Ciąg [N]

# Kinematyka
V_MAX_MS: float        = 18.0        # Maksymalna prędkość przelotowa [m/s]
ACCELERATION: float    = MAX_THRUST_NET_N / DRONE_MASS_KG  # = 4.0 [m/s²]

# Kinematyka zakrętów
MAX_LATERAL_ACCEL: float = 7.0       # Maks. przyspieszenie dośrodkowe [m/s²] (~0.7g)
MIN_TURN_SPEED: float    = 0.5       # Minimalna prędkość w zakręcie [m/s]

# Formuła: r_turn = TURN_RADIUS_CONST / sin(angle/2)
TURN_RADIUS_CONST: float = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRY ALGORYTMÓW
# ─────────────────────────────────────────────────────────────────────────────
RISK_WEIGHT: float    = 20.0         # Domyślna waga kosztu ryzyka (W)

# Wszystkie 3 algorytmy (Dijkstra, A*, Risk-Aware A*) operują na tym samym
#   - heurystyce (Dijkstra: brak, A*/Risk: euklidesowa — TA SAMA dla obu)
#   - kinematyce (tylko Risk-Aware: profil prędkości + bufor hamowania)
TURN_PENALTY: float = 20.0

# Mnożnik heurystyki — TEN SAM dla A* Standard i Risk-Aware A*.
# Wartość 1.0 → czysta dopuszczalność (gwarancja optymalności).
# Mnożnik > 1.0 → Weighted A* (ε = multiplier − 1.0), szybsze ale nieoptymalne.
HEURISTIC_MULTIPLIER: float = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# SYMULACJA ONLINE
# ─────────────────────────────────────────────────────────────────────────────
OBSTACLE_RADIUS: int = 8             # Promień dynamicznej przeszkody [kratki]
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

# Ochrona startu/celu przed budynkami [kratki]
BUILDING_SAFE_MARGIN: float = 4.0

# Parametry gradientu ryzyka wokół budynków
GRADIENT_RANGE: int   = 15           # Zasięg gradientu [kratki]
GRADIENT_DECAY: float = 6.0          # Stała zaniku eksponencjalnego (exp(-d / DECAY))

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRY FIZYCZNE DRONA
# ─────────────────────────────────────────────────────────────────────────────
DRONE_RADIUS_M: float  = 1.0         # Fizyczny promień drona [m]
SAFE_MARGIN_M: float   = 2.0         # Margines bezpieczeństwa od budynków [m]

# COLLISION_RADIUS używany przez algorytmy = fizyczny promień + margines
COLLISION_RADIUS: float = DRONE_RADIUS_M + SAFE_MARGIN_M   # = 3.0 [m]

DRONE_MASS_KG: float       = 30.0    # Masa drona [kg]
MAX_THRUST_NET_N: float    = 120.0   # Ciąg netto (po odjęciu ciężaru) [N]

# Kinematyka liniowa
V_MAX_MS: float        = 18.0        # Maksymalna prędkość przelotowa [m/s]
#   UWAGA: Poprzednio używane 65 km/h = 18.05 m/s → zaokrąglone do 18.0 m/s
#   65 km/h / 3.6 = 18.055... m/s  →  używamy V_MAX_MS = 18.0 WSZĘDZIE

ACCELERATION: float    = MAX_THRUST_NET_N / DRONE_MASS_KG  # = 4.0 [m/s²]

# Kinematyka zakrętów
MAX_LATERAL_ACCEL: float = 7.0       # Maks. przyspieszenie dośrodkowe [m/s²] (~0.7g)
MIN_TURN_SPEED: float    = 0.5       # Minimalna prędkość w zakręcie [m/s]

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRY ALGORYTMÓW
# ─────────────────────────────────────────────────────────────────────────────
RISK_WEIGHT: float    = 20.0         # Domyślna waga kosztu ryzyka (W)

# Kary za zakręty – celowo rozdzielone:
#   CLASSIC: stała kara niezależna od kąta – Dijkstra i A* Standard pozostają
#            w oryginalnej, niezmienionej formie (algorytmy bazowe/baseline).
#   RISK:    kara proporcjonalna do kąta zakrętu – część modelu kinematycznego
#            Risk-Aware A*.
#            Wzór: turn_cost = TURN_PENALTY_RISK * (angle / (pi/2))
#            →  zakręt  45° = 0.50 × TURN_PENALTY_RISK
#            →  zakręt  90° = 1.00 × TURN_PENALTY_RISK
#            →  zakręt 135° = 1.50 × TURN_PENALTY_RISK
TURN_PENALTY_CLASSIC: float = 20.0   # Używane przez: Dijkstra, A* Standard
TURN_PENALTY_RISK: float    = 20.0   # Używane przez: Risk-Aware A* (proporcjonalne do kąta)

# Heurystyki (tie-breakery) – jawnie rozdzielone dla każdego algorytmu
HEURISTIC_MULT_ASTAR: float = 1.001  # A* Standard: minimalne wzmocnienie (prawie Dijkstra)
HEURISTIC_MULT_RISK:  float = 1.001    # Risk-Aware A*: silniejsze wzmocnienie (przy W=20)
#   UWAGA: Poprzednio Risk A* używał min(2.5, 1.0 + risk_weight*0.05),
#          co dawało RÓŻNĄ heurystykę przy każdym W → niesprawiedliwe porównanie.
#          Teraz STAŁA wartość → uczciwy benchmark.

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────
N_TESTS: int = 5                # Liczba prób Monte Carlo (min. 20-30 dla istotności stat.)

# Wagi ryzyka używane w sweep Pareto (od 0 do 50, co 5)
PARETO_WEIGHT_STEP: int = 5
PARETO_WEIGHT_MAX:  int = 50

# ─────────────────────────────────────────────────────────────────────────────
# WIZUALIZACJA
# ─────────────────────────────────────────────────────────────────────────────
COLORBAR_V_MAX: float = V_MAX_MS     # Górna granica paska prędkości (zawsze = V_MAX_MS)

# ─────────────────────────────────────────────────────────────────────────────
# PLANOWANIE KINEMATYCZNE
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_RANGE: int = 60               # Zasięg sensora drona [kratki] – lookahead dla planera

# Dynamiczny margines kolizji przy prędkości:
# effective_radius = COLLISION_RADIUS + node_speed² / (2 * ACCELERATION)
# Dotyczy TYLKO twardych przeszkód (budynki, dist_matrix).
# Strefy ryzyka (grid < 0.90) są nadal przelatywalne przy każdej prędkości.
DYNAMIC_COLLISION: bool = True       # Włącz/wyłącz dynamiczny margines
import numpy as np
import random
from typing import List, Tuple
from scipy.ndimage import distance_transform_edt


class GridMap:
    def __init__(self, width: int, height: int, risk_zones_count: int = 5, obstacle_density: float = 0.15) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.float64)

        # Macierz odległości (do kolizji ze statycznymi budynkami)
        self.dist_matrix = np.zeros((width, height), dtype=np.float64)

        # Lista przechowująca dynamiczne zagrożenia: (x, y, radius)
        self.dynamic_obstacles: List[Tuple[int, int, int]] = []

        self._generate_urban_layout(obstacle_density)

    def _generate_urban_layout(self, density: float) -> None:
        total_pixels = self.width * self.height
        target_pixels = int(total_pixels * density)
        current_pixels = 0
        attempts = 0

        # Pozycje Startu i Celu (muszą być zgrane z main.py)
        start_pos = (5, 5)
        goal_pos = (95, 95)

        # Wymagany odstęp: 3 metry wymogu + 1 metr zapasu = 4.0
        SAFE_MARGIN = 4.0

        while current_pixels < target_pixels and attempts < 20000:
            attempts += 1
            w = random.randint(8, 25)
            h = random.randint(8, 25)
            x = random.randint(1, self.width - w - 1)
            y = random.randint(1, self.height - h - 1)

            # --- OCHRONA STARTU I CELU ---
            # Sprawdzamy dystans od Startu do najbliższego punktu planowanego budynku
            closest_x_s = max(x, min(start_pos[0], x + w))
            closest_y_s = max(y, min(start_pos[1], y + h))
            dist_start = np.sqrt((start_pos[0] - closest_x_s) ** 2 + (start_pos[1] - closest_y_s) ** 2)

            # To samo dla Celu
            closest_x_g = max(x, min(goal_pos[0], x + w))
            closest_y_g = max(y, min(goal_pos[1], y + h))
            dist_goal = np.sqrt((goal_pos[0] - closest_x_g) ** 2 + (goal_pos[1] - closest_y_g) ** 2)

            # Jeśli budynek wchodzi na start lub metę -> anuluj ten budynek
            if dist_start < SAFE_MARGIN or dist_goal < SAFE_MARGIN:
                continue
            # -----------------------------

            region = self.grid[x:x + w, y:y + h]
            if np.any(region == 1.0):
                if random.random() > 0.3:
                    continue

            self.grid[x:x + w, y:y + h] = 1.0
            current_pixels += w * h

        # Generowanie mapy kosztów statycznych
        walls = (self.grid == 1.0).astype(float)
        inverted_grid = 1.0 - walls
        self.dist_matrix = distance_transform_edt(inverted_grid)

        risk_gradient = np.exp(-self.dist_matrix / 3.0)
        risk_gradient = np.clip(risk_gradient, 0.0, 0.99)
        risk_gradient[self.dist_matrix > 6] = 0.0

        self.grid = np.maximum(self.grid, risk_gradient)
        self.grid[walls == 1.0] = 1.0

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[x, y])
        return 1.0

    def is_collision(self, x, y, drone_radius: float = 3.0) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True

        # 1. Kolizja Statyczna (Budynki) + POPRAWKA NA SKOSY
        # Mnożnik 1.41 (sqrt(2)) zapewnia, że dystans 3.0m jest liczony
        # jako 3 pełne kratki nawet po przekątnej narożnika.
        if self.dist_matrix[x,y] <= (drone_radius*1.41):
            return True

        # 2. Kolizja Logiczna (Wartość ryzyka na mapie)
        # Gwarantuje, że dron nigdy nie wejdzie w pole o ryzyku >= 0.9 (np. ogień)
        if self.grid[x, y] >= 0.90:
            return True

        # 2. Kolizja Dynamiczna (Strefy Ryzyka)
        # Margines bezpieczeństwa: 1 metr (żeby nie "szorował" po ogniu)
        for (ox, oy, r) in self.dynamic_obstacles:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            # Wymagany dystans = Promień przeszkody + Promień drona + Bufor
            required_dist = r + drone_radius

            if dist <= required_dist:
                return True

        return False

    def add_dynamic_risk_zone(self, cx: int, cy: int, radius: int = 10) -> None:
        """
        Dodaje strefę logicznie (do listy) i wizualnie (na mapę kosztów).
        """
        self.dynamic_obstacles.append((cx, cy, radius))

        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    if self.grid[x, y] < 1.0:
                        self.grid[x, y] = 0.95
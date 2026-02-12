import numpy as np
import random
from typing import Tuple
from scipy.ndimage import distance_transform_edt


class GridMap:
    def __init__(self, width: int, height: int, risk_zones_count: int = 5, obstacle_density: float = 0.15) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.float64)

        # Macierz odległości od najbliższej ściany (kluczowe dla kolizji)
        self.dist_matrix = np.zeros((width, height), dtype=np.float64)

        self._generate_urban_layout(obstacle_density)

    def _generate_urban_layout(self, density: float) -> None:
        total_pixels = self.width * self.height
        target_pixels = int(total_pixels * density)
        current_pixels = 0
        attempts = 0

        # --- ETAP 1: Budynki ---
        while current_pixels < target_pixels and attempts < 20000:
            attempts += 1
            w = random.randint(8, 25)
            h = random.randint(8, 25)
            x = random.randint(1, self.width - w - 1)
            y = random.randint(1, self.height - h - 1)

            # Ochrona startu i celu
            dist_start = np.sqrt((x - 5) ** 2 + (y - 5) ** 2)
            dist_goal = np.sqrt((x - 95) ** 2 + (y - 95) ** 2)
            if dist_start < 15 or dist_goal < 15:
                continue

            region = self.grid[x:x + w, y:y + h]
            if np.any(region == 1.0):
                if random.random() > 0.3:
                    continue

            self.grid[x:x + w, y:y + h] = 1.0
            current_pixels += w * h

        # --- ETAP 2: Obliczanie marginesów i ryzyka ---
        walls = (self.grid == 1.0).astype(float)
        inverted_grid = 1.0 - walls

        # Obliczamy odległość każdego punktu od najbliższej ściany
        self.dist_matrix = distance_transform_edt(inverted_grid)

        # Generowanie aury ryzyka (zanika po 6 kratkach)
        risk_gradient = np.exp(-self.dist_matrix / 3.0)
        risk_gradient = np.clip(risk_gradient, 0.0, 0.99)
        risk_gradient[self.dist_matrix > 6] = 0.0

        self.grid = np.maximum(self.grid, risk_gradient)
        self.grid[walls == 1.0] = 1.0

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[x, y])
        return 1.0

    def is_collision(self, x: int, y: int, drone_radius: float = 2.0) -> bool:
        """
        Sprawdza czy dron może wlecieć w dany punkt.
        Zwraca True (KOLIZJA) jeśli:
        1. Jest za blisko ściany budynku.
        2. Znajduje się wewnątrz strefy wysokiego ryzyka (czerwona plama).
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True

        # 1. Kolizja geometryczna (ściany statyczne)
        if self.dist_matrix[x, y] <= drone_radius:
            return True

        # 2. Kolizja logiczna (NOWOŚĆ - naprawia przelatywanie przez strefę)
        # Jeśli pole jest "bardzo czerwone" (ryzyko >= 0.9), traktujemy je jak ścianę.
        # W metodzie add_dynamic_risk_zone ustawiamy wartość na 0.95, więc to zadziała.
        if self.grid[x, y] >= 0.90:
            return True

        return False

        # ... (reszta metod bez zmian)

    def add_dynamic_risk_zone(self, cx: int, cy: int, radius: int = 10) -> None:
        """
        H3: Dynamiczne dodanie strefy ryzyka w trakcie misji.
        Tworzy okrągły obszar wysokiego ryzyka w punkcie (cx, cy).
        """
        # Iterujemy tylko po kwadracie wokół kliknięcia dla wydajności
        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Równanie okręgu
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    # Dodajemy ryzyko = 0.95 (bardzo wysokie, ale nie ściana)
                    # Dzięki temu dron ominie to, chyba że nie ma wyjścia.
                    # Jeśli była tam ściana (1.0), zostawiamy ścianę.
                    if self.grid[x, y] < 1.0:
                        self.grid[x, y] = 0.95

        # UWAGA: W idealnym świecie powinniśmy przeliczyć self.dist_matrix,
        # ale dla celów prototypu magisterskiego wystarczy nadpisanie siatki kosztów (self.grid),
        # ponieważ A* korzysta głównie z grid.get_cost().

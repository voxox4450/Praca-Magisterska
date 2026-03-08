import numpy as np
import random
from typing import List, Tuple
from scipy.ndimage import distance_transform_edt


class GridMap:
    def __init__(self, width: int, height: int,
                 start_pos: Tuple[int, int],
                 goal_pos: Tuple[int, int],
                 risk_zones_count: int = 5,
                 obstacle_density: float = 0.15) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.float64)
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # Macierz odległości (do kolizji ze statycznymi budynkami)
        self.dist_matrix = np.zeros((width, height), dtype=np.float64)

        # Lista przechowująca dynamiczne zagrożenia: (x, y, radius)
        self.dynamic_obstacles: List[Tuple[int, int, int]] = []

        # 1. Najpierw generujemy budynki (twarde przeszkody)
        self._generate_urban_layout(obstacle_density, start_pos, goal_pos)

        # 2. Następnie "rozsypujemy" miękkie strefy ryzyka
        self._generate_soft_risk_zones(risk_zones_count, start_pos, goal_pos)

    def _generate_urban_layout(self, density: float, start_pos, goal_pos) -> None:
        total_pixels = self.width * self.height
        target_pixels = int(total_pixels * density)
        current_pixels = 0
        attempts = 0

        # Wymagany odstęp: 3 metry wymogu + 1 metr zapasu = 4.0
        SAFE_MARGIN = 4.0

        while current_pixels < target_pixels and attempts < 20000:
            attempts += 1
            w = random.randint(15, 60)
            h = random.randint(15, 40)
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

        risk_gradient = np.exp(-self.dist_matrix / 6.0)
        risk_gradient = np.clip(risk_gradient, 0.0, 0.99)
        risk_gradient[self.dist_matrix > 15] = 0.0

        self.grid = np.maximum(self.grid, risk_gradient)
        self.grid[walls == 1.0] = 1.0

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[x, y])
        return 1.0

    def is_collision(self, x, y, drone_radius: float = 3.0) -> bool:
        # 1. Zabezpieczenie krawędzi mapy (Tylko FIZYCZNY promień drona)
        # Wiemy, że fizyczny promień to 1.0 (margin = 1 kratka).
        # Dron może lecieć blisko krawędzi mapy, byle z niej nie wyleciał.
        physical_margin = 1
        if not (physical_margin <= x < self.width - physical_margin and physical_margin <= y < self.height - physical_margin):
            return True

        # 2. Kolizja Statyczna (Budynki) + POPRAWKA NA SKOSY
        # Tutaj używamy pełnego drone_radius (czyli 3.0: 1m drona + 2m marginesu od ścian)
        # Mnożnik 1.41 (sqrt(2)) zapewnia, że dystans 3.0m jest liczony
        # jako 3 pełne kratki nawet po przekątnej narożnika.
        if self.dist_matrix[x, y] <= (drone_radius * 1.41):
            return True

        # 3. Kolizja Logiczna (Wartość ryzyka na mapie)
        if self.grid[x, y] >= 0.90:
            return True

        # 4. Kolizja Dynamiczna (Strefy Ryzyka)
        for (ox, oy, r) in self.dynamic_obstacles:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            required_dist = r + drone_radius

            if dist <= required_dist:
                return True

        return False

    def add_dynamic_risk_zone(self, cx: int, cy: int, radius: int = 10) -> None:
        """
        Dodaje strefę logicznie (do listy) i wizualnie (na mapę kosztów),
        wraz z szerokim gradientem ostrzegawczym.
        """
        self.dynamic_obstacles.append((cx, cy, radius))

        # Zasięg gradientu (taki sam jak dla budynków)
        gradient_range = 15
        total_radius = radius + gradient_range

        x_min = max(0, cx - total_radius)
        x_max = min(self.width, cx + total_radius)
        y_min = max(0, cy - total_radius)
        y_max = min(self.height, cy + total_radius)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                if dist <= radius:
                    # Rdzeń przeszkody (Czysta Czerwień - Ryzyko 0.95)
                    if self.grid[x, y] < 1.0:
                        self.grid[x, y] = 0.95
                elif dist <= total_radius:
                    # --- GRADIENT DLA PRZESZKODY DYNAMICZNEJ ---
                    # Liczymy odległość od krawędzi rdzenia przeszkody
                    dist_from_edge = dist - radius

                    # Używamy tej samej matematyki co dla budynków
                    risk_val = np.exp(-dist_from_edge / 6.0)
                    risk_val = np.clip(risk_val, 0.0, 0.94)  # Zabezpieczenie, żeby nie przekroczyć wartości rdzenia

                    if self.grid[x, y] < 1.0:
                        # Bierzemy "max", żeby gradienty się ładnie nakładały (np. gdy przeszkoda jest blisko budynku)
                        self.grid[x, y] = max(self.grid[x, y], risk_val)

    def _generate_soft_risk_zones(self, num_zones: int, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> None:
        """
        Generuje półprzezroczyste strefy ryzyka o różnych kształtach (koło, kwadrat, prostokąt).
        Weryfikuje, czy strefa nie pokrywa się z budynkami (grid == 1.0).
        """
        import math

        zones_placed = 0
        attempts = 0

        # Pętla działa do momentu rozstawienia wszystkich stref lub wyczerpania limitu prób
        while zones_placed < num_zones and attempts < 2000:
            attempts += 1

            shape = random.choice(['circle', 'square', 'rectangle'])
            peak_risk = random.uniform(0.2, 0.7)  # Maksymalne ryzyko w samym centrum strefy

            # 1. Definiowanie wymiarów i obszaru brzegowego (Bounding Box)
            if shape == 'circle':
                r = random.randint(15, 30)
                cx = random.randint(r, self.width - r - 1)
                cy = random.randint(r, self.height - r - 1)
                x_min, x_max = cx - r, cx + r
                y_min, y_max = cy - r, cy + r
            elif shape == 'square':
                s = random.randint(20, 45)
                x_min = random.randint(0, self.width - s - 1)
                y_min = random.randint(0, self.height - s - 1)
                x_max, y_max = x_min + s, y_min + s
            else:  # rectangle
                w = random.randint(20, 50)
                h = random.randint(15, 35)
                # Losowo obracamy prostokąt (pion/poziom)
                if random.random() > 0.5:
                    w, h = h, w
                x_min = random.randint(0, self.width - w - 1)
                y_min = random.randint(0, self.height - h - 1)
                x_max, y_max = x_min + w, y_min + h

            # 2. WERYFIKACJA KOLIZJI Z BUDYNKAMI
            # Pobieramy cały wycinek mapy, na którym chcemy położyć strefę.
            # Jeśli w tym wycinku jest jakikolwiek piksel oznaczający budynek (1.0) - przerywamy!
            region = self.grid[x_min:x_max, y_min:y_max]
            if np.any(region == 1.0):
                continue

            # 3. Ochrona startu i celu (zabezpieczenie na wypadek zablokowania bazy)
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            max_r = math.sqrt(((x_max - x_min) / 2.0) ** 2 + ((y_max - y_min) / 2.0) ** 2)

            dist_start = np.sqrt((start_pos[0] - center_x) ** 2 + (start_pos[1] - center_y) ** 2)
            dist_goal = np.sqrt((goal_pos[0] - center_x) ** 2 + (goal_pos[1] - center_y) ** 2)

            if dist_start < max_r + 5 or dist_goal < max_r + 5:
                continue

            # 4. APLIKACJA GRADIENTÓW (Zależna od kształtu)
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):

                    if shape == 'circle':
                        # Gradient kołowy (radialny)
                        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dist <= r:
                            risk_val = peak_risk * (1.0 - (dist / r))
                            self.grid[x, y] = min(0.85, self.grid[x, y] + risk_val)

                    else:  # Kwadrat i prostokąt
                        # Gradient piramidalny (odległość do najbliższej krawędzi)
                        dx = min(x - x_min, x_max - x - 1)
                        dy = min(y - y_min, y_max - y - 1)
                        dist_to_edge = min(dx, dy)

                        max_dist = min((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)

                        if max_dist > 0:
                            # 0 na krawędzi kształtu, peak_risk w samym jego centrum
                            risk_val = peak_risk * (dist_to_edge / max_dist)
                            self.grid[x, y] = min(0.85, self.grid[x, y] + risk_val)

            # Jeśli algorytm tu dotarł, to strefa została pomyślnie nałożona!
            zones_placed += 1
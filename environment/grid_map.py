import numpy as np
import random
from typing import Tuple


class GridMap:
    def __init__(self, width: int, height: int, risk_zones_count: int = 5, obstacle_density: float = 0.15) -> None:
        self.width = width
        self.height = height

        # 1. TŁO: Zaczynamy od CZYSTEGO BIAŁEGO (0.0 - bezpieczna ulica/powietrze)
        self.grid = np.zeros((width, height), dtype=np.float64)

        # 2. BUDYNKI: Stawiamy kamienice
        self._generate_urban_layout(obstacle_density)

    def _generate_urban_layout(self, density: float) -> None:
        """
        Generuje miasto poprzez stawianie budynków na pustym terenie.
        Gwarantuje przejezdność (ulice) i tworzy 'aurę' ryzyka (chodniki).
        """
        total_pixels = self.width * self.height
        target_pixels = int(total_pixels * density)
        current_pixels = 0
        attempts = 0

        # Lista budynków do późniejszego wygenerowania chodników
        buildings = []

        # --- ETAP 1: STAWIANIE KAMIENIC ---
        while current_pixels < target_pixels and attempts < 20000:
            attempts += 1

            # Losujemy wymiary kamienicy (nieregularne kształty)
            w = random.randint(8, 25)
            h = random.randint(8, 25)
            x = random.randint(1, self.width - w - 1)
            y = random.randint(1, self.height - h - 1)

            # STREFA OCHRONNA DLA STARTU I METY
            # Nie stawiamy budynku blisko (5,5) ani (95,95)
            # Dystans Euklidesowy do startu i mety
            dist_start = np.sqrt((x - 5) ** 2 + (y - 5) ** 2)
            dist_goal = np.sqrt((x - 95) ** 2 + (y - 95) ** 2)

            if dist_start < 15 or dist_goal < 15:
                continue

            # Sprawdzamy, czy nie nachodzi za bardzo na inne budynki (żeby zachować ulice)
            # Wycinek mapy gdzie chcemy postawić budynek
            region = self.grid[x:x + w, y:y + h]
            if np.any(region == 1.0):
                # Jeśli już tu coś stoi, to z dużą szansą odpuszczamy (zachowanie odstępów/ulic)
                # Ale czasem pozwalamy na 'przyklejenie' się (pierzeja ulicy)
                if random.random() > 0.3:
                    continue

            # Stawiamy budynek (ŚCIANA = 1.0)
            self.grid[x:x + w, y:y + h] = 1.0
            buildings.append((x, y, w, h))

            # Aktualizacja licznika (zgrubna)
            current_pixels += w * h

        # --- ETAP 2: GENEROWANIE CHODNIKÓW I RYZYKA (GRADIENT) ---
        # Iterujemy, żeby rozmyć granice budynków.
        # To stworzy czerwoną aurę wokół czarnych prostokątów.

        risk_layer = np.zeros_like(self.grid)

        # Promień oddziaływania ryzyka (szerokość chodnika/strefy zrzutu)
        # Im większy, tym szersze czerwone pole.
        spread_radius = 6

        # Używamy distance_transform (lub symulacji) dla wydajności
        # Tutaj ręczna symulacja propagacji ryzyka:

        # Kopiujemy same ściany
        walls = (self.grid == 1.0).astype(float)

        # Algorytm 'rozlewania' ryzyka
        from scipy.ndimage import distance_transform_edt

        # Obliczamy odległość każdego piksela od najbliższej ściany
        # distance_transform_edt liczy dystans do ZERA, więc odwracamy logikę (ściany to 0 w obliczeniach)
        inverted_grid = 1.0 - walls
        dist_matrix = distance_transform_edt(inverted_grid)

        # Tworzymy gradient
        # Ryzyko = 1.0 przy ścianie, maleje do 0.0 w odległości 'spread_radius'
        # Formuła: exp(-dist) daje ładny, miękki spadek
        risk_gradient = np.exp(-dist_matrix / 3.0)

        # Normalizacja i przycięcie
        # Chcemy, żeby tuż przy ścianie było np. 0.9, a 5 kratek dalej 0.1
        risk_gradient = np.clip(risk_gradient, 0.0, 0.99)

        # Tam gdzie dystans jest duży (środek ulicy), zerujemy ryzyko do idealnej bieli
        risk_gradient[dist_matrix > spread_radius] = 0.0

        # --- ETAP 3: ŁĄCZENIE ---
        # Nakładamy gradient na mapę
        self.grid = np.maximum(self.grid, risk_gradient)

        # Przywracamy ściany jako idealne 1.0 (bo gradient mógł je lekko zmienić)
        self.grid[walls == 1.0] = 1.0

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[x, y])
        return 1.0
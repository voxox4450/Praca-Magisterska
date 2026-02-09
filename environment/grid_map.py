import numpy as np
import random
from typing import Tuple


class GridMap:
    def __init__(self, width: int, height: int, risk_zones_count: int = 5, obstacle_density: float = 0.15) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.float64)

        self._generate_obstacles(obstacle_density)
        self._generate_risk_zones(risk_zones_count)

    def _generate_obstacles(self, density: float) -> None:
        """Generuje czarne prostokąty (budynki)."""
        total_pixels = self.width * self.height
        target_obstacle_pixels = int(total_pixels * density)
        current_obstacle_pixels = 0
        attempts = 0

        while current_obstacle_pixels < target_obstacle_pixels and attempts < 5000:
            w = random.randint(5, 20)
            h = random.randint(5, 20)
            x = random.randint(0, self.width - w)
            y = random.randint(0, self.height - h)

            self.grid[x:x + w, y:y + h] = 1.0
            current_obstacle_pixels = np.count_nonzero(self.grid == 1.0)
            attempts += 1

    def _generate_risk_zones(self, count: int) -> None:
        """
        Generuje strefy ryzyka w kształcie prostokątów (chodniki, ulice).
        Ryzyko jest największe w środku prostokąta i zanika ku krawędziom.
        """
        # Zwiększamy liczbę stref, bo chodniki są węższe niż wielkie koła
        effective_count = count * 2

        for _ in range(effective_count):
            # Losujemy wymiary - chcemy więcej podłużnych kształtów (chodniki)
            if random.random() > 0.3:
                # Długi i wąski (chodnik)
                if random.random() > 0.5:  # Poziomy
                    w = random.randint(20, 60)
                    h = random.randint(3, 8)
                else:  # Pionowy
                    w = random.randint(3, 8)
                    h = random.randint(20, 60)
            else:
                # Kwadratowy (plac, rynek)
                w = random.randint(10, 25)
                h = random.randint(10, 25)

            # Pozycja środka
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height)

            intensity = random.uniform(0.5, 0.95)  # Max ryzyko w centrum (np. 0.95)

            # Obliczanie gradientu prostokątnego
            y_indices, x_indices = np.ogrid[:self.width, :self.height]

            # Odległość znormalizowana od środka (0 w środku, 1 na krawędzi prostokąta)
            # Używamy abs() aby zrobić gradient liniowy
            # dx = odległość x od środka / połowa szerokości
            dx = np.abs(x_indices - cx) / (w / 2)
            dy = np.abs(y_indices - cy) / (h / 2)

            # Bierzemy maximum z dx i dy - to tworzy kształt prostokąta (metryka Czebyszewa)
            dist_normalized = np.maximum(dx, dy)

            # Maska: bierzemy tylko to co jest wewnątrz prostokąta (dist <= 1.0)
            mask = dist_normalized <= 1.0

            # Formuła ryzyka: Im dalej od środka, tym mniejsze
            # risk = intensity * (1 - dist)
            risk_values = intensity * (1.0 - dist_normalized[mask])

            # Nakładanie na mapę
            current_values = self.grid[mask]

            # Ważne: Bierzemy MAX, żeby chodniki się łączyły, a nie nadpisywały
            new_values = np.maximum(current_values, risk_values)

            # Nie przekraczamy 0.99 (bo 1.0 to ściana)
            new_values = np.minimum(new_values, 0.99)

            # Jeśli w tym miejscu jest już ściana (1.0), zostawiamy ścianę
            is_wall = self.grid[mask] == 1.0
            final_values = np.where(is_wall, 1.0, new_values)

            self.grid[mask] = final_values

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[x, y])
        return 1.0
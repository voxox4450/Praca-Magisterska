import numpy as np
import random


class GridMap:
    def __init__(self, width: int, height: int, risk_zones_count: int =5) -> None:
        """
        Inicjalizacja mapy.
        :param width: Szerokość siatki (np. 100, 150, 200)
        :param height: Wysokość siatki
        :param risk_zones_count: Ile stref ryzyka wygenerować
        """
        self.width = width
        self.height = height
        # Inicjalizacja siatki zerami (bezpieczna strefa)
        # Typ float64, zakres 0.0 do 1.0
        self.grid = np.zeros((width, height), dtype=np.float64)

        self._generate_obstacles()
        self._generate_risk_zones(risk_zones_count)

    def _generate_obstacles(self) -> None:
        """
        Generuje twarde przeszkody (wartość 1.0 - zakaz lotu).
        Symulacja budynków jako prostokątów.
        """
        num_obstacles = int((self.width * self.height) * 0.05)  # 5% mapy to budynki

        for _ in range(20):  # Np. 20 budynków
            w = random.randint(5, 15)
            h = random.randint(5, 15)
            x = random.randint(0, self.width - w)
            y = random.randint(0, self.height - h)

            # Ustawienie 1.0 oznacza fizyczną przeszkodę / No-Fly Zone
            self.grid[x:x + w, y:y + h] = 1.0

    def _generate_risk_zones(self, count: int) -> None:
        """
        Generuje strefy ryzyka (wartości od 0.1 do 0.9).
        Symulacja zakłóceń sygnału (gradient ryzyka).
        """
        for _ in range(count):
            # Losowe centrum strefy ryzyka
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height)
            radius = random.randint(10, 30)
            intensity = random.uniform(0.3, 0.8)  # Maksymalne ryzyko w centrum strefy

            # Tworzymy gradient ryzyka wokół centrum
            y, x = np.ogrid[:self.width, :self.height]
            dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Maska dla okręgu
            mask = dist_from_center <= radius

            # Funkcja ryzyka: im bliżej centrum, tym wyższe ryzyko
            # risk = intensity * (1 - dystans/promień)
            risk_values = intensity * (1 - dist_from_center[mask] / radius)

            # Dodajemy ryzyko do siatki (uważając, żeby nie nadpisać budynków 1.0)
            # Używamy np.maximum, aby zachować najwyższe ryzyko jeśli strefy się nakładają
            current_values = self.grid[mask]
            # Nie chcemy nadpisać przeszkód (1.0), więc modyfikujemy tylko tam, gdzie < 1.0
            new_values = np.maximum(current_values, risk_values)

            # Zabezpieczenie przed przekroczeniem 1.0 (chyba że to budynek)
            new_values = np.minimum(new_values, 0.99)  # Ryzyko max 0.99, 1.0 to ściana

            # Aplikujemy zmiany, ale nie ruszamy miejsc gdzie już jest ściana (1.0)
            # Logika: jeśli było 1.0, zostaje 1.0. Jeśli było mniej, bierzemy max(stare, nowe)
            is_wall = self.grid[mask] == 1.0
            final_values = np.where(is_wall, 1.0, new_values)

            self.grid[mask] = final_values

    def get_cost(self, x: int, y: int) -> float:
        """Zwraca koszt (ryzyko) danej komórki."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x, y]
        return 1.0  # Poza mapą jest ściana
import numpy as np
import random
from typing import List, Tuple
from scipy.ndimage import distance_transform_edt
from config import (
    BUILDING_SAFE_MARGIN, GRADIENT_RANGE, GRADIENT_DECAY,
    COLLISION_RADIUS
)


class GridMap:
    def __init__(self, width: int, height: int,
                 start_pos: Tuple[int, int],
                 goal_pos: Tuple[int, int],
                 risk_zones_count: int = 10,
                 obstacle_density: float = 0.15) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.float64)
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.dist_matrix = np.zeros((width, height), dtype=np.float64)
        self.dynamic_obstacles: List[Tuple[int, int, int]] = []

        self._generate_urban_layout(obstacle_density, start_pos, goal_pos)
        self._generate_soft_risk_zones(risk_zones_count, start_pos, goal_pos)

    def _generate_urban_layout(self, density: float, start_pos, goal_pos) -> None:
        total_pixels = self.width * self.height
        target_pixels = int(total_pixels * density)
        current_pixels = 0
        attempts = 0

        while current_pixels < target_pixels and attempts < 20000:
            attempts += 1
            w = random.randint(15, 60)
            h = random.randint(15, 40)
            x = random.randint(1, self.width - w - 1)
            y = random.randint(1, self.height - h - 1)

            closest_x_s = max(x, min(start_pos[0], x + w))
            closest_y_s = max(y, min(start_pos[1], y + h))
            dist_start = np.sqrt((start_pos[0] - closest_x_s) ** 2 + (start_pos[1] - closest_y_s) ** 2)

            closest_x_g = max(x, min(goal_pos[0], x + w))
            closest_y_g = max(y, min(goal_pos[1], y + h))
            dist_goal = np.sqrt((goal_pos[0] - closest_x_g) ** 2 + (goal_pos[1] - closest_y_g) ** 2)

            if dist_start < BUILDING_SAFE_MARGIN or dist_goal < BUILDING_SAFE_MARGIN:
                continue

            region = self.grid[x:x + w, y:y + h]
            if np.any(region == 1.0):
                if random.random() > 0.3:
                    continue

            self.grid[x:x + w, y:y + h] = 1.0
            current_pixels += w * h

        walls = (self.grid == 1.0).astype(float)
        inverted_grid = 1.0 - walls
        self.dist_matrix = distance_transform_edt(inverted_grid)

        risk_gradient = np.exp(-self.dist_matrix / GRADIENT_DECAY)
        risk_gradient = np.clip(risk_gradient, 0.0, 0.99)
        risk_gradient[self.dist_matrix > GRADIENT_RANGE] = 0.0

        self.grid = np.maximum(self.grid, risk_gradient)
        self.grid[walls == 1.0] = 1.0

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[x, y])
        return 1.0

    def is_collision(self, x, y, drone_radius: float = COLLISION_RADIUS) -> bool:
        physical_margin = 1
        if not (physical_margin <= x < self.width - physical_margin
                and physical_margin <= y < self.height - physical_margin):
            return True

        # Sprawdzenie odległości od budynków (z korektą na skosy: *1.41)
        if self.dist_matrix[x, y] <= (drone_radius * 1.41):
            return True

        if self.grid[x, y] >= 0.90:
            return True

        for (ox, oy, r) in self.dynamic_obstacles:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if dist <= (r + drone_radius):
                return True

        return False

    def add_dynamic_risk_zone(self, cx: int, cy: int, radius: int = 10) -> None:
        """
        Dodaje strefę dynamiczną logicznie (do listy) i wizualnie (gradient na mapie).
        Parametry gradientu pobierane z config.
        """
        self.dynamic_obstacles.append((cx, cy, radius))

        total_radius = radius + GRADIENT_RANGE

        x_min = max(0, cx - total_radius)
        x_max = min(self.width, cx + total_radius)
        y_min = max(0, cy - total_radius)
        y_max = min(self.height, cy + total_radius)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                if dist <= radius:
                    if self.grid[x, y] < 1.0:
                        self.grid[x, y] = 0.95
                elif dist <= total_radius:
                    dist_from_edge = dist - radius
                    risk_val = np.exp(-dist_from_edge / GRADIENT_DECAY)
                    risk_val = np.clip(risk_val, 0.0, 0.94)
                    if self.grid[x, y] < 1.0:
                        self.grid[x, y] = max(self.grid[x, y], risk_val)

    def _generate_soft_risk_zones(self, num_zones: int, start_pos: Tuple[int, int],
                                   goal_pos: Tuple[int, int]) -> None:
        """
        Generuje półprzezroczyste strefy ryzyka o różnych kształtach.
        Nie pokrywają się z budynkami ani ze startem/celem.
        """
        import math

        zones_placed = 0
        attempts = 0

        while zones_placed < num_zones and attempts < 2000:
            attempts += 1

            shape = random.choice(['circle', 'square', 'rectangle'])
            peak_risk = random.uniform(0.2, 0.7)

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
                if random.random() > 0.5:
                    w, h = h, w
                x_min = random.randint(0, self.width - w - 1)
                y_min = random.randint(0, self.height - h - 1)
                x_max, y_max = x_min + w, y_min + h

            region = self.grid[x_min:x_max, y_min:y_max]
            if np.any(region == 1.0):
                continue

            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            max_r = math.sqrt(((x_max - x_min) / 2.0) ** 2 + ((y_max - y_min) / 2.0) ** 2)

            dist_start = np.sqrt((start_pos[0] - center_x) ** 2 + (start_pos[1] - center_y) ** 2)
            dist_goal = np.sqrt((goal_pos[0] - center_x) ** 2 + (goal_pos[1] - center_y) ** 2)

            if dist_start < max_r + 5 or dist_goal < max_r + 5:
                continue

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    if shape == 'circle':
                        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dist <= r:
                            risk_val = peak_risk * (1.0 - (dist / r))
                            self.grid[x, y] = min(0.85, self.grid[x, y] + risk_val)
                    else:
                        dx = min(x - x_min, x_max - x - 1)
                        dy = min(y - y_min, y_max - y - 1)
                        dist_to_edge = min(dx, dy)
                        max_dist = min((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)
                        if max_dist > 0:
                            risk_val = peak_risk * (dist_to_edge / max_dist)
                            self.grid[x, y] = min(0.85, self.grid[x, y] + risk_val)

            zones_placed += 1
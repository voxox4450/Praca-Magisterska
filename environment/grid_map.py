import numpy as np
import random
from typing import List, Tuple
from scipy.ndimage import distance_transform_edt, convolve
from config import (
    BUILDING_SAFE_MARGIN, GRADIENT_RANGE, GRADIENT_DECAY,
    COLLISION_RADIUS, COLLISION_GRID_THRESHOLD, SOFT_RISK_CAP
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

        self.current_phys_r = 1.0
        self.current_col_r = 3.0

        self.collision_mask = np.ones((width, height), dtype=bool)
        self.risk_grid = np.zeros((width, height), dtype=np.float64)  # DODANE C-Space ryzyka

        self._generate_urban_layout(obstacle_density, start_pos, goal_pos)
        self._generate_soft_risk_zones(risk_zones_count, start_pos, goal_pos)

        self.update_drone_footprint(self.current_phys_r, self.current_col_r)

    def _recompute_dist_matrix(self) -> None:
        walls = (self.grid == 1.0).astype(float)
        inverted_grid = 1.0 - walls
        self.dist_matrix = distance_transform_edt(inverted_grid)

    def update_drone_footprint(self, physical_radius: float, collision_radius: float) -> None:
        self.current_phys_r = physical_radius
        self.current_col_r = collision_radius

        w, h = self.width, self.height
        physical_margin = max(1, int(np.ceil(physical_radius)))

        mask = np.ones((w, h), dtype=bool)
        interior = np.zeros((w, h), dtype=bool)
        interior[physical_margin:w - physical_margin, physical_margin:h - physical_margin] = True

        safe = (
                interior
                & (self.dist_matrix > collision_radius)
                & (self.grid < COLLISION_GRID_THRESHOLD)
        )
        mask[safe] = False

        for (ox, oy, r) in self.dynamic_obstacles:
            x_min = max(0, ox - int(r + collision_radius) - 1)
            x_max = min(w, ox + int(r + collision_radius) + 2)
            y_min = max(0, oy - int(r + collision_radius) - 1)
            y_max = min(h, oy + int(r + collision_radius) + 2)

            xs = np.arange(x_min, x_max)
            ys = np.arange(y_min, y_max)
            xx, yy = np.meshgrid(xs, ys, indexing='ij')
            dist_sq = (xx - ox) ** 2 + (yy - oy) ** 2
            mask[x_min:x_max, y_min:y_max] |= (dist_sq <= (r + collision_radius) ** 2)

        self.collision_mask = mask

        r_int = int(np.ceil(physical_radius))
        if r_int > 0:
            y_idx, x_idx = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
            kernel = (x_idx ** 2 + y_idx ** 2 <= physical_radius ** 2).astype(float)
            kernel /= kernel.sum()  # normalizacja → średnia ważona
            self.risk_grid = convolve(self.grid, kernel, mode='constant', cval=0.0)
        else:
            self.risk_grid = np.copy(self.grid)

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

            new_pixels = int(np.sum(region < 1.0))
            self.grid[x:x + w, y:y + h] = 1.0
            current_pixels += new_pixels

        self._recompute_dist_matrix()

        risk_gradient = np.exp(-self.dist_matrix / GRADIENT_DECAY)
        risk_gradient = np.clip(risk_gradient, 0.0, 0.99)
        risk_gradient[self.dist_matrix > GRADIENT_RANGE] = 0.0

        walls = (self.grid == 1.0)
        self.grid = np.maximum(self.grid, risk_gradient)
        self.grid[walls] = 1.0

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.risk_grid[x, y])
        return 1.0

    def is_collision(self, x, y, drone_radius: float = COLLISION_RADIUS) -> bool:
        physical_margin = max(1, int(np.ceil(self.current_phys_r)))
        if not (physical_margin <= x < self.width - physical_margin
                and physical_margin <= y < self.height - physical_margin):
            return True

        if self.dist_matrix[x, y] <= drone_radius:
            return True

        if self.grid[x, y] >= COLLISION_GRID_THRESHOLD:
            return True

        for (ox, oy, r) in self.dynamic_obstacles:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if dist <= (r + drone_radius):
                return True

        return False

    def add_dynamic_risk_zone(self, cx: int, cy: int, radius: int = 10) -> None:
        self.dynamic_obstacles.append((cx, cy, radius))

        total_radius = radius + GRADIENT_RANGE

        x_min = max(0, cx - total_radius)
        x_max = min(self.width, cx + total_radius)
        y_min = max(0, cy - total_radius)
        y_max = min(self.height, cy + total_radius)

        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        region = self.grid[x_min:x_max, y_min:y_max]
        not_wall = region < 1.0

        core_mask = (dist <= radius) & not_wall
        region[core_mask] = 0.95

        gradient_mask = (dist > radius) & (dist <= total_radius) & not_wall
        dist_from_edge = dist - radius
        risk_val = np.exp(-dist_from_edge / GRADIENT_DECAY)
        risk_val = np.clip(risk_val, 0.0, 0.94)
        region[gradient_mask] = np.maximum(region[gradient_mask], risk_val[gradient_mask])

        self.grid[x_min:x_max, y_min:y_max] = region

        self._recompute_dist_matrix()
        self.update_drone_footprint(self.current_phys_r, self.current_col_r)

    def _generate_soft_risk_zones(self, num_zones: int, start_pos: Tuple[int, int],
                                   goal_pos: Tuple[int, int]) -> None:
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
            else:
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
            #1
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            max_r = math.sqrt(((x_max - x_min) / 2.0) ** 2 + ((y_max - y_min) / 2.0) ** 2)

            dist_start = np.sqrt((start_pos[0] - center_x) ** 2 + (start_pos[1] - center_y) ** 2)
            dist_goal = np.sqrt((goal_pos[0] - center_x) ** 2 + (goal_pos[1] - center_y) ** 2)

            if dist_start < max_r + 5 or dist_goal < max_r + 5:
                continue

            xs = np.arange(x_min, x_max)
            ys = np.arange(y_min, y_max)
            xx, yy = np.meshgrid(xs, ys, indexing='ij')

            if shape == 'circle':
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                inside = dist <= r
                risk_val = peak_risk * (1.0 - (dist / r))
                risk_val = np.clip(risk_val, 0.0, peak_risk)
                new_grid = np.minimum(SOFT_RISK_CAP, region + risk_val)
                region[inside] = new_grid[inside]
            else:
                dx_arr = np.minimum(xx - x_min, x_max - xx - 1)
                dy_arr = np.minimum(yy - y_min, y_max - yy - 1)
                dist_to_edge = np.minimum(dx_arr, dy_arr).astype(float)
                max_dist = min((x_max - x_min) / 2.0, (y_max - y_min) / 2.0)
                if max_dist > 0:
                    risk_val = peak_risk * (dist_to_edge / max_dist)
                    region[:] = np.minimum(SOFT_RISK_CAP, region + risk_val)

            self.grid[x_min:x_max, y_min:y_max] = region
            zones_placed += 1
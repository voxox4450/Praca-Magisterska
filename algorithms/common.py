from typing import Tuple, List, Optional, Callable, Dict, Any
import math
from environment.grid_map import GridMap


class Node:
    def __init__(self, x: int, y: int, cost: float, parent: Optional['Node'] = None,
                 direction: Tuple[int, int] = (0, 0), heuristic: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.direction = direction
        self.heuristic = heuristic

    @property
    def total_cost(self) -> float:
        return self.cost + self.heuristic

    def __lt__(self, other: 'Node') -> bool:
        return self.total_cost < other.total_cost


def reconstruct_path(node: Node, grid_map: GridMap) -> Tuple[List[Tuple[int, int]], float, float, int]:
    """Odtwarza ścieżkę i oblicza metryki."""
    path = []
    total_risk = 0.0
    total_length = 0.0
    turns = 0

    current = node
    last_dir = None

    while current:
        path.append((current.x, current.y))
        val = grid_map.get_cost(current.x, current.y)
        if val < 1.0:
            total_risk += val

        if current.parent:
            dx = current.x - current.parent.x
            dy = current.y - current.parent.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            total_length += dist

            curr_dir = (dx, dy)
            if last_dir is not None and curr_dir != last_dir:
                turns += 1
            last_dir = curr_dir

        current = current.parent

    return path[::-1], total_length, total_risk, turns


def calculate_dynamic_penalty(
        grid_map: GridMap,
        current_direction: Tuple[int, int],
        move_vector: Tuple[int, int],
        current_pos: Tuple[int, int],
        braking_distance: int = 4
) -> float:
    """
    Oblicza karę za pęd (momentum).
    Jeśli dron leci w danym kierunku i kontynuuje ruch (move_vector == current_direction),
    sprawdzamy czy w przyszłości nie ma przeszkody.
    """
    penalty = 0.0

    # Sprawdzamy tylko, jeśli dron ma pęd i go utrzymuje
    if current_direction != (0, 0) and current_direction == move_vector:
        nx, ny = current_pos
        dx, dy = move_vector

        # Patrzymy w przyszłość (symulacja drogi hamowania)
        for step in range(1, braking_distance + 1):
            look_x = nx + (dx * step)
            look_y = ny + (dy * step)

            # Czy punkt jest na mapie?
            if 0 <= look_x < grid_map.width and 0 <= look_y < grid_map.height:
                future_risk = grid_map.get_cost(look_x, look_y)

                # Jeśli wykryjemy ryzyko (czerwone lub czarne)
                if future_risk > 0.4:
                    # Im bliżej przeszkody (mniejszy step), tym kara większa
                    impact = (braking_distance - step + 1)

                    # Kara bazowa za ryzyko
                    penalty += impact * future_risk * 5.0

                    # Jeśli to ściana (1.0), kara jest KRYTYCZNA
                    if future_risk >= 1.0:
                        penalty += 50.0
                        break  # Wiemy że uderzymy, nie trzeba sprawdzać dalej
            else:
                # Wyjście poza mapę traktujemy jak ścianę
                penalty += 50.0
                break

    return penalty


def calculate_segment_risk(path: List[Tuple[int, int]], env: GridMap) -> float:
    """Oblicza całkowite ryzyko na ścieżce."""
    total_risk = 0.0
    for (x, y) in path:
        total_risk += env.get_cost(x, y)
    return total_risk


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    """Oblicza długość ścieżki."""
    length = 0.0
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        length += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length


def generate_analysis_table(
        env: GridMap,
        start_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        search_func: Callable,
        base_len: float,
        base_risk: float,
        collision_radius: float,
        table_title: str = "ANALIZA"
) -> None:
    """
    Wspólna funkcja do generowania tabeli analizy dla różnych wag ryzyka.
    Używana zarówno w trybie offline jak i online.
    """
    risk_weights = [float(x) for x in range(0, 101, 5)]

    print("-" * 90)
    print(f"{table_title}")
    print("-" * 90)
    print(f"Baza: Dystans: {base_len:.2f} | Ryzyko: {base_risk:.2f}")
    print("-" * 90)
    print(f"{'Waga (W)':<10} | {'Dystans':<10} | {'Koszt [%]':<10} | {'Ryzyko':<10} | {'Zmiana Ryzyka [%]':<20}")
    print("-" * 90)

    for w in risk_weights:
        _, stats = search_func(env, start_pos, target_pos, risk_weight=w, turn_penalty=20.0, drone_radius=collision_radius)

        if stats['found'] and base_len > 0:
            len_inc = ((stats['length'] - base_len) / base_len) * 100

            risk_change = 0.0
            if base_risk > 0:
                risk_change = ((stats['risk'] - base_risk) / base_risk) * 100

            print(
                f"{w:<10.1f} | {stats['length']:<10.2f} | +{len_inc:<9.2f} | {stats['risk']:<10.2f} | {risk_change:<+19.2f}")
        else:
            print(f"{w:<10.1f} | BRAK TRASY")

    print("-" * 90)
import heapq
import math
import time
from typing import List, Tuple, Dict, Any, Optional, Set
from environment.grid_map import GridMap

class Node:
    def __init__(self, x:
                 int, y: int, cost: float,
                 parent: Optional['Node'] = None,
                 direction: Tuple[int, int] = (0, 0),
                 heuristic: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost  # G (koszt od startu)
        self.parent = parent
        self.heuristic = 0 # H (szacowany koszt do celu)
        self.direction = direction  # Kierunek ruchu (dx, dy) -
        self.heuristic = heuristic  # H (szacowany koszt do celu)

    @property
    def total_cost(self) -> float:
        return self.cost + self.heuristic  # F = G + H

    def __lt__(self, other: 'Node') -> bool:
        return self.total_cost < other.total_cost


def run_search(grid_map: GridMap,
               start: Tuple[int, int],
               goal: Tuple[int, int],
               algorithm_type: str ="astar",
               risk_weight: float = 0.0,
               turn_penalty: float = 0.0) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Główna funkcja wyszukiwania.
    :param grid_map: Obiekt GridMap
    :param start: krotka (x, y)
    :param goal: krotka (x, y)
    :param algorithm_type: "dijkstra", "astar", "risk_aware"
    :param risk_weight: Waga ryzyka (dla Optimized A*)
    :return: (path, visited_nodes_count, path_cost)
    """
    t0 = time.time()

    start_node = Node(start[0], start[1], 0, direction=(0, 0))
    open_list = []
    heapq.heappush(open_list, start_node)

    # Słownik kosztów: (x, y) -> najniższy koszt G
    g_score = {(start[0], start[1]): 0}
    visited = set()
    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        # Sprawdzenie celu
        if (current.x, current.y) == goal:
            execution_time = time.time() - t0
            path, stats = reconstruct_path_and_stats(current, grid_map)
            stats['time'] = execution_time
            stats['nodes_expanded'] = nodes_expanded
            return path, stats

        if (current.x, current.y) in visited:
            continue
        visited.add((current.x, current.y))

        # Ruchy: 8 kierunków (sąsiedztwo Moore'a)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in neighbors:
            nx, ny = current.x + dx, current.y + dy

            # 1. Sprawdzenie granic i przeszkód twardych (No-Fly Zone)
            if not (0 <= nx < grid_map.width and 0 <= ny < grid_map.height):
                continue

            cell_risk = grid_map.get_cost(nx, ny)
            if cell_risk >= 1.0:  # Fizyczna przeszkoda lub absolutny zakaz
                continue

            # 2. Obliczenie kosztu ruchu (bazowy dystans)
            dist_cost = math.sqrt(dx ** 2 + dy ** 2)

            # 3. Dodanie kosztu Ryzyka (Realizacja H1)
            # Wzór: f(n) = g + h + W * R
            risk_cost = 0
            if algorithm_type == "risk_aware":
                risk_cost = cell_risk * risk_weight

            # 4. Dodanie kosztu Skrętu (Realizacja H2 - Płynność)
            turn_cost = 0
            # Jeśli mieliśmy poprzedni kierunek i jest on inny niż obecny -> zakręt
            if current.parent is not None and current.direction != (dx, dy):
                turn_cost = turn_penalty

            new_g = current.cost + dist_cost + risk_cost + turn_cost

            # Relaksacja krawędzi
            if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_g
                neighbor = Node(nx, ny, new_g, current, direction=(dx, dy))

                # Heurystyka (Dijkstra ma h=0)
                if algorithm_type == "dijkstra":
                    neighbor.heuristic = 0
                else:
                    # Euklidesowa
                    neighbor.heuristic = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2)

                heapq.heappush(open_list, neighbor)

    return [], {}  # Brak ścieżki


def reconstruct_path_and_stats(node: Node, grid_map: GridMap) -> Tuple[List[Tuple[int, int]], float, float, int]:
    """
    Odtwarza ścieżkę i oblicza metryki wymagane w pracy (długość, suma ryzyka, zakręty).
    """
    path = []
    total_risk = 0.0
    total_length = 0.0
    turns = 0

    current = node
    last_dir = None

    while current:
        path.append((current.x, current.y))

        # Zbieranie ryzyka (skumulowane ryzyko)
        cell_risk = grid_map.get_cost(current.x, current.y)
        # Ignorujemy 1.0 przy sumowaniu, bo to przeszkoda (nie wchodzimy na nią teoretycznie)
        if cell_risk < 1.0:
            total_risk += cell_risk

        if current.parent:
            # Oblicz długość
            dx = current.x - current.parent.x
            dy = current.y - current.parent.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            total_length += dist

            # Oblicz zakręty
            curr_dir = (dx, dy)
            if last_dir is not None and curr_dir != last_dir:
                turns += 1
            last_dir = curr_dir

        current = current.parent

    return path[::-1], {
        "length": total_length,
        "total_risk": total_risk,
        "turns": turns
    }
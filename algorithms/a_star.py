import heapq
import math


class Node:
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost  # G (koszt od startu)
        self.parent = parent
        self.heuristic = 0  # H (szacowany koszt do celu)

    @property
    def total_cost(self):
        return self.cost + self.heuristic  # F = G + H

    def __lt__(self, other):
        return self.total_cost < other.total_cost


def run_search(grid_map, start, goal, algorithm_type="astar", risk_weight=0.0):
    """
    Główna funkcja wyszukiwania.
    :param grid_map: Obiekt GridMap
    :param start: krotka (x, y)
    :param goal: krotka (x, y)
    :param algorithm_type: "dijkstra", "astar", "risk_aware"
    :param risk_weight: Waga ryzyka (dla Optimized A*)
    :return: (path, visited_nodes_count, path_cost)
    """
    start_node = Node(start[0], start[1], 0)

    open_list = []
    heapq.heappush(open_list, start_node)

    visited = set()
    # Słownik do przechowywania najniższego kosztu dotarcia do punktu
    g_score = {(start[0], start[1]): 0}

    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if (current.x, current.y) == goal:
            return reconstruct_path(current), nodes_expanded, current.cost

        if (current.x, current.y) in visited:
            continue
        visited.add((current.x, current.y))

        # Sprawdź sąsiadów (8 kierunków - ruch na skos dozwolony)
        # Możesz zmienić na 4 kierunki, jeśli dron nie lata na skos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = current.x + dx, current.y + dy

            # Czy w granicach mapy?
            if 0 <= nx < grid_map.width and 0 <= ny < grid_map.height:
                cell_risk = grid_map.get_cost(nx, ny)

                # Jeśli ryzyko == 1.0 (ściana), pomiń
                if cell_risk >= 1.0:
                    continue

                # Koszt ruchu:
                # 1.0 za prosto, 1.41 za skos (pierwiastek z 2)
                move_cost = math.sqrt(dx ** 2 + dy ** 2)

                # --- KLUCZOWE DLA TWOJEJ PRACY ---
                # Modyfikacja kosztu w zależności od algorytmu

                added_risk_cost = 0
                if algorithm_type == "risk_aware":
                    # Wzór: Koszt = Dystans + (Waga * Ryzyko)
                    added_risk_cost = cell_risk * risk_weight

                new_g = current.cost + move_cost + added_risk_cost

                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_g
                    neighbor = Node(nx, ny, new_g, current)

                    # Obliczanie H (Heurystyki)
                    if algorithm_type == "dijkstra":
                        neighbor.heuristic = 0
                    else:
                        # A* i Risk-Aware używają heurystyki (Euklidesowa)
                        neighbor.heuristic = math.sqrt((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2)

                    heapq.heappush(open_list, neighbor)

    return None, nodes_expanded, 0  # Nie znaleziono ścieżki


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]  # Odwracamy, żeby było od startu do celu
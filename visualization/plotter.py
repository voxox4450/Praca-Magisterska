import matplotlib.pyplot as plt
import numpy as np


def plot_simulation(grid_map, path, title="Simulation"):
    plt.figure(figsize=(10, 8))

    # Rysowanie siatki ryzyka (cmap='hot_r' da ładne przejście od białego do czerwonego)
    # Transpozycja (.T) jest potrzebna, bo numpy [x,y] vs matplotlib [row, col]
    plt.imshow(grid_map.grid.T, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(label='Poziom Ryzyka (0=Bezpiecznie, 1=Zakaz)')

    # Rysowanie ścieżki
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, color='blue', linewidth=2, label='Wyznaczona Trasa')
        plt.scatter([path_x[0]], [path_y[0]], color='green', marker='o', s=100, label='Start')
        plt.scatter([path_x[-1]], [path_y[-1]], color='purple', marker='x', s=100, label='Cel')

    plt.title(title)
    plt.legend()
    plt.grid(which='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.show()
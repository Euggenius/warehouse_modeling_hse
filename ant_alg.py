import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from random import choices

n = 7
alpha = 1
beta = 4
Q = 2.4
p = 0.8
t0 = 0.1

distance_matrix = np.random.rand(n, n)
feromon_matrix = np.zeros((n, n))
feromon_upd_matrix = np.zeros((n, n))

G = nx.Graph()
G.add_nodes_from(range(n))

pos = {i: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)}

plt.figure(figsize=(8, 8))
plt.ion()
# plt.title(f"Граф с {n} вершинами и жирностью ребер")
# plt.axis("off")

for i in range(0, n):
    for j in range(i, n):

        if i == j:
            distance_matrix[j][i] = 0
        else:
            G.add_edge(i, j, weight=int(distance_matrix[i][j] * 100))
            # G.add_edge(i, j)
            distance_matrix[j][i] = 0

for i in range(0, n):
    for j in range(i, n):
        if i != j:
            feromon_matrix[i][j] = t0
        feromon_matrix[j][i] = 0


def draw_graph(n, feromon_matrix):
    edge_widths = []

    max_feromon_value = 10 / np.max(feromon_matrix)

    for i in range(n - 1):
        for j in range(i + 1, n):
            edge_widths.append(feromon_matrix[i][j] * max_feromon_value)

    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10,
            width=edge_widths, font_color="black", font_weight="bold", font_family="sans-serif",
            edge_color="#c9f2c9", node_shape="o")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black', font_size=8)

    # plt.title(f"Граф с {n} вершинами и жирностью ребер")
    # plt.axis("off")
    plt.pause(10 ** -6)

    plt.clf()


def next_point_choise(actual_point, available_points):
    available_points_list = list(available_points)
    probabilities_list = []

    for point in available_points_list:
        pos1, pos2 = sorted([point, actual_point])
        probabilities_list.append((feromon_matrix[pos1][pos2] ** alpha) / (distance_matrix[pos1][pos2] ** beta))

    return choices(available_points_list, weights=probabilities_list)[0]


iterations_count = 1000

while True:
    path_matrix = np.zeros((n, n))

    for start_pos in range(n):
        current_pos = start_pos
        path_length = 0

        path_matrix[start_pos][0] = start_pos
        current_path = {start_pos}

        for _ in range(n - 1):
            update_pos = next_point_choise(current_pos, set(range(n)) - current_path)
            path_length += distance_matrix[min(current_pos, update_pos)][max(current_pos, update_pos)]
            current_pos = update_pos

            path_matrix[start_pos][len(current_path)] = current_pos
            current_path.add(current_pos)

        path_length += distance_matrix[min(current_pos, start_pos)][max(current_pos, start_pos)]

        for pos_i in range(n):
            pos1, pos2 = sorted([path_matrix[start_pos][pos_i], path_matrix[start_pos][(pos_i + 1) % n]])
            feromon_upd_matrix[int(pos1)][int(pos2)] += Q / path_length

    feromon_matrix *= (1 - p)
    feromon_matrix += feromon_upd_matrix
    feromon_upd_matrix = np.zeros((n, n))

    draw_graph(n, feromon_matrix)

plt.ioff()
plt.show()

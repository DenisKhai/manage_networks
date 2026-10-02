import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


MATRIX = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
])

PAGERANK_ALPHAS = (0.15, 0.25, 0.50, 0.85)
KATZ_ALPHAS = (0.10, 0.25, 0.40, 0.49, 0.50, 0.60)


def print_metric(title, values):
    print(title)
    for node, value in values.items():
        print(f'узел {node + 1}: {value:.6f}')


def pagerank_power(graph, alpha, tol=1e-12, max_iter=1000):
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    ranks = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - alpha) / n)

    for _ in range(max_iter):
        next_ranks = teleport.copy()
        dangling_sum = 0.0

        for node in nodes:
            i = index[node]
            successors = list(graph.successors(node))
            if successors:
                share = alpha * ranks[i] / len(successors)
                for successor in successors:
                    next_ranks[index[successor]] += share
            else:
                dangling_sum += ranks[i]

        if dangling_sum:
            next_ranks += alpha * dangling_sum / n

        if np.abs(next_ranks - ranks).sum() < n * tol:
            return dict(zip(nodes, next_ranks))
        ranks = next_ranks

    raise RuntimeError('PageRank did not converge')


def spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return max(abs(value) for value in eigenvalues), eigenvalues


def draw_graph(graph):
    pos = nx.spring_layout(graph, seed=3)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos=pos,
        labels={node: node + 1 for node in graph.nodes()},
        node_color='lightblue',
        node_size=850,
        arrows=True,
        arrowsize=22,
        connectionstyle='arc3,rad=0.08',
    )
    plt.title('Ориентированная сеть для дополнительной матрицы')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('experiment_matrix_network.png')


def main():
    graph = nx.from_numpy_array(MATRIX, create_using=nx.DiGraph)
    radius, eigenvalues = spectral_radius(MATRIX)
    alpha_limit = 1 / radius

    print('Задание 2. Дополнительная матрица')
    print('Эксперименты с alpha для PageRank и центральности Каца-Боначича')
    print()
    print('Матрица смежности:')
    print(MATRIX)
    print()

    print('Собственные значения матрицы g:')
    for value in eigenvalues:
        print(f'{value.real:+.6f}{value.imag:+.6f}j, |lambda| = {abs(value):.6f}')
    print()
    print(f'Максимальное по модулю собственное значение rho(g): {radius:.6f}')
    print(f'Верхняя граница для alpha в центральности Каца: alpha < 1 / rho(g) = {alpha_limit:.6f}')
    print()

    print_metric('Центральность по собственному вектору:', nx.eigenvector_centrality(graph))
    print()

    for alpha in PAGERANK_ALPHAS:
        print_metric(f'PageRank, alpha = {alpha:.2f}:', pagerank_power(graph, alpha))
        print()

    for alpha in KATZ_ALPHAS:
        if alpha < alpha_limit:
            values = nx.katz_centrality(graph, alpha=alpha, beta=1.0, normalized=True)
            print_metric(f'Центральность Каца-Боначича, alpha = {alpha:.2f}:', values)
        else:
            print(
                f'Центральность Каца-Боначича, alpha = {alpha:.2f}: '
                f'неоднозначно определена, так как alpha >= {alpha_limit:.6f}.'
            )
        print()

    print('Вывод:')
    print('При росте alpha в PageRank сильнее учитывается структура переходов по ребрам,')
    print('поэтому веса смещаются к узлам, которые получают больше входящих путей.')
    print('Для центральности Каца-Боначича ряд сходится только при alpha < 1 / rho(g),')
    print('где rho(g) - максимальное по модулю собственное значение матрицы g.')

    draw_graph(graph)


if __name__ == '__main__':
    main()

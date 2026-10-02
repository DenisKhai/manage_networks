import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

DIRECTED_MATRIX = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
])
PAGERANK_ALPHAS = (0.25, 0.50)
KATZ_ALPHA = 0.50
STAR_N_VALUES = (1, 2, 3, 4, 5, 6)

def print_metric(title, values):
    print(title)
    for node, value in values.items():
        print(f'узел {node}: {value:.6f}')

def draw_graphs(digraph, star):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    pos_digraph = nx.circular_layout(digraph)
    nx.draw_networkx(
        digraph,
        pos=pos_digraph,
        ax=axes[0],
        node_color='lightblue',
        node_size=850,
        arrows=True,
        arrowsize=22,
        connectionstyle='arc3,rad=0.08',
    )
    axes[0].set_title('Ориентированная сеть g')
    axes[0].axis('off')
    pos_star = nx.spring_layout(star, seed=2)
    nx.draw_networkx(
        star,
        pos=pos_star,
        ax=axes[1],
        node_color='lightgreen',
        node_size=850,
        edge_color='gray',
    )
    axes[1].set_title('Неориентированная звезда, N = 5')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('task_2_networks.png')


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
            successors = list(graph.successors(node)) if graph.is_directed() else list(graph.neighbors(node))
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


def normalized_katz_centrality(graph, alpha):
    return nx.katz_centrality(graph, alpha=alpha, beta=1.0, normalized=True)

def main():
    digraph = nx.from_numpy_array(DIRECTED_MATRIX, create_using=nx.DiGraph)
    star_for_picture = nx.star_graph(5)
    print('Задание 2')
    print('Проверка расчетов с помощью NetworkX')
    print()
    print('A. Ориентированная сеть g')
    print('Матрица смежности:')
    print(DIRECTED_MATRIX)
    print()
    eigen_directed = nx.eigenvector_centrality(digraph)
    print_metric('Центральность по собственному вектору:', eigen_directed)
    print()
    pageranks = {}

    for alpha in PAGERANK_ALPHAS:
        pageranks[alpha] = pagerank_power(digraph, alpha=alpha)
        print_metric(f'PageRank, alpha = {alpha:.2f}:', pageranks[alpha])
        print()
    print('Изменение PageRank при увеличении alpha с 0.25 до 0.50:')

    for node in digraph.nodes:
        diff = pageranks[0.50][node] - pageranks[0.25][node]
        print(f'узел {node}: {diff:+.6f}')
    print()

    print_metric(
        f'Центральность Каца, alpha = {KATZ_ALPHA:.2f}:',
        normalized_katz_centrality(digraph, KATZ_ALPHA),
    )
    print()

    print('Б. Неориентированная звезда с N + 1 узлами')
    print('Для проверки считаются звезды при N = 1, 2, 3, 4, 5, 6.')
    print('Узел 0 - центр звезды, остальные узлы - листья.')
    print()

    for n in STAR_N_VALUES:
        star = nx.star_graph(n)
        spectral_radius = math.sqrt(n)
        print(f'N = {n}, число узлов = {n + 1}, спектральный радиус = sqrt(N) = {spectral_radius:.6f}')

        print_metric(
            'Центральность по собственному вектору:',
            nx.eigenvector_centrality(star),
        )

        for alpha in PAGERANK_ALPHAS:
            print_metric(f'PageRank, alpha = {alpha:.2f}:', pagerank_power(star, alpha=alpha))

        if KATZ_ALPHA < 1 / spectral_radius:
            print_metric(
                f'Центральность Каца, alpha = {KATZ_ALPHA:.2f}:',
                normalized_katz_centrality(star, KATZ_ALPHA),
            )
        else:
            print(
                f'Центральность Каца при alpha = {KATZ_ALPHA:.2f} не определяется '
                f'по условию сходимости alpha < 1 / lambda_max, так как '
                f'{KATZ_ALPHA:.2f} >= {1 / spectral_radius:.6f}.'
            )
        print()

    print('Вывод по центральности Каца для звезды:')
    print('lambda_max = sqrt(N), поэтому при alpha = 0.5 требуется 0.5 < 1 / sqrt(N).')
    print('Отсюда sqrt(N) < 2, то есть N < 4. Для целых N центральность однозначно определена при N = 1, 2, 3.')
    draw_graphs(digraph, star_for_picture)


if __name__ == '__main__':
    main()

from collections import Counter
from math import comb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


N = 1000
ER_PROBABILITY = 0.01
SBM_SIZES = (500, 500)
SBM_PROBABILITIES = ((0.019, 0.001), (0.001, 0.019))
SEED = 42


def analyze_graph(title, graph):
    """Вычисляет и печатает характеристики одной сети."""
    n = graph.number_of_nodes()
    degrees = [degree for _, degree in graph.degree()]
    distribution = Counter(degrees)
    average_degree = sum(degrees) / n
    clustering = nx.average_clustering(graph)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    path_lengths = []
    weighted_distance_sum = 0.0
    connected_pairs = 0

    for nodes in components:
        size = len(nodes)
        path_length = 0.0
        if size > 1:
            component = graph.subgraph(nodes).copy()
            path_length = nx.average_shortest_path_length(component)
        path_lengths.append(path_length)

        pairs = comb(size, 2)
        weighted_distance_sum += path_length * pairs
        connected_pairs += pairs
    weighted_path_length = (
        weighted_distance_sum / connected_pairs if connected_pairs else 0.0
    )
    communities = list(nx.community.greedy_modularity_communities(graph))
    modularity = nx.community.modularity(graph, communities)

    print(title)
    print(f'Число вершин: {n}')
    print(f'Число ребер: {graph.number_of_edges()}')
    print(f'Средняя степень: {average_degree:.4f}')
    print('Распределение степеней (степень: число вершин):')
    print(', '.join(f'{degree}: {distribution[degree]}' for degree in sorted(distribution)))
    print(f'Средний локальный коэффициент кластеризации: {clustering:.6f}')
    print(f'Глобальный коэффициент кластеризации: {nx.transitivity(graph):.6f}')
    print(f'Число компонент связности: {len(components)}')
    print(
        f'Гигантская компонента: {len(components[0])} вершин '
        f'({len(components[0]) / n:.2%})'
    )
    print(f'Средняя длина пути в гигантской компоненте: {path_lengths[0]:.4f}')
    print(
        'Средняя длина пути по всем парам внутри компонент: '
        f'{weighted_path_length:.4f}'
    )
    print(f'Число сообществ, найденных по модульности: {len(communities)}')
    print(f'Модульность найденного разбиения: {modularity:.4f}')
    print()

    return {
        'degrees': degrees,
        'average_degree': average_degree,
        'clustering': clustering,
        'modularity': modularity,
        'communities': communities,
    }


def draw_results(er_graph, sbm_graph, er, sbm):
    """Строит распределения степеней, сравнение показателей и фрагменты сетей."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    max_degree = max(er['degrees'] + sbm['degrees'])
    bins = [degree - 0.5 for degree in range(max_degree + 2)]

    axes[0, 0].hist(er['degrees'], bins=bins, alpha=0.65, label='Эрдёш—Реньи')
    axes[0, 0].hist(sbm['degrees'], bins=bins, alpha=0.65, label='SBM')
    axes[0, 0].set_title('Распределение степеней')
    axes[0, 0].set_xlabel('Степень')
    axes[0, 0].set_ylabel('Число вершин')
    axes[0, 0].legend()

    metric_names = ['Средняя\nстепень', 'Кластеризация\n(×1000)', 'Модульность\n(×10)']
    er_values = [er['average_degree'], er['clustering'] * 1000, er['modularity'] * 10]
    sbm_values = [sbm['average_degree'], sbm['clustering'] * 1000, sbm['modularity'] * 10]
    axes[0, 1].bar([-0.18, 0.82, 1.82], er_values, 0.36, label='Эрдёш—Реньи')
    axes[0, 1].bar([0.18, 1.18, 2.18], sbm_values, 0.36, label='SBM')
    axes[0, 1].set_xticks([0, 1, 2], metric_names)
    axes[0, 1].set_title('Сравнение характеристик (с масштабированием)')
    axes[0, 1].legend()

    graphs = [er_graph, sbm_graph]
    results = [er, sbm]
    titles = ['Эрдёш—Реньи: фрагмент из 100 вершин', 'SBM: по 50 вершин из каждого блока']
    for index in range(2):
        graph = graphs[index]
        sample_nodes = list(range(100))
        if index == 1:
            sample_nodes = list(range(50)) + list(range(SBM_SIZES[0], SBM_SIZES[0] + 50))
        sample = graph.subgraph(sample_nodes)
        colors = {}
        for group, community in enumerate(results[index]['communities']):
            for node in community:
                colors[node] = group
        nx.draw_networkx(
            sample,
            pos=nx.spring_layout(sample, seed=SEED),
            ax=axes[1, index],
            node_color=[colors[node] for node in sample],
            cmap='tab20',
            node_size=30,
            edge_color='lightgray',
            width=0.6,
            with_labels=False,
        )
        axes[1, index].set_title(titles[index])
        axes[1, index].axis('off')

    figure.suptitle('Сравнение модели Эрдёша—Реньи и SBM')
    figure.tight_layout()
    figure.savefig('task_4_results.png', dpi=160)
    plt.close(figure)


def main():
    er_graph = nx.fast_gnp_random_graph(N, ER_PROBABILITY, seed=SEED)
    sbm_graph = nx.stochastic_block_model(
        SBM_SIZES, SBM_PROBABILITIES, seed=SEED
    )
    er = analyze_graph(
        f'А. Модель Эрдёша—Реньи G(n={N}, p={ER_PROBABILITY})',
        er_graph,
    )
    sbm = analyze_graph(
        'Б. Стохастическая блочная модель '
        f'(размеры={SBM_SIZES}, p_in={SBM_PROBABILITIES[0][0]}, '
        f'p_out={SBM_PROBABILITIES[0][1]})',
        sbm_graph,
    )

    print('Состав найденных сообществ SBM:')
    for index, community in enumerate(sbm['communities'], start=1):
        first = sum(sbm_graph.nodes[node]['block'] == 0 for node in community)
        second = len(community) - first
        print(
            f'Сообщество {index}: всего вершин — {len(community)}; '
            f'из первой исходной группы — {first}, из второй — {second}.'
        )
    draw_results(er_graph, sbm_graph, er, sbm)


if __name__ == '__main__':
    main()

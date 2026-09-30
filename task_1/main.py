import matplotlib.pyplot as plt
import networkx as nx

G = nx.karate_club_graph()

n = G.number_of_nodes()
m = G.number_of_edges()

print('Задание 1')
print('Сеть: клуб карате Захари')
print()

print('Число узлов:', n)
print('Число ребер:', m)
print('Число компонент связности:', nx.number_connected_components(G))

sum_degrees = 0
all_degrees = []
for node in G.nodes():
    degree = G.degree(node)
    sum_degrees += degree
    all_degrees.append(degree)

print('Средняя степень:', round(sum_degrees / n, 4))

print('\nРаспределение степеней:')
for degree in sorted(set(all_degrees)):
    print('степень', degree, '-', all_degrees.count(degree), 'узл.')

print('\nСреднее кратчайшее расстояние:', round(nx.average_shortest_path_length(G), 4))
print('Глобальная кластеризация:', round(nx.transitivity(G), 4))

local_clustering = nx.clustering(G)
values = list(local_clustering.values())

print('Средняя локальная кластеризация:', round(nx.average_clustering(G), 4))
print('Минимальная локальная кластеризация:', round(min(values), 4))
print('Максимальная локальная кластеризация:', round(max(values), 4))

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
plt.title('Клуб карате Захари')
plt.savefig('karate_club_network.png')

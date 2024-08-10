import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils import from_networkx
import random

def create_complete_subgraph(k):
    """
    Cria um subgrafo completo K_{k+1}.
    """
    return nx.complete_graph(k + 1)

def create_random_graph_with_k_clique(n_nodes, k):
    """
    Cria um grafo aleatório com n_nodes, incluindo um subgrafo completo K_{k+1}.
    """
    G = nx.erdos_renyi_graph(n_nodes, 0.3)  # Grafo aleatório
    if n_nodes < k + 1:
        raise ValueError("n_nodes deve ser pelo menos k+1")
    
    # Adiciona um subgrafo completo K_{k+1}
    clique_nodes = random.sample(G.nodes(), k + 1)
    G.add_edges_from(nx.complete_graph(k + 1).edges(clique_nodes))
    return G

def create_random_graph_without_k_clique(n_nodes, k):
    """
    Cria um grafo aleatório com n_nodes, sem subgrafos completos K_{k+1}.
    """
    G = nx.erdos_renyi_graph(n_nodes, 0.3)  # Grafo aleatório
    # Verifica se há um subgrafo completo K_{k+1} e remove se existir
    for subgraph in nx.find_cliques(G):
        if len(subgraph) == k + 1:
            G.remove_nodes_from(subgraph)
            break
    return G

def graph_to_data(graph):
    """
    Converte um grafo NetworkX para o formato torch_geometric Data.
    """
    data = from_networkx(graph)
    return data

def generate_graphs(num_graphs, n_nodes, k):
    """
    Gera grafos artificiais para treinamento e validação.
    """
    graphs_with_k_clique = [create_random_graph_with_k_clique(n_nodes, k) for _ in range(num_graphs // 2)]
    graphs_without_k_clique = [create_random_graph_without_k_clique(n_nodes, k) for _ in range(num_graphs // 2)]
    
    # Converte os grafos para o formato torch_geometric Data
    data_with_k_clique = [graph_to_data(g) for g in graphs_with_k_clique]
    data_without_k_clique = [graph_to_data(g) for g in graphs_without_k_clique]
    
    return data_with_k_clique, data_without_k_clique

# Parâmetros
num_graphs = 10.000  # Número total de grafos a gerar
n_nodes = 20       # Número de nós em cada grafo
k = 4             # Tamanho do subgrafo completo K_{k+1}

data_with_k_clique, data_without_k_clique = generate_graphs(num_graphs, n_nodes, k)

# Salvando os dados para uso posterior
torch.save(data_with_k_clique, 'graphs_with_k_clique.pt')
torch.save(data_without_k_clique, 'graphs_without_k_clique.pt')

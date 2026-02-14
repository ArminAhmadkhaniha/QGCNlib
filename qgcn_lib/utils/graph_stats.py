import torch
import networkx as nx
from torch_geometric.utils import degree

def get_topk_degree_edges(data, k: int = 40) -> torch.Tensor:
    """Selects top-k edges based on sum of degrees."""
    num_nodes = data.x.size(0)
    node_degrees = degree(data.edge_index[0], num_nodes=num_nodes) + \
                   degree(data.edge_index[1], num_nodes=num_nodes)
    source_indices = data.edge_index[0]
    target_indices = data.edge_index[1]
    edge_scores = node_degrees[source_indices] + node_degrees[target_indices]
    num_edges_to_select = min(k, edge_scores.size(0))
    _, topk_score_indices = torch.topk(edge_scores, k=num_edges_to_select)
    return data.edge_index[:, topk_score_indices]

def calculate_feature_smoothness(edge_index, X):
    """Computes average squared Euclidean distance across all edges."""
    row, col = edge_index
    diff = X[row] - X[col]
    squared_diff = (diff ** 2).sum(dim=1)
    return squared_diff.mean().item()

def calculate_topology_stats(edge_index, num_nodes):
    """Calculates clustering coefficient and connected components."""
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)
    avg_clustering = nx.average_clustering(G)
    num_components = nx.number_connected_components(G)
    return avg_clustering, num_components

def get_component_sizes(edge_index, num_nodes):
    """Returns sorted list of connected component sizes."""
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)
    components = list(nx.connected_components(G))
    sizes = [len(c) for c in components]
    sizes.sort(reverse=True)
    return sizes
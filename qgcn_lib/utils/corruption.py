import torch

def feature_shuffling_corruption(node_features: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a corrupted version of the graph by shuffling node features.
    """
    num_nodes = node_features.size(0)
    permutation_indices = torch.randperm(num_nodes, device=node_features.device)
    corrupted_features = node_features[permutation_indices]
    return corrupted_features, edge_index
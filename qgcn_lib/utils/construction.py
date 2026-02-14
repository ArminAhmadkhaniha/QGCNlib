import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

def get_similarity_matrix(X_np, metric_type):
    """Computes an N x N similarity matrix."""
    if metric_type == 'cosine':
        return cosine_similarity(X_np)
    elif metric_type == 'l1':
        dists = pairwise_distances(X_np, metric='l1', n_jobs=-1)
        return -dists 
    elif metric_type == 'l2':
        dists = pairwise_distances(X_np, metric='euclidean', n_jobs=-1)
        return -dists
    elif metric_type == 'jaccard':
        dists = pairwise_distances(X_np, metric='jaccard', n_jobs=-1)
        return 1 - dists
    elif metric_type == 'hamming':
        dists = pairwise_distances(X_np, metric='hamming', n_jobs=-1)
        return 1 - dists
    elif metric_type == 'correlation':
        return np.corrcoef(X_np)
    else:
        raise ValueError(f"Unknown metric: {metric_type}")

def build_graph_structure(similarity_matrix, k=5):
    """Mutual kNN logic to build edge index."""
    n = similarity_matrix.shape[0]
    topk = np.argsort(similarity_matrix, axis=1)[:, -k-1:-1]
    edges = set()
    for i in range(n):
        for j in topk[i]:
            if i in topk[j.item()]:
                edges.add((i, j.item()))
                edges.add((j.item(), i))
    
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
    isolates = [i for i, d in enumerate(deg) if d == 0]
    for i in isolates:
        for j in topk[i]:
            edges.add((i, j.item()))
            edges.add((j.item(), i))
            
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index, len(isolates)
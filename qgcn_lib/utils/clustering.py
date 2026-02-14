import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Sequence, Dict

def perform_kmeans_clustering(z: torch.Tensor, k: int = 2) -> tuple[np.ndarray, np.ndarray, float]:
    """Performs KMeans clustering on a PyTorch tensor embedding (z)."""
    z_np = z.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto').fit(z_np)
    labels = kmeans.labels_
    score = silhouette_score(z_np, labels)
    print(f"Silhouette Score (k={k}): {score:.4f}")
    return labels, z_np, score

def calculate_kmeans_inertia(z: torch.Tensor, k_range: Sequence[int]) -> Dict[int, float]:
    """Calculates inertia for KMeans clustering across a range of k."""
    z_np = z.detach().cpu().numpy()
    inertia_results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=123, n_init='auto').fit(z_np)
        inertia_results[k] = kmeans.inertia_
    return inertia_results
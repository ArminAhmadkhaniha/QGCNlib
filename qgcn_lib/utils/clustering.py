import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Sequence, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def perform_kmeans_clustering(
    z: torch.Tensor,
    k: int,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Apply K-means to one saved node-embedding matrix.

    Parameters
    ----------
    z:
        Node embeddings with shape [number_of_nodes, embedding_dimension].

    k:
        Number of clusters:
            - 5 for the 1000 Genomes super-population task;
            - 7 for Cora.

    seed:
        Random seed used by K-means.

    Returns
    -------
    predicted_clusters:
        K-means cluster assignment for every node.

    z_numpy:
        Embedding matrix converted to NumPy.

    silhouette:
        Internal silhouette score calculated from the predicted clusters.
    """
    z_numpy = z.detach().cpu().numpy()

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=seed,
    )

    predicted_clusters = kmeans.fit_predict(z_numpy)

    silhouette = silhouette_score(
        z_numpy,
        predicted_clusters,
    )

    print(
        f"Silhouette score for k={k}: "
        f"{silhouette:.4f}"
    )

    return predicted_clusters, z_numpy, silhouette

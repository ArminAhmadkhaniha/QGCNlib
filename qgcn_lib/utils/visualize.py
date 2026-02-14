import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from typing import Dict

def visualize_embedding(z_np: np.ndarray, labels: np.ndarray, score: float, k: int):
    """Applies t-SNE dimensionality reduction and visualizes clustering."""
    z_2d = TSNE(n_components=2, random_state=123, init='pca', learning_rate='auto').fit_transform(z_np)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='Set1', s=10, rasterized=True)
    plt.title(f"t-SNE Embedding with (k={k})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig("embedding.pdf", format="pdf", dpi=150)
    plt.close()

def plot_elbow_method(inertia_data: Dict[int, float], save_path: str = None):
    """Generates a line plot for the Elbow Method."""
    k_values = list(inertia_data.keys())
    inertias = list(inertia_data.values())
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o', linestyle='-', color='blue')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Elbow plot saved to {save_path}")
    else:
        plt.show()
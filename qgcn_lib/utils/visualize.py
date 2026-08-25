import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from typing import Dict

def visualize_embedding(
    z_numpy: np.ndarray,
    color_labels: np.ndarray,
    class_names: list[str],
    output_path: str,
    title: str,
    tsne_seed: int = 123,
) -> None:
    """
    Produce a two-dimensional t-SNE visualization.

    color_labels should contain the external ground-truth labels when
    the figure caption states that colors indicate real classes.
    """
    z_2d = TSNE(
        n_components=2,
        random_state=tsne_seed,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    ).fit_transform(z_numpy)

    plt.figure(figsize=(8, 8))

    unique_labels = np.unique(color_labels)

    for label in unique_labels:
        node_mask = color_labels == label

        label_name = (
            class_names[int(label)]
            if int(label) < len(class_names)
            else str(label)
        )

        plt.scatter(
            z_2d[node_mask, 0],
            z_2d[node_mask, 1],
            s=10,
            label=label_name,
            rasterized=True,
        )

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(
        title="External class",
        markerscale=2,
        frameon=False,
    )
    plt.tight_layout()

    plt.savefig(
        output_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )

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
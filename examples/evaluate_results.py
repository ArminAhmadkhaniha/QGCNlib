import torch
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, normalized_mutual_info_score

# --- Import from our Library ---
from qgcn_lib.utils import (
    perform_kmeans_clustering, 
    visualize_embedding, 
    calculate_kmeans_inertia, 
    plot_elbow_method
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Toggle this to match what you ran in main_paper.py
DATASET_NAME = "cora"  # Options: "cora", "snp"

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
z_path = os.path.join(current_dir, 'data', f"z_{DATASET_NAME}.pt")
data_path = os.path.join(current_dir, 'data', f"{DATASET_NAME}.pt")

# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------
def evaluate_embeddings():
    print(f"--> Starting Evaluation for {DATASET_NAME}...")
    
    # 1. Load Embeddings
    if not os.path.exists(z_path):
        print(f"[ERROR] Embedding file not found: {z_path}")
        print("Please run 'main_paper.py' first to generate embeddings.")
        return

    z = torch.load(z_path, map_location="cpu")
    print(f"--> Loaded Embeddings: {z.shape}")

    # 2. Elbow Method (Determine optimal k)
    print("--> Calculating Elbow Method (Inertia)...")
    k_range = range(2, 11)
    inertia = calculate_kmeans_inertia(z, k_range)
    
    elbow_plot_path = os.path.join(current_dir, f"elbow_{DATASET_NAME}.png")
    plot_elbow_method(inertia, save_path=elbow_plot_path)

    # 3. Clustering & Visualization
    # Define k based on dataset knowledge
    if DATASET_NAME == "cora":
        k = 7
    elif DATASET_NAME == "snp":
        k = 5 
    

    print(f"--> Performing K-Means (k={k})...")
    # labels: Cluster assignments from K-Means
    # z_np: Numpy version of embeddings
    # score: Silhouette Score
    labels, z_np, score = perform_kmeans_clustering(z, k)
    
    # Generate t-SNE Plot
    viz_path = os.path.join(current_dir, f"tsne_{DATASET_NAME}.pdf")

    visualize_embedding(z_np, labels, score, k)
    print(f"--> t-SNE visualization saved.")

    # 4. Classification / NMI Evaluation
    print("--> Evaluating Semantic Quality...")
    
    # Load Ground Truth if available
    ground_truth = None
    if os.path.exists(data_path):
        data_obj = torch.load(data_path, weights_only=False)
        if hasattr(data_obj, 'y') and data_obj.y is not None:
            ground_truth = data_obj.y.cpu().numpy()
            print("--> Ground Truth Labels Found.")

    if ground_truth is not None:
        # Case A: We have labels (Cora). Compare Clusters vs Truth.
        nmi = normalized_mutual_info_score(ground_truth, labels)
        print(f"--> NMI Score (Cluster Quality): {nmi:.4f}")
        
        # Linear Probing (Classification Accuracy)
        print("--> Running Logistic Regression (Linear Probe)...")
        X_train, X_test, y_train, y_test = train_test_split(
            z_np, ground_truth, test_size=0.2, random_state=123, stratify=ground_truth
        )
    else:
        # Case B: No labels (SNP/Unsupervised). Train Classifier on Cluster Labels.
        # This checks if the clusters are linearly separable.
        print("--> No Ground Truth. Evaluating Cluster Separability...")
        X_train, X_test, y_train, y_test = train_test_split(
            z_np, labels, test_size=0.2, random_state=123, stratify=labels
        )

    # Train Classifier
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n" + "="*40)
    print(f"RESULTS FOR {DATASET_NAME}")
    print("="*40)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("="*40)

if __name__ == "__main__":
    evaluate_embeddings()
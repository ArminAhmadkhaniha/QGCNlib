import torch
import os
import numpy as np
from torch_geometric.data import Data, InMemoryDataset

class MicroBenchmark(InMemoryDataset):
    """
    A synthetic clustering dataset generated on-the-fly or loaded from disk.
    
    This dataset mimics the structure of citations networks (like Cora) but 
    scaled down for rapid quantum simulation testing.
    """
    def __init__(self, root, n_nodes=1000, d_feat=16, n_clusters=3, transform=None):
        self.n_nodes = n_nodes
        self.d_feat = d_feat
        self.n_clusters = n_clusters
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['micro_unsup.pt']

    def process(self):
        # --- YOUR GENERATION LOGIC HERE ---
        print("Generating MicroBenchmark data...")
        
        # 1. Ground Truth Labels
        y = torch.arange(self.n_nodes) % self.n_clusters

        # 2. Generate Features with Cluster Separation
        # d=16 is perfect for 4-qubit Quantum Feature Extraction
        x = torch.zeros((self.n_nodes, self.d_feat))
        for i in range(self.n_clusters):
            mask = (y == i)
            # Shift the mean for each cluster to create semantic structure
            centroid = torch.randn(self.d_feat) * 2.5
            x[mask] = torch.randn(mask.sum(), self.d_feat) + centroid

        # 3. Generate Edges (Assortative structure)
        edge_list = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                # 20% chance to connect within cluster, 1% chance outside
                prob = 0.20 if y[i] == y[j] else 0.01
                if np.random.rand() < prob:
                    edge_list.append([i, j])
                    edge_list.append([j, i])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # 4. Wrap into a clean Data Object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        print(f"Nodes: {self.n_nodes} | Edges: {data.num_edges} | Qubit Req: {int(np.log2(self.d_feat))}")

        # Save using the standard PyG 'collate' method
        torch.save(self.collate([data]), self.processed_paths[0])


class ExperimentDataset(InMemoryDataset):
    """
    Wrapper for loading generic .pt files (Cora, SNP) via the library.
    """
    def __init__(self, root, file_path, transform=None):
        self.file_path = file_path
        super().__init__(root, transform)
        # Load the data directly from the path provided
        self.data, self.slices = torch.load(self.file_path, weights_only=False)
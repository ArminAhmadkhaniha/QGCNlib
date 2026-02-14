import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from sklearn.decomposition import PCA

def convert_pt_file(input_path, output_csv=None, output_txt=None):
    """Converts a PyTorch .pt file to CSV/TXT."""
    try:
        data = torch.load(input_path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            np_data = data.detach().cpu().numpy()
            print(f"Data shape: {np_data.shape}")
            if output_csv:
                pd.DataFrame(np_data).to_csv(output_csv, index=False, header=False)
                print(f"Saved to CSV: {output_csv}")
            if output_txt:
                np.savetxt(output_txt, np_data)
                print(f"Saved to TXT: {output_txt}")
        else:
            print(f"File contains {type(data)}, custom conversion needed.")
    except Exception as e:
        print(f"Error: {e}")

def extract_experiment_subgraph(data, num_nodes=80, start_node=0, target_dim=16):
    """Extracts a connected subgraph and reduces feature dimension via PCA."""
    # 1. Connected Component Extraction
    node_indices, _, _, _ = k_hop_subgraph(
        node_idx=start_node, 
        num_hops=10, 
        edge_index=data.edge_index, 
        relabel_nodes=False
    )
    # 2. Trim
    if len(node_indices) > num_nodes:
        node_indices = node_indices[:num_nodes]
    # 3. Subgraph
    edge_index_sub, _ = subgraph(node_indices, data.edge_index, relabel_nodes=True)
    # 4. Features & PCA
    x_sub = data.x[node_indices]
    if torch.is_tensor(x_sub):
        x_sub = x_sub.numpy()

    if x_sub.shape[1] > target_dim:
        pca = PCA(n_components=target_dim)
        x_reduced = pca.fit_transform(x_sub)
        x_final = torch.tensor(x_reduced, dtype=torch.float)
    else:
        x_final = torch.tensor(x_sub, dtype=torch.float)

    # 5. Labels
    y_sub = None
    if hasattr(data, 'y') and data.y is not None:
        y_sub = data.y[node_indices]
        print("-> Labels preserved.")
    else:
        print("-> No labels. Unsupervised mode.")

    sub_data = Data(x=x_final, edge_index=edge_index_sub, y=y_sub)
    print(f"Subgraph created: {sub_data.num_nodes} nodes, {target_dim} features.")
    return sub_data
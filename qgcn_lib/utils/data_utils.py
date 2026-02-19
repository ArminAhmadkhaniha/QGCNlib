import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, degree
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

def extract_experiment_subgraph(
    data, 
    num_nodes=None, 
    start_node=0, 
    num_edges=None, 
    edge_chunk_idx=0, 
    target_dim=None
):
    """
    Dynamically processes a graph for Iterative Chunking.
    
    Modes:
      1. Node-Centric: If `num_nodes` is set, extracts a local subgraph.
      2. Edge-Centric: If `num_edges` is set, selects top-scoring edges. 
         Use `edge_chunk_idx` to iterate through the 1st, 2nd, 3rd chunks, etc.
      3. Full Graph: If neither is set, processes the entire graph.
      
      * PCA is completely optional and only applied if `target_dim` is provided.
    """
    # 1. Base initialization
    x_final = data.x
    edge_index_final = data.edge_index
    y_final = data.y if hasattr(data, 'y') else None
    
    num_total_nodes = x_final.size(0)
    
    # 2. Routing Logic: Edge-Centric vs Node-Centric
    if num_edges is not None:
        # --- EDGE-CENTRIC MODE (Sparse Graphs like SNP) ---
        print(f"--> [Mode] Edge-Centric: Chunk {edge_chunk_idx} ({num_edges} edges per chunk)")
        
        # Calculate edge scores based on node degrees
        node_degrees = degree(data.edge_index[0], num_nodes=num_total_nodes) + \
                       degree(data.edge_index[1], num_nodes=num_total_nodes)
        edge_scores = node_degrees[data.edge_index[0]] + node_degrees[data.edge_index[1]]
        
        # Sort edges by score descending
        sorted_indices = torch.argsort(edge_scores, descending=True)
        
        # Calculate the chunk slice
        start_idx = edge_chunk_idx * num_edges
        end_idx = min(start_idx + num_edges, sorted_indices.size(0))
        
        if start_idx >= sorted_indices.size(0):
            print("    [WARNING] edge_chunk_idx is out of bounds. Returning graph with NO edges.")
            selected_edge_indices = torch.tensor([], dtype=torch.long)
        else:
            selected_edge_indices = sorted_indices[start_idx:end_idx]
            
        edge_index_final = data.edge_index[:, selected_edge_indices]
        # Note: We keep all nodes in x_final only limiting the edges.

    elif num_nodes is not None:
        # --- NODE-CENTRIC MODE (Dense Graphs like Cora) ---
        print(f"--> [Mode] Node-Centric: Target {num_nodes} nodes around node {start_node}")
        
        subset, _, _, _ = k_hop_subgraph(
            node_idx=start_node, 
            num_hops=10, 
            edge_index=data.edge_index, 
            relabel_nodes=False
        )
        
        # Trim to exact requested size
        if len(subset) > num_nodes:
            subset = subset[:num_nodes]
            
        # Create subgraph (This automatically relabels edge indices to match the new matrix)
        edge_index_final, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
        
        # Slice features and labels
        x_final = data.x[subset]
        if y_final is not None:
            y_final = y_final[subset]
            
    else:
        # --- FULL GRAPH MODE ---
        print("--> [Mode] Full Graph: No node or edge limits applied.")

    # 3. Optional Feature Reduction (PCA)
    if target_dim is not None:
        actual_features = x_final.size(1)
        actual_samples = x_final.size(0)
        
        if actual_features > target_dim:
            # PCA Safety: n_components cannot exceed the number of nodes
            n_components = min(target_dim, actual_samples)
            
            print(f"--> [PCA] Applying PCA: {actual_features} -> {n_components} dimensions.")
            if n_components < target_dim:
                print(f"    [WARNING] Node count ({actual_samples}) is smaller than target_dim ({target_dim}).")
            
            # Convert to numpy, fit PCA, convert back
            x_np = x_final.cpu().numpy() if torch.is_tensor(x_final) else x_final
            pca = PCA(n_components=n_components)
            x_reduced = pca.fit_transform(x_np)
            x_final = torch.tensor(x_reduced, dtype=torch.float)
        else:
            print(f"--> [PCA] Skipped: Current features ({actual_features}) already <= target ({target_dim}).")
            x_final = x_final.clone().detach().float() if torch.is_tensor(x_final) else torch.tensor(x_final, dtype=torch.float)
    else:
        print("--> [PCA] Disabled: Keeping original feature dimensions.")
        # Ensure it's a tensor
        if not torch.is_tensor(x_final):
            x_final = torch.tensor(x_final, dtype=torch.float)

    # 4. Construct Final PyG Data Object
    sub_data = Data(x=x_final, edge_index=edge_index_final, y=y_final)
    
    print(f"--> Resulting Graph: {sub_data.num_nodes} Nodes | {sub_data.num_edges} Edges | {sub_data.x.size(1)} Features\n")
    return sub_data
import sys
import os

# -----------------------------------------------------------------------------
# PATH FIX: Allow importing 'qgcn_lib' from the project root
# -----------------------------------------------------------------------------
# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root (one level up from examples)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add project root to Python's search path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"--> Added project root to path: {project_root}")
# -----------------------------------------------------------------------------


import torch
import os
import math
import tqdm
from torch_geometric.nn import DeepGraphInfomax

# --- 1. Import from our Library ---
from qgcn_lib.datasets import ExperimentDataset 
from qgcn_lib.nn import QGCNConv, SummaryMLP
from qgcn_lib.utils import set_all_seeds, feature_shuffling_corruption

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 2. Data Loading via our custom Wrapper
# -----------------------------------------------------------------------------


# Configuration: Choose 'cora' or 'snp'
dataset_name = 'snp' 

# Construct the path relative to this script
# We assume data is in a './data' folder next to this script
file_path = os.path.join(current_dir, 'data', f'{dataset_name}.pt')

print(f"--> Initializing Library Wrapper for: {dataset_name}")

# USE THE WRAPPER: This creates a PyG-compatible dataset object
try:
    dataset = ExperimentDataset(root=os.path.join(current_dir, 'data'), file_path=file_path)
    data = dataset[0]  # Get the graph object
    
    # Move to device immediately
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    print(f"--> Data Loaded Successfully via qgcn_lib")
    print(f"    Nodes: {data.num_nodes} | Features: {data.num_features}")
    
except FileNotFoundError:
    print(f"\n[ERROR] Could not find {dataset_name}.pt")
    print(f"Please ensure you have placed the file at: {file_path}")
    exit()

# -----------------------------------------------------------------------------
# 3. Dynamic Qubit Allocation
# -----------------------------------------------------------------------------
in_channels = data.x.size(1)
hidden_channels = math.ceil(math.log2(in_channels))

print(f"--> Feature Dimension (d): {in_channels}")
print(f"--> Quantum Latent Space (log d): {hidden_channels} Qubits")

# -----------------------------------------------------------------------------
# 4. Training Engine
# -----------------------------------------------------------------------------
def train_quantum_dgi(model, features, edge_index, epochs):
    """
    Standard training loop.
    """
    model.to(device)
    features = features.to(device)
    edge_index = edge_index.to(device)

    param_groups = []
    
    
    if hasattr(model.encoder, 'qc'):
        param_groups.append({'params': model.encoder.qc.parameters(), 'lr': 0.01})
    if hasattr(model.encoder, 'local_mp'):
        param_groups.append({'params': model.encoder.local_mp.parameters(), 'lr': 0.01})

    
    classical_params = []
    if hasattr(model.encoder, 'q_proj'): classical_params.extend(model.encoder.q_proj.parameters())
    if hasattr(model.encoder, 'bias'): classical_params.append(model.encoder.bias)
    if hasattr(model.encoder, 'prelu'): classical_params.extend(model.encoder.prelu.parameters())
    if hasattr(model, 'summary'): classical_params.extend(model.summary.parameters())
    
    param_groups.append({'params': classical_params, 'lr': 0.001})

    optimizer = torch.optim.Adam(param_groups)

    print(f"--> Starting Training for {epochs} epochs...")
    for epoch in tqdm.tqdm(range(epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(features, edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        
    print("--> Training finished.")
    return model

def run_experiment(features, idx_edge):
    set_all_seeds(123)
    num_nodes = features.size(0)
    
    encoder = QGCNConv(
        in_channels=features.size(1), 
        points=num_nodes, 
        hidden_channels=hidden_channels, 
        q_depth=5 
    )
    
    summary = SummaryMLP(hidden_channels)
    
    model = DeepGraphInfomax(
        hidden_channels=hidden_channels,
        encoder=encoder,
        summary=summary,
        corruption=feature_shuffling_corruption
    )
    
    model = train_quantum_dgi(model, features, idx_edge, epochs=50)
    
    model.eval()
    with torch.no_grad():
        z, _, _ = model(features, idx_edge)
    return z

# -----------------------------------------------------------------------------
# 5. Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    z_path = os.path.join(current_dir, f"z_{dataset_name}.pt")
    
    if os.path.exists(z_path):
        print(f"--> Loading existing embeddings from {z_path}")
        z = torch.load(z_path, map_location=device)
    else:
        print("--> Computing embeddings...")
        z = run_experiment(data.x, data.edge_index)
        torch.save(z.cpu(), z_path)
        print(f"--> Embeddings saved to {z_path}")
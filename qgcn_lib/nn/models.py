import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .quantum_networks import quantum_feature_extraction, local_qmp_layer 

class QGCNConv(MessagePassing):
    """
    A Quantum Graph Convolutional Network (QGCN) layer compatible with quantum simulators.
    """
    def __init__(self, in_channels, points, hidden_channels, q_depth=1):
        super().__init__(aggr='add')
        
        # 1. Quantum Feature Extraction
        self.n_qubits = math.ceil(math.log2(in_channels))
        self.qc = quantum_feature_extraction(self.n_qubits, q_depth)
        
        # 2. Local Quantum Message Passing
        self.n_local = self.n_qubits
        self.local_mp = local_qmp_layer(self.n_local)
        
        # 3. Non-Lineraity & Dimension-matching
        self.prelu = nn.PReLU(self.n_qubits)
        self.relu = nn.ReLU()
        self.q_proj = nn.Linear(2*self.n_local, hidden_channels)
        self.bias = nn.Parameter(torch.zeros(hidden_channels))

    def forward(self, x, edge_index):
        # Feature Extraction
        h = self.qc(x).float()     # shape: [N, log d]
        h = self.prelu(h)
        
        # Message Passing
        out = self.propagate(edge_index, h=h)

        
        q_out = self.relu(self.q_proj(out)) + self.bias
        return F.normalize(q_out + h, dim=1)

    def message(self, h_i, h_j):
        # Concatenate source and target features
        inputs = torch.cat([h_i, h_j], dim=1)  # shape: [E, 2*n_qubits]
        # Apply Angle Encoding circuit
        m_ij = self.local_mp(inputs) 
        return m_ij

class NISQQGCNConv(MessagePassing):
    """
    A Pure NISQ-compatible Quantum Graph Convolutional Network layer.
    """
    def __init__(self, in_channels, q_depth=1):
        super().__init__(aggr='add')
        
        # 1. Quantum Feature Extraction (Node-level)
        self.n_qubits = math.ceil(math.log2(in_channels))
        self.qc = quantum_feature_extraction(self.n_qubits, q_depth)
        
        # 2. Local Quantum Message Passing (Edge-level)
        
        self.local_mp = local_qmp_layer(self.n_qubits)

    def forward(self, x, edge_index):
        """
        Forward pass for the quantum graph convolution.
        """
        # 1. Encode initial node features into quantum states
        h = self.qc(x).float()  # shape: [N, log2(D)]
        
        # 2. Trigger message passing across all edges
        # PyG automatically routes 'h' to the message() function as h_i and h_j
        out = self.propagate(edge_index, h=h)  # shape: [N, log2(D)]
        
        return F.normalize(out + h, dim=1)

    def message(self, h_i, h_j):
        """
        Constructs the quantum message for a single directed edge (j -> i).
        PyG automatically parallelizes this across all edges in the batch.
        """
        # 1. Joint State Preparation: Concatenate target (h_i) and source (h_j)
        inputs = torch.cat([h_i, h_j], dim=1)  # shape: [E, 2 * log2(D)]
        
        # 2. Quantum Entanglement & Measurement
        # Applies the parameterized quantum circuit to the joint state
        m_ij = self.local_mp(inputs)           # shape: [E, 2 * log2(D)]
        
        # 3. Intra-Edge Aggregation (The "Folding" Step)
        target_portion = m_ij[:, :self.n_qubits]
        source_portion = m_ij[:, self.n_qubits:]
        
        folded_message = target_portion + source_portion  # shape: [E, log2(D)]
        
        return folded_message

class HybridQGCNConv(MessagePassing):
    """
    A Hybrid Quantum Graph Convolutional Network (QGCN) layer.
    Uses Quantum Feature Extraction + Classical Aggregation.
    """
    def __init__(self, in_channels, points, hidden_channels, q_depth=1):
        super().__init__(aggr='add')

        # Stage 1: Quantum Feature Extraction
        self.n_qubits = math.ceil(math.log2(in_channels))
        self.qc = quantum_feature_extraction(self.n_qubits, q_depth)
        self.q_proj = nn.Linear(self.n_qubits, hidden_channels)
        self.bias = nn.Parameter(torch.zeros(hidden_channels))

    def forward(self, x, edge_index):
        h = self.qc(x).float()     # shape: [N, log d]
        out = self.propagate(edge_index, x=h)
        q_out = torch.tanh(self.q_proj(out)) + self.bias
        return q_out

class SummaryMLP(torch.nn.Module):
    """
    Computes a graph-level summary vector (S) for DGI.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.lin = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, z, *args, **kwargs):
        return torch.sigmoid(self.lin(z.mean(dim=0)))
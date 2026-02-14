
import pennylane as qml
import torch


# ----------------------------
# Quantum feature extraction
# ----------------------------
def quantum_feature_extraction(n_qubits, q_depth):
    """
    Constructs a trainable Quantum Neural Network (QNN) layer for feature extraction 
    using PennyLane, configured as a PyTorch layer.

    This function utilizes Amplitude Embedding to encode the classical features 
    into the quantum state and a Strongly Entangling Layers ansatz as the VQC 
    to process the data.

    Args:
        n_qubits: The number of qubits (wires) in the quantum circuit, 
                  which must be sufficient for amplitude embedding (i.e., 
                  2**n_qubits >= number of input features).
        q_depth: The number of layers (depth) in the Strongly Entangling Layers ansatz.

    Returns:
        qml.qnn.TorchLayer: A fully initialized and trainable PyTorch layer 
                            that executes the defined quantum circuit.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)


    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        """
        The core quantum circuit for feature extraction.
        """
        eps = 1e-7
        inp = inputs + eps

        # Amplitude embedding
        qml.AmplitudeEmbedding(inp, wires=range(n_qubits), normalize=True, pad_with=1)

        # Variational circuit
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Return Z expectations
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (q_depth, n_qubits, 3)}
    q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
    with torch.no_grad():
        
        q_layer.weights.uniform_(-0.01, 0.01)
    return q_layer

# ----------------------------
# Local Quantum message passing
# ----------------------------
def local_qmp_layer(L):
    """
    
    Constructs a Batched Local Quantum Message Passing (QMP) layer as a PyTorch module.

    This QMP layer processes messages between two connected nodes (i and j) 
    by performing parallel, feature-wise quantum interactions. It uses 
    Angle Encoding to map classical features to qubit states.

    Args:
        feature_dim: The dimension (L) of the node features (h_i and h_j). 
                     

    Returns:
        qml.qnn.TorchLayer: A trainable PyTorch layer that executes the defined 
                            quantum circuit for message passing.
    """
    
    # We use L wires for Node i and L wires for Node j
    n_wires = 2 * L


    dev = qml.device("lightning.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, gamma, betta):
        
        for k in range(L):
            # Encode h_i[k] into wire k
            qml.RX(inputs[:, k], wires=k)
            # Encode h_j[k] into wire L+k
            qml.RX(inputs[:, L + k], wires=L + k)

        # 3. Local Interaction 
        # Interact Feature k of Node i with Feature k of Node j
        for k in range(L):
            i_wire = k
            j_wire = L + k

            qml.CNOT([i_wire, j_wire])
            qml.RZ(gamma[k], wires=j_wire) # Trainable interaction strength
            qml.CNOT([i_wire, j_wire])

        # 4. Local Mixers (Trainable)
        for k in range(n_wires):
            qml.RX(betta[k], wires=k)

        # 5. Measurement (Batched)
        return [qml.expval(qml.PauliZ(k)) for k in range(n_wires)]

    weight_shapes = {"gamma": (L,), "betta": (n_wires,)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)
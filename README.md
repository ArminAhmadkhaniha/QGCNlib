# QGCN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.40+-CD32A8.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.6+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 📋 Table of Contents

* **[📄 About the Paper](#-about-the-paper)**
* **[🛠 Installation](#-installation)**
    * [Clone & Install](#1-clone-the-repository)
* **[📦 Package Overview: qgcn_lib](#-package-overview-qgcn_lib)**
    * [Neural Network Modules](#1-neural-network-modules-qgcn_libnn)
    * [Datasets](#2-datasets-qgcn_libdatasets)
    * [Research Utilities](#3-research-utilities-qgcn_libutils)


## 📄 About the Paper
This Repository contains the model and algorithms in our research paper:
> **[Edge-Local and Qubit-Efficient Quantum Graph Learning for the NISQ Era]** > 

*Authors: Armin Ahmadkhaniha,Jake Doliskani* 


## 🛠 Installation

To install and use the implementation described in the paper, follow the steps below. Full-scale reproduction of the reported experiments can require substantial computational resources because of the cost of classical quantum-circuit simulation.

### 1. Clone the Repository

```bash
git clone [https://github.com/ArminAhmadkhaniha/QGCNlib.git](https://github.com/ArminAhmadkhaniha/QGCNlib.git)
```

Ensure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### Hardware Requirements

**Please be aware:** Simulating quantum circuits on classical hardware is computationally intensive. The runtime and memory usage of `NISQQGCNConv` depend heavily on three factors:
1.  **Qubit Count ($\lceil \log_2 d \rceil$):** The Hilbert space grows exponentially ($2^n$).
2.  **Edge Count ($|E|$):** Our **Latent Quantum Message Passing** requires calculating interactions for *every* edge in the graph.
3.  **RAM Availability:** Storing state vectors for large batched operations requires significant memory.

## 📦 Package Overview: `qgcn_lib`

The code is organized as a modular Python package `qgcn_lib`, designed to be compatible with the **PyTorch Geometric (PyG)** ecosystem using **PennyLane**. Please take a look at the examples for more details on how to use the library.

### 1. Neural Network Modules (`qgcn_lib.nn`)
This module contains the core quantum layers.
* **`NISQQGCNConv`**: The primary convolution layer. It implements the architecture using:
    * *Quantum Feature Extraction:* Amplitude Embedding + VQC (Strongly Entangling Layers).
    * *Latent Quantum Message Passing:* Structural aggregation in the logarithmic qubit space.
* **`HybridQGCNConv`**: A **Semi-Quantum** variant graph learning designed for ablation studies.
    * *Quantum Feature Extraction:* Uses the same encoding as QGCN.
    * *Classical Aggregation:* Replaces the quantum message passing with standard classical aggregation. 

### 2. Datasets (`qgcn_lib.datasets`)
Standardized data loaders that mirror `torch_geometric.datasets`.
* **`MicroBenchmark`**: A synthetic generator for rapid testing. Creates random graphs with controllable clusters and features to verify algorithmic logic without heavy compute.
* **`ExperimentDataset`**: A wrapper for loading real-world research datasets (e.g., **Cora**, **Genomics**) saved as `.pt` files, ensuring consistent formatting for the model.

### 3. Research Utilities (`qgcn_lib.utils`)
A comprehensive toolkit for the entire experimental lifecycle:
* **Corruption Functions:** `feature_shuffling_corruption` implements the negative sampling strategy required for unsupervised DGI training.
* **Evaluation:** Functions like `perform_kmeans_clustering` allow for immediate assessment of latent space quality.
* **Visualization:** Tools to generate **t-SNE** plots (`visualize_embedding`) to analyze cluster separation.
* **Reproducibility:** `set_all_seeds` ensures that all quantum and classical random processes are deterministic for valid comparisons.




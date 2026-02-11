# 🧬 Quantum-Enhanced LABS Optimization — NVIDIA iQuHACK 2026 Challenge

> **Hybrid quantum-classical approach to the Low Autocorrelation Binary Sequences (LABS) problem using NVIDIA CUDA-Q**

[![CUDA-Q](https://img.shields.io/badge/CUDA--Q-v0.13.0-76B900?style=flat&logo=nvidia)](https://nvidia.github.io/cuda-quantum/latest/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![iQuHACK 2026](https://img.shields.io/badge/iQuHACK-2026-purple.svg)](https://github.com/iQuHACK/2026-NVIDIA)

---

## Overview

This repository contains our team's submission for the [NVIDIA iQuHACK 2026 Challenge](https://github.com/iQuHACK/2026-NVIDIA) at MIT. The challenge tackles the **Low Autocorrelation Binary Sequences (LABS)** problem — a notoriously hard combinatorial optimization problem with critical applications in radar systems and telecommunications.

Our approach evolves the classical state-of-the-art **Memetic Tabu Search (MTS)** into a **hybrid quantum-enhanced workflow**, where quantum algorithm samples seed the classical MTS population. The solution leverages **NVIDIA CUDA-Q** for GPU-accelerated quantum simulation alongside GPU-accelerated classical search components.

## The LABS Problem

Given a binary sequence $S = \{s_1, s_2, \dots, s_N\}$ where $s_i \in \{-1, +1\}$, the goal is to minimize the **energy function** (sum of squared autocorrelations):

$$E(S) = \sum_{k=1}^{N-1} C_k^2, \quad \text{where} \quad C_k = \sum_{i=1}^{N-k} s_i \cdot s_{i+k}$$

Equivalently, we maximize the **merit factor**:

$$F(S) = \frac{N^2}{2E(S)}$$

Finding optimal sequences is computationally intractable for large $N$ — the search space grows as $2^N$ and the problem is known to be NP-hard.

## Approach

Our hybrid strategy combines quantum sampling with classical local search:

1. **Quantum Seed Generation** — Use quantum algorithms (implemented in CUDA-Q) to produce diverse, high-quality initial candidate sequences that explore the solution landscape more broadly than random initialization.
2. **Classical Memetic Tabu Search** — Feed quantum-generated seeds into an enhanced MTS algorithm that performs deep local search with tabu constraints to avoid cycling.
3. **GPU Acceleration** — Leverage NVIDIA GPUs to accelerate both the quantum circuit simulation and the classical search components, enabling scaling to larger sequence lengths.

## Repository Structure

```
├── Phase 1/                # Prototyping & CPU validation (qBraid)
│   ├── ...                 # Tutorial completion, algorithm design
│   └── ...                 # Research & planning deliverables
│
├── Phase 2/                # GPU acceleration & deployment (Brev)
│   ├── ...                 # CUDA-Q implementation
│   └── ...                 # Performance benchmarks & results
│
└── README.md
```

### Phase 1 — Prototyping on qBraid

Milestones completed during this phase:

- **Ramp Up**: Completed the scaffolded LABS tutorial to master the state-of-the-art Memetic Tabu Search algorithm and understand the LABS optimization landscape.
- **Research & Plan**: Designed a custom quantum strategy and acceleration plan, including choice of quantum ansatz, parameter optimization approach, and GPU deployment architecture.

### Phase 2 — GPU Acceleration on Brev

Milestones completed during this phase:

- **Build**: Implemented the hybrid quantum-enhanced algorithm in CUDA-Q, validated on CPU, then migrated to NVIDIA Brev for full GPU acceleration across multiple hardware configurations (L4, T4, A100).
- **Showcase & Retrospective**: Benchmarked performance, documented results, and presented findings including an analysis of the AI-driven development workflow.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Quantum Programming | [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/) v0.13.0 |
| Prototyping Platform | [qBraid](https://account-v2.qbraid.com/) |
| GPU Deployment | [NVIDIA Brev](https://brev.nvidia.com/) |
| Language | Python 3.10+ |
| Notebooks | Jupyter |

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA CUDA-Q v0.13.0+
- (Optional) Access to NVIDIA GPU for accelerated simulation

### Running Locally

```bash
# Clone the repository
git clone https://github.com/vbinvu68/Nvidia-challenge-MIT-2026.git
cd Nvidia-challenge-MIT-2026

# Install CUDA-Q (see https://nvidia.github.io/cuda-quantum/latest/install.html)
pip install cuda-quantum

# Navigate to the desired phase and open the notebooks
jupyter notebook
```

### Running on qBraid

1. Visit [qBraid](https://account-v2.qbraid.com/) and create an account.
2. Clone this repo in the qBraid environment.
3. Ensure the `CUDA-Q (v0.13.0)` environment is installed.
4. Set the notebook kernel to `Python 3 [cuda q-v0.13.0]`.

## Challenge Context

This project was developed during **MIT iQuHACK 2026** (January 31 – February 1, 2026), a quantum computing hackathon hosted at MIT and organized in collaboration with NVIDIA. The challenge emphasized an agentic, AI-assisted development workflow where teams operated as Technical Leadership — decomposing the problem, delegating across team members and AI agents, and verifying results.

## Acknowledgments

- **NVIDIA** for designing the LABS challenge and providing CUDA-Q, Brev GPU resources, and mentorship.
- **qBraid** for the zero-setup quantum development environment.
- **MIT iQuHACK** organizers for hosting an outstanding event.

## References

- Bernasconi, J. (1987). *Low autocorrelation binary sequences: statistical mechanics and configuration space analysis.* Journal de Physique, 48(4), 559–567.
- Gallardo, J. E., Cotta, C., & Fernández, A. J. (2009). *Finding low autocorrelation binary sequences with memetic algorithms.* Applied Soft Computing, 9(4), 1252–1262.
- [NVIDIA CUDA-Q Documentation](https://nvidia.github.io/cuda-quantum/latest/)
- [iQuHACK 2026 NVIDIA Challenge Repository](https://github.com/iQuHACK/2026-NVIDIA)

---

<p align="center">
  Built with ⚛️ and 🔥 at MIT iQuHACK 2026
</p>

# 🐝 Beerantum: Quantum-Enhanced LABS Optimization

## NVIDIA CUDA-Q Academic Challenge 2026

**Team Beerantum** | iQuHack

---

## Overview

This project implements a **GPU-accelerated quantum-enhanced optimization workflow** for solving the Low Autocorrelation Binary Sequences (LABS) problem. We combine:

- **Digitized Counterdiabatic Quantum Optimization** (CUDA-Q)
- **GPU-Accelerated Memetic Tabu Search** (CuPy)
- **Parallel Async Sampling** for efficient population generation

**Target Problem Size:** N = 29-30 qubits (practical maximum for statevector simulation)

---

## Team

| Role | Name | GitHub |
|------|------|--------|
| **Project Lead** | Anna Kristha Almazán Favela | [@Akri-A] |
| **GPU Acceleration PIC** | Van Binh Vu | [@vbinvu68] |
| **Quality Assurance PIC** | Ziwoong (Jim) Jang & Rudraksh Sharma | [@jjmain] [@Rudra1x] |
| **Technical Marketing PIC** | Sadiya Ansari | [@sadieea] |

---

## Repository Structure

```
├── README.md                          # This file
├── AI_REPORT.md                       # AI agent workflow documentation
├── PRD_Requirements_Checklist.md      # PRD coverage verification
├── LABS_GPU_Scaling_Plan_Beerantum.md # Technical scaling documentation
│
├── Notebooks/
│   ├── 01_quantum_enhanced_optimization_LABS_complete_final_Beerantum.ipynb  # Milestone 1
│   ├── 02_LABS_GPU_Acceleration_Milestone3_Beerantum.ipynb                   # Milestone 3
│   ├── Step_A.ipynb                   # CPU Validation
│   ├── GPU_Acceleration_and_Hardware_Migration.ipynb  # Step B
│   └── Step_C_GPU_Acceleration_of_the_classical_algorithm.ipynb
│
├── Source/
│   ├── labs_gpu.py                    # GPU acceleration module
│   ├── labs_utils_cpu.py              # CPU utility functions
│   ├── quantum_kernels.py             # CUDA-Q kernel definitions
│   ├── mts_cpu.py                     # Memetic Tabu Search (CPU)
│   └── labs_utils.py                  # Original utility functions
│
├── tests.py                           # Unit test suite
└── Presentation/
    └── Beerantum_Presentation.pptx    # Final presentation
```

---

## Quick Start

### Prerequisites

```bash
pip install cudaq
pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11
```

### Running on CPU (Development)

```python
import cudaq
from labs_gpu import GPUConfig, quantum_enhanced_mts

# Initialize (auto-detects GPU, falls back to CPU)
GPUConfig.initialize()

# Run quantum-enhanced optimization
result = quantum_enhanced_mts(N=7, quantum_shots=500, mts_iterations=50)
print(f"Best energy: {result['best_energy']}")
```

### Running on GPU (Brev)

```python
import cudaq
cudaq.set_target("nvidia")  # or "nvidia-mgpu" for N>=28

from labs_gpu import quantum_enhanced_mts
result = quantum_enhanced_mts(N=29, quantum_shots=1000, mts_iterations=100)
```

### Running Tests

```bash
pytest tests.py -v
```

---

## Milestones

### ✅ Milestone 1: Ramp Up (Complete)
- Implemented counterdiabatic quantum circuit (Eq. B3)
- Classical MTS baseline
- 7-category validation suite

### ✅ Milestone 2: Research & Plan (Complete)
- PRD with acceleration strategy
- Team roles assigned
- Verification plan defined

### ✅ Milestone 3: Build (Complete)
- **Step A:** CPU validation for N=3-10
- **Step B:** GPU backend migration with graceful fallback
- **Step C:** CuPy-accelerated classical MTS

### 🔄 Milestone 4: Showcase (Current)
- AI Report documentation
- Final presentation
- Benchmark results

---

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Validation Pass Rate | 100% | ✅ 100% |
| Symmetry Tests | 900/900 | ✅ 100% |
| Max Problem Size | N ≥ 29 | ✅ N=30 ready |
| GPU Speedup | >10x | ✅ 10-50x |
| CPU Fallback | Works | ✅ Verified |

---

## Algorithm

We implement the **Digitized Counterdiabatic Quantum Optimization** approach from:

> Hegde et al. (2024). "Scaling advantage with quantum-enhanced memetic tabu search for LABS." arXiv:2511.04553v1

### Workflow

1. **Quantum Sampling**: Trotterized counterdiabatic circuit generates initial population
2. **Population Selection**: Top sequences selected by LABS energy
3. **GPU-Accelerated MTS**: CuPy batch evaluation + tabu search refinement
4. **Output**: Optimal or near-optimal binary sequence

### Circuit Components

- **2-body rotations**: R_YZ, R_ZY
- **4-body rotations**: R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY

---

## Troubleshooting

### "cudaErrorInsufficientDriver"

The code automatically falls back to CPU. For GPU acceleration, use Brev with proper CUDA drivers.

```python
# Force CPU mode if needed
GPUConfig.force_cpu()
```

### Memory Limits

| N | Memory | Recommended GPU |
|---|--------|-----------------|
| 25 | 512 MB | Any |
| 29 | 8.6 GB | L4/A10 (24GB) |
| 30 | 17.2 GB | A100-40GB |

---

## References

1. Hegde et al. (2024). "Scaling advantage with quantum-enhanced memetic tabu search for LABS." arXiv:2511.04553v1
2. Mertens, S. (1996). "Exhaustive search for low-autocorrelation binary sequences." J. Phys. A
3. NVIDIA CUDA-Q Documentation: https://nvidia.github.io/cuda-quantum/

---

## License

Apache-2.0 AND CC-BY-NC-4.0

---

*Team Beerantum - NVIDIA CUDA-Q Academic Challenge 2026*

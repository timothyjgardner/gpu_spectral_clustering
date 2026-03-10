# gpu-spectral

GPU-accelerated spectral clustering using PyTorch. No RAPIDS/cuML dependency — just PyTorch for the GPU k-NN kernel and scipy/sklearn for sparse eigen and KMeans.

## Methods

| Class | Strategy | Practical scale |
|---|---|---|
| `GPUSpectral` | Full spectral with GPU k-NN | ~100K points |
| `NystromSpectral` | Nyström landmark approximation | ~1M points |
| `TwoStageSpectral` | Subsample + k-NN propagation | 10M+ points |

## Installation

```bash
pip install -e .
```

Requires a CUDA-capable GPU and PyTorch with CUDA support.

## Quick start

```python
import numpy as np
from gpu_spectral import GPUSpectral, TwoStageSpectral

X = np.random.randn(100_000, 128).astype(np.float32)

# Full GPU spectral
labels = GPUSpectral(n_clusters=10, n_neighbors=30).fit_predict(X)

# Two-stage for large datasets
labels = TwoStageSpectral(n_clusters=10, n_subsample=10000).fit_predict(X)
```

## Benchmarks (RTX 5090, 128D)

| Points | Full GPU | Nyström (5K) | Two-stage (10K) |
|---|---|---|---|
| 100K | 0.6s | 0.3s | 0.1s |
| 1M | 90s | 4.4s | 0.8s |
| 5M | — | — | 2.1s |
| 10M | — | — | 3.7s |
| 50M | — | — | 16.6s |

## API

### Low-level

- `gpu_knn(X, k, batch_size=2048)` — batched GPU k-NN, returns `(distances, indices)`
- `gpu_knn_cross(X_query, X_ref, k, batch_size=2048)` — cross k-NN, returns `indices`
- `spectral_core(X, n_clusters, n_neighbors, seed)` — full spectral, returns `labels`

### Classes

All have `.fit_predict(X)` returning integer cluster labels:

- `GPUSpectral(n_clusters, n_neighbors=30, seed=42)`
- `NystromSpectral(n_clusters, n_neighbors=30, seed=42, n_landmarks=5000)`
- `TwoStageSpectral(n_clusters, n_neighbors=30, seed=42, n_subsample=10000)`

## Benchmark script

```bash
python benchmark.py --sizes 10000,100000,1000000 --methods twostage,nystrom,full
```

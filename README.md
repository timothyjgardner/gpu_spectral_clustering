# gpu-spectral-clustering

GPU-accelerated spectral clustering using PyTorch. No RAPIDS/cuML dependency — just PyTorch for the GPU k-NN kernel and scipy/sklearn for sparse eigen and KMeans.

## Methods

| Class | Strategy | Practical scale |
|---|---|---|
| `GPUSpectral` | Full spectral with GPU k-NN | ~100K points |
| `NystromSpectral` | Nyström landmark approximation | ~1M points |
| `TwoStageSpectral` | Subsample + k-NN propagation | 10M+ points |

### Full spectral (`GPUSpectral`)

Standard spectral clustering with the k-NN affinity graph constructed on GPU:

1. **GPU k-NN**: For each of the *n* points, find the `n_neighbors` nearest neighbors using batched `torch.cdist`. Batching keeps GPU memory at O(batch × n) rather than O(n²).
2. **Affinity graph**: Build a sparse symmetric affinity matrix *W* using a Gaussian kernel with bandwidth set to the median neighbor distance: \( W_{ij} = \exp(-d_{ij}^2 / 2\sigma^2) \).
3. **Normalized Laplacian**: Compute the symmetric normalized Laplacian \( L = D^{-1/2} W D^{-1/2} \) where *D* is the degree matrix.
4. **Eigen decomposition**: Extract the top-*k* eigenvectors of *L* via `scipy.sparse.linalg.eigsh`.
5. **KMeans**: L2-normalize the eigenvectors and run KMeans to produce the final cluster assignments.

The GPU k-NN step is fast, but the sparse eigen decomposition runs on CPU and becomes the bottleneck beyond ~100K points. The two approximation methods below address this.

### Nyström approximation (`NystromSpectral`)

The Nyström method avoids computing the full n × n Laplacian by working with a smaller set of *m* landmark points and then extending the solution to all *n* points.

**Algorithm:**

1. Randomly sample `n_landmarks` (*m*) points from the dataset.
2. Run full spectral clustering on just the *m* landmarks: build the m × m affinity graph, compute its Laplacian, and extract *k* eigenvectors \( V_m \) with eigenvalues \( \lambda_1, \ldots, \lambda_k \).
3. Compute the cross-affinity matrix \( W_{nm} \) between all *n* points and the *m* landmarks, using the same Gaussian kernel and bandwidth \( \sigma \) from step 2. This step uses GPU cross-k-NN to find each point's nearest landmarks efficiently.
4. Extend the eigenvectors to all points via the Nyström formula: \( V_n \approx W_{nm} \cdot V_m \cdot \mathrm{diag}(1/\lambda) \).
5. L2-normalize the extended eigenvectors and run KMeans.

**Key parameter:**

- `n_landmarks` (default: 5000) — Number of landmark points. The eigen decomposition runs on an m × m matrix, so this controls the tradeoff between quality and speed. Values of 2000–10000 work well in practice. Higher values give results closer to full spectral but increase the cost of the landmark eigen step.

**Complexity:** O(m² + n·m) instead of O(n²), making it practical for ~1M points.

### Two-stage subsample + propagate (`TwoStageSpectral`)

The simplest and fastest approximation. It runs exact spectral clustering on a small subsample, then uses GPU k-NN to propagate labels to all remaining points.

**Algorithm:**

1. Randomly sample `n_subsample` (*m*) points from the dataset.
2. Run full spectral clustering on the *m* subsampled points (using `GPUSpectral` internally).
3. For every point in the full dataset, find its single nearest neighbor among the *m* subsampled points using GPU cross-k-NN.
4. Assign each point the cluster label of its nearest subsampled neighbor.

**Key parameter:**

- `n_subsample` (default: 10000) — Number of points for the exact spectral substep. Must be large enough to capture the cluster structure of the data. The spectral step runs in O(m²) and the propagation step runs in O(n·m), but with very small constants due to GPU acceleration. Values of 5000–20000 are typical.

**Complexity:** O(m² + n·m) with much smaller constants than Nyström since there is no cross-affinity matrix construction — just a single nearest-neighbor lookup. This makes it practical for 10M+ points.

**Tradeoff:** Points near cluster boundaries may be assigned to the wrong cluster if the subsample doesn't densely cover the boundary region. Increasing `n_subsample` mitigates this.

### Shared parameters

All three methods share these parameters:

- `n_clusters` — Number of clusters to find.
- `n_neighbors` (default: 30) — Number of neighbors for the k-NN affinity graph. Higher values produce a denser graph that captures broader structure but increase computation. For data with well-separated clusters, 15–30 is sufficient; for more complex manifold structure, 50–100 may help.
- `seed` (default: 42) — Random seed for reproducibility (affects KMeans initialization, random subsampling/landmark selection).

## Installation

```bash
pip install -e .
```

Requires a CUDA-capable GPU and PyTorch with CUDA support.

## Quick start

```python
import numpy as np
from gpu_spectral import GPUSpectral, NystromSpectral, TwoStageSpectral

X = np.random.randn(100_000, 128).astype(np.float32)

# Full GPU spectral (best quality, up to ~100K points)
labels = GPUSpectral(n_clusters=10, n_neighbors=30).fit_predict(X)

# Nyström approximation (up to ~1M points)
labels = NystromSpectral(n_clusters=10, n_landmarks=5000).fit_predict(X)

# Two-stage for large datasets (10M+ points)
labels = TwoStageSpectral(n_clusters=10, n_subsample=10000).fit_predict(X)
```

## Benchmarks (RTX 5090, 128D)

| Points | Full GPU | Nyström (5K landmarks) | Two-stage (10K subsample) |
|---|---|---|---|
| 100K | 0.6s | 0.3s | 0.1s |
| 1M | 90s | 4.4s | 0.8s |
| 5M | — | — | 2.1s |
| 10M | — | — | 3.7s |
| 50M | — | — | 16.6s |

## API

### Low-level functions

- `gpu_knn(X, k, batch_size=2048)` — Batched GPU k-NN. Returns `(distances, indices)` as numpy arrays, each of shape (n, k).
- `gpu_knn_cross(X_query, X_ref, k, batch_size=2048)` — Cross-set GPU k-NN: for each query point, find *k* nearest in the reference set. Returns `indices` of shape (n_query, k).
- `spectral_core(X, n_clusters, n_neighbors, seed)` — Full spectral clustering pipeline. Returns integer `labels` of shape (n,).

### Classes

All expose a scikit-learn-compatible `.fit_predict(X)` method returning integer cluster labels:

- `GPUSpectral(n_clusters, n_neighbors=30, seed=42)`
- `NystromSpectral(n_clusters, n_neighbors=30, seed=42, n_landmarks=5000)`
- `TwoStageSpectral(n_clusters, n_neighbors=30, seed=42, n_subsample=10000)`

## Benchmark script

```bash
python benchmark.py --sizes 10000,100000,1000000 --methods twostage,nystrom,full
python benchmark.py --sizes 1000000,5000000,10000000 --methods twostage --dim 256
```

"""Minimal example: cluster random data with GPU spectral clustering."""

import numpy as np
from gpu_spectral import GPUSpectral, TwoStageSpectral

# Generate synthetic data: 5 Gaussian blobs in 128D
rng = np.random.RandomState(0)
centers = rng.randn(5, 128).astype(np.float32) * 10
X = np.vstack([c + rng.randn(2000, 128).astype(np.float32) for c in centers])

# Full GPU spectral (good for up to ~100K points)
labels_full = GPUSpectral(n_clusters=5, n_neighbors=30).fit_predict(X)
print(f"Full spectral: {len(np.unique(labels_full))} clusters")

# Two-stage (scales to millions of points)
labels_fast = TwoStageSpectral(n_clusters=5, n_subsample=5000).fit_predict(X)
print(f"Two-stage:     {len(np.unique(labels_fast))} clusters")

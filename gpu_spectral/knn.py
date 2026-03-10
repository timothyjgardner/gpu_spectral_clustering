"""GPU-accelerated k-nearest-neighbor search using PyTorch."""

import numpy as np
import torch


def gpu_knn(X_np, k, batch_size=2048):
    """Batched k-NN on GPU using PyTorch.

    Computes pairwise L2 distances in batches to avoid O(n^2) memory.

    Parameters
    ----------
    X_np : ndarray of shape (n, d)
        Data matrix (float32 recommended).
    k : int
        Number of neighbors.
    batch_size : int
        Query batch size; controls peak GPU memory.

    Returns
    -------
    distances : ndarray of shape (n, k)
    indices : ndarray of shape (n, k)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(np.ascontiguousarray(X_np)).float().to(device)
    n = X.shape[0]
    all_dists = torch.zeros(n, k, device=device)
    all_idx = torch.zeros(n, k, dtype=torch.long, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(X[start:end], X)
        dists[:, start:end].fill_diagonal_(float('inf'))
        topk_d, topk_i = dists.topk(k, dim=1, largest=False)
        all_dists[start:end] = topk_d
        all_idx[start:end] = topk_i

    return all_dists.cpu().numpy(), all_idx.cpu().numpy()


def gpu_knn_cross(X_query, X_ref, k, batch_size=2048):
    """Batched cross k-NN: for each query find k nearest in a reference set.

    Parameters
    ----------
    X_query : ndarray of shape (n_q, d)
    X_ref : ndarray of shape (n_r, d)
    k : int
    batch_size : int

    Returns
    -------
    indices : ndarray of shape (n_q, k)
        Indices into ``X_ref``.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Q = torch.from_numpy(np.ascontiguousarray(X_query)).float().to(device)
    R = torch.from_numpy(np.ascontiguousarray(X_ref)).float().to(device)
    n = Q.shape[0]
    all_idx = torch.zeros(n, k, dtype=torch.long, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(Q[start:end], R)
        _, topk_i = dists.topk(k, dim=1, largest=False)
        all_idx[start:end] = topk_i

    return all_idx.cpu().numpy()

"""GPU-accelerated spectral clustering with exact and approximate variants.

All classes expose a scikit-learn-compatible ``fit_predict(X)`` interface.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from .knn import gpu_knn, gpu_knn_cross


def spectral_core(X, n_clusters, n_neighbors, seed):
    """Full spectral clustering on a dataset.

    1. Build a k-NN affinity graph (GPU-accelerated).
    2. Compute the normalized graph Laplacian.
    3. Extract the top-k eigenvectors via sparse eigendecomposition.
    4. Run KMeans on the L2-normalized eigenvectors.

    Parameters
    ----------
    X : ndarray of shape (n, d), float32
    n_clusters : int
    n_neighbors : int
    seed : int

    Returns
    -------
    labels : ndarray of shape (n,), int
    """
    n = X.shape[0]
    dists, indices = gpu_knn(X, n_neighbors)

    sigma = np.median(dists)
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()
    W = csr_matrix((weights.ravel(), (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2

    d = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = csr_matrix(
        (d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    L_norm = D_inv_sqrt @ W @ D_inv_sqrt

    _, eigenvectors = eigsh(L_norm, k=n_clusters, which='LM')
    embedding = normalize(eigenvectors, norm='l2', axis=1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(embedding)


class GPUSpectral:
    """Full GPU-accelerated spectral clustering.

    Best quality; O(n) GPU k-NN + O(n) sparse eigen.
    Practical up to ~100K points.

    Parameters
    ----------
    n_clusters : int
    n_neighbors : int
        Number of neighbors for the affinity graph.
    seed : int
    """

    def __init__(self, n_clusters, n_neighbors=30, seed=42):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed

    def fit_predict(self, X):
        return spectral_core(
            X, self.n_clusters, self.n_neighbors, self.seed)


class NystromSpectral:
    """Nystrom-approximated spectral clustering.

    Runs full spectral on ``n_landmarks`` randomly sampled points,
    then extends eigenvectors to all n points via the Nystrom formula.
    Practical for ~1M points.

    Parameters
    ----------
    n_clusters : int
    n_neighbors : int
    seed : int
    n_landmarks : int
        Number of landmark points for the Nystrom approximation.
    """

    def __init__(self, n_clusters, n_neighbors=30, seed=42,
                 n_landmarks=5000):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.n_landmarks = n_landmarks

    def fit_predict(self, X):
        n = X.shape[0]
        m = min(self.n_landmarks, n)
        rng = np.random.RandomState(self.seed)

        idx_land = rng.choice(n, m, replace=False)
        X_land = X[idx_land]

        # Full spectral eigenvectors on landmarks
        dists_mm, indices_mm = gpu_knn(X_land, self.n_neighbors)
        sigma = np.median(dists_mm)

        weights_mm = np.exp(-dists_mm ** 2 / (2 * sigma ** 2))
        rows = np.repeat(np.arange(m), self.n_neighbors)
        cols = indices_mm.ravel()
        W_mm = csr_matrix(
            (weights_mm.ravel(), (rows, cols)), shape=(m, m))
        W_mm = (W_mm + W_mm.T) / 2

        d = np.array(W_mm.sum(axis=1)).flatten()
        d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv = csr_matrix(
            (d_inv, (np.arange(m), np.arange(m))), shape=(m, m))
        L_mm = D_inv @ W_mm @ D_inv

        eigenvalues, V_land = eigsh(L_mm, k=self.n_clusters, which='LM')

        # Nystrom extension: affinity of all points to landmarks
        idx_cross = gpu_knn_cross(X, X_land, self.n_neighbors)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.from_numpy(X).float().to(device)
        L_t = torch.from_numpy(X_land).float().to(device)
        dists_nm = torch.zeros(n, self.n_neighbors, device=device)
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            idx_batch = torch.from_numpy(
                idx_cross[start:end]).to(device)
            diff = X_t[start:end].unsqueeze(1) - L_t[idx_batch]
            dists_nm[start:end] = diff.pow(2).sum(-1).sqrt()
        dists_nm = dists_nm.cpu().numpy()

        weights_nm = np.exp(-dists_nm ** 2 / (2 * sigma ** 2))
        rows_nm = np.repeat(np.arange(n), self.n_neighbors)
        cols_nm = idx_cross.ravel()
        W_nm = csr_matrix(
            (weights_nm.ravel(), (rows_nm, cols_nm)), shape=(n, m))

        # Extend eigenvectors: V_all ~ W_nm @ V_land @ diag(1/eigenvalues)
        V_ext = W_nm @ (V_land / eigenvalues[np.newaxis, :])
        embedding = normalize(V_ext, norm='l2', axis=1)
        km = KMeans(n_clusters=self.n_clusters, n_init=10,
                    random_state=self.seed)
        return km.fit_predict(embedding)


class TwoStageSpectral:
    """Two-stage spectral clustering: exact spectral on a subsample,
    then GPU k-NN propagation to assign all remaining points.

    Fastest variant; practical for 10M+ points.

    Parameters
    ----------
    n_clusters : int
    n_neighbors : int
    seed : int
    n_subsample : int
        Number of points for the exact spectral substep.
    """

    def __init__(self, n_clusters, n_neighbors=30, seed=42,
                 n_subsample=10000):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.n_subsample = n_subsample

    def fit_predict(self, X):
        n = X.shape[0]
        m = min(self.n_subsample, n)
        rng = np.random.RandomState(self.seed)

        idx_sub = rng.choice(n, m, replace=False)
        X_sub = X[idx_sub]

        labels_sub = spectral_core(
            X_sub, self.n_clusters, self.n_neighbors, self.seed)

        nn_idx = gpu_knn_cross(X, X_sub, k=1)
        return labels_sub[nn_idx.ravel()]

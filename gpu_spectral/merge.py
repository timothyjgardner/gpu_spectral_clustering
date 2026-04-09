"""Transition-based cluster merging for sequential data.

When spectral clustering over-segments sequential data (e.g. time series),
clusters that frequently transition between each other likely belong to
the same underlying state. This module merges such clusters using
hierarchical clustering on the transition probability matrix.

Typical usage::

    from gpu_spectral.merge import merge_clusters

    labels = clusterer.fit_predict(X)
    merged_labels, info = merge_clusters(labels, seq_len=1024, n_merge=10)
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def build_transition_matrix(labels, seq_len=None, boundary_mask=None):
    """Count transitions between cluster labels at consecutive timesteps.

    Parameters
    ----------
    labels : ndarray of shape (n,), int
        Cluster label for each timestep.
    seq_len : int or None
        If provided, transitions at window boundaries (every ``seq_len``
        steps) are skipped. This is useful when the label sequence is
        formed by concatenating fixed-length windows that are not
        temporally contiguous. If None, all consecutive pairs are counted.
    boundary_mask : ndarray of shape (n-1,), bool, or None
        If provided, ``boundary_mask[t]`` is True when the transition
        from ``labels[t]`` to ``labels[t+1]`` should be skipped (e.g.
        because it crosses a file boundary). This generalizes ``seq_len``
        to variable-length segments.

    Returns
    -------
    T : ndarray of shape (k, k), float64
        Transition count matrix where T[i, j] is the number of times
        the system moved from cluster i to cluster j.
    """
    k = int(labels.max()) + 1
    n = len(labels)

    if boundary_mask is not None:
        valid = ~boundary_mask
        T = np.zeros((k, k), dtype=np.float64)
        src = labels[:-1][valid]
        dst = labels[1:][valid]
        np.add.at(T, (src, dst), 1)
        return T

    T = np.zeros((k, k), dtype=np.float64)
    for t in range(n - 1):
        if seq_len is not None and (t + 1) % seq_len == 0:
            continue
        T[labels[t], labels[t + 1]] += 1
    return T


def transition_to_probability(T):
    """Row-normalize a transition count matrix to probabilities.

    Parameters
    ----------
    T : ndarray of shape (k, k)
        Transition counts.

    Returns
    -------
    P : ndarray of shape (k, k)
        Row-stochastic transition probability matrix.
    """
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def merge_by_transitions(T, n_merge, method='average'):
    """Merge clusters using hierarchical clustering on transition similarity.

    Clusters that frequently transition to each other are considered
    similar and merged first. Similarity is defined as the symmetrized
    transition probability: S[i,j] = (P[i,j] + P[j,i]) / 2.

    Parameters
    ----------
    T : ndarray of shape (k, k)
        Transition count matrix (from ``build_transition_matrix``).
    n_merge : int
        Target number of merged clusters.
    method : str
        Linkage method for ``scipy.cluster.hierarchy.linkage``.
        Default is 'average'.

    Returns
    -------
    merge_map : ndarray of shape (k,), int
        Mapping from old cluster label to new label:
        ``new_label = merge_map[old_label]``.
    """
    P = transition_to_probability(T)
    S = (P + P.T) / 2
    np.fill_diagonal(S, 0)

    D = 1.0 - S
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, None)

    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method=method)
    group_labels = fcluster(Z, t=n_merge, criterion='maxclust')

    unique = np.unique(group_labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[g] for g in group_labels])


def apply_merge(labels, merge_map):
    """Remap cluster labels through a merge map.

    Parameters
    ----------
    labels : ndarray of shape (n,), int
        Original cluster labels.
    merge_map : ndarray of shape (k,), int
        Mapping from ``merge_by_transitions``.

    Returns
    -------
    merged_labels : ndarray of shape (n,), int
    """
    return merge_map[labels]


def boundary_mask_from_indices(file_indices):
    """Build a boundary mask from a file/segment index array.

    Parameters
    ----------
    file_indices : ndarray of shape (n,), int
        Segment or file index for each timestep.

    Returns
    -------
    mask : ndarray of shape (n-1,), bool
        True at positions where the file index changes (boundary).
    """
    return np.diff(file_indices) != 0


def merge_clusters(labels, n_merge, seq_len=None, boundary_mask=None,
                   method='average'):
    """End-to-end cluster merging: build transitions, merge, relabel.

    Convenience function combining ``build_transition_matrix``,
    ``merge_by_transitions``, and ``apply_merge``.

    Parameters
    ----------
    labels : ndarray of shape (n,), int
        Cluster label for each timestep.
    n_merge : int
        Target number of merged clusters.
    seq_len : int or None
        Window length for skipping boundary transitions (see
        ``build_transition_matrix``).
    boundary_mask : ndarray of shape (n-1,), bool, or None
        If provided, skip transitions where mask is True. Overrides
        ``seq_len``. Use ``boundary_mask_from_indices(file_indices)``
        to build this from a file index array.
    method : str
        Linkage method (default: 'average').

    Returns
    -------
    merged_labels : ndarray of shape (n,), int
        Relabeled clusters (0 to n_merge-1).
    info : dict
        Diagnostic information:

        - ``'T_before'``: Transition matrix before merging.
        - ``'T_after'``: Transition matrix after merging.
        - ``'merge_map'``: Array mapping old labels to new labels.
        - ``'k_before'``: Number of clusters before merging.
        - ``'k_after'``: Number of clusters after merging.
    """
    T_before = build_transition_matrix(labels, seq_len=seq_len,
                                       boundary_mask=boundary_mask)
    merge_map = merge_by_transitions(T_before, n_merge, method=method)
    merged_labels = apply_merge(labels, merge_map)
    T_after = build_transition_matrix(merged_labels, seq_len=seq_len,
                                      boundary_mask=boundary_mask)

    return merged_labels, {
        'T_before': T_before,
        'T_after': T_after,
        'merge_map': merge_map,
        'k_before': T_before.shape[0],
        'k_after': int(merged_labels.max()) + 1,
    }

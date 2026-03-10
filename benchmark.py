#!/usr/bin/env python
"""Benchmark GPU spectral clustering at various dataset sizes."""

import argparse
import time

import numpy as np

from gpu_spectral import GPUSpectral, NystromSpectral, TwoStageSpectral


def run_benchmark(n_points, dim, n_clusters, n_neighbors, seed,
                  methods=('twostage', 'nystrom', 'full')):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_points, dim).astype(np.float32)
    print(f"\n{n_points:,} points x {dim}D, k={n_clusters}\n")

    clusterers = {
        'full': ('Full GPU spectral',
                 GPUSpectral(n_clusters, n_neighbors, seed)),
        'nystrom': ('Nystrom (m=5000)',
                    NystromSpectral(n_clusters, n_neighbors, seed,
                                   n_landmarks=5000)),
        'twostage': ('Two-stage (m=10000)',
                     TwoStageSpectral(n_clusters, n_neighbors, seed,
                                      n_subsample=10000)),
    }

    for key in methods:
        name, cls = clusterers[key]
        t0 = time.time()
        labels = cls.fit_predict(X)
        dt = time.time() - t0
        n_found = len(np.unique(labels))
        print(f"  {name:25s}  {dt:8.3f}s  clusters: {n_found}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=str,
                        default='10000,100000,1000000',
                        help='Comma-separated dataset sizes')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--n-neighbors', type=int, default=30)
    parser.add_argument('--methods', type=str, default='twostage,nystrom,full',
                        help='Comma-separated methods: full,nystrom,twostage')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]

    for n in sizes:
        run_benchmark(n, args.dim, args.n_clusters, args.n_neighbors,
                      args.seed, methods=methods)


if __name__ == '__main__':
    main()

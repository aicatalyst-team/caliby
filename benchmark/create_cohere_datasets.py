#!/usr/bin/env python3
"""
Create synthetic Cohere-like datasets for benchmarking.

Real Cohere datasets are Wikipedia article embeddings (768-dim from Cohere's embedding model).
This script creates synthetic normalized vectors that can be used for benchmarking.

For real Cohere datasets, visit:
- HuggingFace: https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings
- ann-benchmarks: https://github.com/erikbern/ann-benchmarks
"""

import numpy as np
import h5py
import os
import argparse
from pathlib import Path


def create_synthetic_cohere_dataset(name, n_vectors, n_queries=10000, dim=768, output_dir=None):
    """Create a synthetic dataset similar to Cohere format.
    
    Args:
        name: Dataset name (e.g., 'cohere1m', 'cohere10m')
        n_vectors: Number of base vectors
        n_queries: Number of query vectors
        dim: Vector dimension (Cohere uses 768)
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = Path.home() / 'Workspace' / 'datasets' / 'cohere'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{name}.hdf5'
    
    if output_file.exists():
        print(f"✓ {name} already exists at {output_file}")
        overwrite = input("Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            return str(output_file)
        print(f"  Removing existing file...")
        output_file.unlink()
    
    print(f"\nCreating synthetic {name} dataset")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Queries: {n_queries:,}")
    print(f"  Dimension: {dim}")
    print(f"  Output: {output_file}")
    print()
    
    # Generate random vectors (normalized for cosine similarity)
    print(f"  Generating {n_vectors:,} base vectors...")
    np.random.seed(42)
    train = np.random.randn(n_vectors, dim).astype(np.float32)
    print(f"  Normalizing vectors...")
    train = train / np.linalg.norm(train, axis=1, keepdims=True)
    
    print(f"  Generating {n_queries:,} query vectors...")
    test = np.random.randn(n_queries, dim).astype(np.float32)
    test = test / np.linalg.norm(test, axis=1, keepdims=True)
    
    # Compute ground truth (top 100 nearest neighbors using cosine similarity)
    print(f"  Computing ground truth (top-100 for each query)...")
    k = 100
    neighbors = np.zeros((n_queries, k), dtype=np.int32)
    distances = np.zeros((n_queries, k), dtype=np.float32)
    
    batch_size = 100  # Process queries in batches to show progress
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        
        # Compute similarities for batch
        sims = np.dot(train, test[start_idx:end_idx].T)  # (n_vectors, batch_size)
        
        # Find top-k for each query in batch
        for i, sim_col in enumerate(sims.T):
            query_idx = start_idx + i
            top_k_indices = np.argpartition(-sim_col, k)[:k]
            top_k_indices = top_k_indices[np.argsort(-sim_col[top_k_indices])]
            
            neighbors[query_idx] = top_k_indices
            distances[query_idx] = sim_col[top_k_indices]
        
        if (end_idx) % 1000 == 0:
            print(f"    Processed {end_idx:,}/{n_queries:,} queries ({100*end_idx/n_queries:.1f}%)")
    
    # Save to HDF5
    print(f"  Saving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train', data=train, compression='gzip')
        f.create_dataset('test', data=test, compression='gzip')
        f.create_dataset('neighbors', data=neighbors, compression='gzip')
        f.create_dataset('distances', data=distances, compression='gzip')
        f.attrs['distance'] = 'angular'  # cosine similarity
        f.attrs['dimension'] = dim
        f.attrs['description'] = f'Synthetic Cohere-like dataset with {n_vectors} vectors'
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✓ Created {output_file}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print()
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Create synthetic Cohere datasets for benchmarking')
    parser.add_argument('--dataset', type=str, default='cohere1m',
                        choices=['cohere1m', 'cohere10m', 'cohere100k', 'both'],
                        help='Dataset to create')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ~/Workspace/datasets/cohere)')
    parser.add_argument('--dim', type=int, default=768,
                        help='Vector dimension (default: 768, same as Cohere)')
    parser.add_argument('--queries', type=int, default=10000,
                        help='Number of query vectors (default: 10000)')
    
    args = parser.parse_args()
    
    datasets = {
        'cohere100k': 100_000,
        'cohere1m': 1_000_000,
        'cohere10m': 10_000_000,
    }
    
    if args.dataset == 'both':
        to_create = ['cohere1m', 'cohere10m']
    else:
        to_create = [args.dataset]
    
    for dataset_name in to_create:
        n_vectors = datasets[dataset_name]
        
        if n_vectors >= 10_000_000:
            print(f"\nWARNING: Creating {dataset_name} with {n_vectors:,} vectors")
            print(f"This will take ~30-60 minutes and use ~30GB of RAM")
            proceed = input("Continue? (y/n): ")
            if proceed.lower() != 'y':
                print("Skipped.")
                continue
        
        create_synthetic_cohere_dataset(
            name=dataset_name,
            n_vectors=n_vectors,
            n_queries=args.queries,
            dim=args.dim,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()

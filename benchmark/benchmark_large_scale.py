#!/usr/bin/env python3
"""
Large-Scale Caliby Benchmark (SIFT1M-scale, SIFT10M-scale)

Tests Caliby at production scales: 100K, 500K, 1M, 2M vectors.
Measures insertion throughput, search QPS, P50/P95/P99 latency, and recall@10.

Run with: python benchmark/benchmark_large_scale.py
"""

import numpy as np
import time
import os
import sys
import tempfile
import json
import subprocess as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
BUILD_PATH = str(Path(__file__).parent.parent / "build")

SCALES = [100000, 500000, 1000000, 2000000]
DIM = 128
K = 10
SEED = 42
EF_CONSTRUCTION = 200
EF_SEARCH = 100
M = 16


def generate_dataset(n):
    np.random.seed(SEED)
    base = np.random.randn(n, DIM).astype(np.float32)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    nq = 1000
    queries = base[:nq].copy()
    # Perturb slightly so queries aren't exact copies
    queries += np.random.randn(nq, DIM).astype(np.float32) * 0.01
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return base, queries


def benchmark_scale(n):
    """Run Caliby benchmark at a given scale, in a subprocess for clean state."""
    td = tempfile.mkdtemp()
    base_path = os.path.join(td, "base.npy")
    q_path = os.path.join(td, "queries.npy")

    base, queries = generate_dataset(n)
    np.save(base_path, base)
    np.save(q_path, queries)

    script = f'''
import sys, os, json, time
import numpy as np
sys.path.insert(0, "{BUILD_PATH}")
import caliby

td = "{td}"
base = np.load("{base_path}")
queries = np.load("{q_path}")
n, dim = base.shape
k = {K}

caliby.set_buffer_config(size_gb=8)
caliby.open(os.path.join(td, "data"))

schema = caliby.Schema()
schema.add_field("id", caliby.FieldType.INT)
col = caliby.Collection("bench", schema, vector_dim=dim)
col.create_hnsw_index("idx", M={M}, ef_construction={EF_CONSTRUCTION})

# Insert
t0 = time.time()
col.add([f"doc_{{i}}" for i in range(n)],
        [{{"id": i}} for i in range(n)],
        base.tolist())
t_insert = time.time() - t0

# Warmup
for _ in range(10):
    col.search_vector(queries[0].tolist(), "idx", k=k)

# Search
latencies = []
t0 = time.time()
for q in queries:
    tq = time.time()
    results = col.search_vector(q.tolist(), "idx", k=k)
    latencies.append(time.time() - tq)
t_search = time.time() - t0

# Ground truth for recall (top-100 from first 100 queries on small subset)
np.random.seed({SEED})
sample_idx = np.random.choice(n, min(n, 20000), replace=False)
sample = base[sample_idx]
sample_sim = queries[:100] @ sample.T
sample_gt = np.argsort(-sample_sim, axis=1)[:, :k]

pred = np.zeros((100, k), dtype=np.int64)
for i in range(100):
    r = col.search_vector(queries[i].tolist(), "idx", k=k)
    pred[i] = [rr.doc_id for rr in r]

# Remap pred labels to sample indices (only count if in sample)
hits = 0
for i in range(100):
    gt_set = set(sample_idx[g] for g in sample_gt[i])
    pred_set = set(pred[i])
    hits += len(gt_set & pred_set)
recall = hits / (100 * k)

caliby.close()

result = {{
    "scale": n,
    "dim": dim,
    "insert_s": round(t_insert, 2),
    "insert_vps": int(n / t_insert) if t_insert > 0 else 0,
    "search_qps": int(len(queries) / t_search) if t_search > 0 else 0,
    "p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 3),
    "p95_ms": round(float(np.percentile(latencies, 95)) * 1000, 3),
    "p99_ms": round(float(np.percentile(latencies, 99)) * 1000, 3),
    "recall": round(recall, 4),
}}
print("RESULT:" + json.dumps(result))
'''
    r = sp.run([sys.executable, "-c", script],
               capture_output=True, text=True, timeout=1200,
               env={**os.environ, "PYTHONPATH": BUILD_PATH})

    import shutil
    shutil.rmtree(td, ignore_errors=True)

    for line in r.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    print("STDERR:", r.stderr[-1000:])
    raise RuntimeError(f"Benchmark failed at scale {n}")


def main():
    print("=" * 80)
    print("  Caliby Large-Scale Vector Search Benchmark")
    print(f"  Config: HNSW M={M}, ef_construction={EF_CONSTRUCTION}, ef_search={EF_SEARCH}")
    print(f"  Dimension: {DIM}")
    print("=" * 80)

    results = []
    for n in SCALES:
        print(f"\n{'='*60}")
        print(f"  Testing {n:,} vectors ({n * DIM * 4 / 1024**2:.0f} MB raw data)")
        print(f"{'='*60}")
        try:
            r = benchmark_scale(n)
            results.append(r)
            print(f"  Insert:  {r['insert_vps']:>12,} v/s ({r['insert_s']:.1f}s)")
            print(f"  Search:  {r['search_qps']:>12,} q/s")
            print(f"  Latency: P50={r['p50_ms']:.3f}ms  P95={r['p95_ms']:.3f}ms  P99={r['p99_ms']:.3f}ms")
            print(f"  Recall:  {r['recall']:.1%}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Summary table
    if results:
        print(f"\n{'='*90}")
        print(f"  SUMMARY: Caliby at Scale")
        print(f"{'='*90}")
        print(f"{'Vectors':>12} {'Insert v/s':>14} {'QPS':>10} {'P50 ms':>10} {'P95 ms':>10} {'P99 ms':>10} {'Recall':>10}")
        print("-" * 78)
        for r in results:
            print(f"{r['scale']:>12,} {r['insert_vps']:>14,} {r['search_qps']:>10,} "
                  f"{r['p50_ms']:>10.3f} {r['p95_ms']:>10.3f} {r['p99_ms']:>10.3f} "
                  f"{r['recall']:>9.1%}")

        # Save to JSON
        out = "benchmark/caliby_scale_results.json"
        json.dump(results, open(out, "w"), indent=2)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

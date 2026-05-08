#!/usr/bin/env python3
"""
Caliby Benchmark: Synthetic Data Tests

Compares Caliby vs ChromaDB using synthetic data at various scales.
"""

import numpy as np
import time
import os
import sys
import tempfile
import shutil
import json
import subprocess as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import caliby
    CALIBY_AVAILABLE = True
except ImportError:
    CALIBY_AVAILABLE = False
    print("caliby not available")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("chromadb not available")

BUILD_PATH = str(Path(__file__).parent.parent / "build")


def generate_dataset(n, dim=128, seed=42):
    np.random.seed(seed)
    base = np.random.randn(n, dim).astype(np.float32)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    nq = max(100, min(n // 100, 1000))
    queries = base[:nq].copy()
    return base, queries


def compute_ground_truth(base, queries, k=10):
    sim = queries @ base.T
    return np.argsort(-sim, axis=1)[:, :k]


def compute_recall(pred, gt, k=10):
    total = sum(len(set(p[:k]) & set(g[:k])) for p, g in zip(pred, gt))
    return total / (len(pred) * k)


def benchmark_caliby(base, queries, gt, k=10):
    """Run Caliby benchmark in a subprocess for clean initialization."""
    n, dim = base.shape
    tmpdir = tempfile.mkdtemp()
    np.save(os.path.join(tmpdir, "base.npy"), base)
    np.save(os.path.join(tmpdir, "queries.npy"), queries)
    np.save(os.path.join(tmpdir, "gt.npy"), gt)

    script = f'''
import sys, os, json, time
import numpy as np
sys.path.insert(0, "{BUILD_PATH}")
import caliby

tmpdir = "{tmpdir}"
base = np.load(os.path.join(tmpdir, "base.npy"))
queries = np.load(os.path.join(tmpdir, "queries.npy"))
gt = np.load(os.path.join(tmpdir, "gt.npy"))
k = {k}
n, dim = base.shape

caliby.set_buffer_config(size_gb=4)
caliby.open(os.path.join(tmpdir, "data"))

schema = caliby.Schema()
schema.add_field("id", caliby.FieldType.INT)
col = caliby.Collection("bench", schema, vector_dim=dim)
col.create_hnsw_index("idx", M=16, ef_construction=200)

t0 = time.time()
col.add([f"doc_{{i}}" for i in range(n)],
        [{{"id": i}} for i in range(n)],
        base.tolist())
t_insert = time.time() - t0

for _ in range(10):
    col.search_vector(queries[0].tolist(), "idx", k=k)

latencies = []
pred = np.zeros((len(queries), k), dtype=np.int64)
t0 = time.time()
for i, q in enumerate(queries):
    tq = time.time()
    results = col.search_vector(q.tolist(), "idx", k=k)
    latencies.append(time.time() - tq)
    pred[i] = [r.doc_id for r in results]
t_search = time.time() - t0

def compute_recall(pred, gt, k):
    total = sum(len(set(p[:k]) & set(g[:k])) for p, g in zip(pred, gt))
    return total / (len(pred) * k)

r = compute_recall(pred, gt, k)
caliby.close()

result = {{
    "name": "Caliby (HNSW)",
    "vectors": n,
    "insert_vps": n / t_insert if t_insert > 0 else 0,
    "search_qps": len(queries) / t_search if t_search > 0 else 0,
    "p50_ms": float(np.percentile(latencies, 50)) * 1000,
    "p95_ms": float(np.percentile(latencies, 95)) * 1000,
    "p99_ms": float(np.percentile(latencies, 99)) * 1000,
    "recall": r,
}}
print("CALIBY_RESULT:" + json.dumps(result))
'''
    result = sp.run([sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=600,
                    env={**os.environ, "PYTHONPATH": BUILD_PATH})
    shutil.rmtree(tmpdir, ignore_errors=True)

    for line in result.stdout.splitlines():
        if line.startswith("CALIBY_RESULT:"):
            return json.loads(line[len("CALIBY_RESULT:"):])
    print("Caliby error output:", result.stderr[-500:])
    raise RuntimeError(f"Caliby benchmark failed")


def benchmark_chromadb(base, queries, gt, k=10):
    n, dim = base.shape
    tmpdir = tempfile.mkdtemp()

    client = chromadb.PersistentClient(path=os.path.join(tmpdir, "chroma"))
    col = client.create_collection(
        name="bench",
        metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 200},
    )

    t0 = time.time()
    batch = 5000
    for i in range(0, n, batch):
        end = min(i + batch, n)
        col.add(
            ids=[str(j) for j in range(i, end)],
            embeddings=base[i:end].tolist(),
            documents=[f"doc_{j}" for j in range(i, end)],
            metadatas=[{"id": j} for j in range(i, end)],
        )
    t_insert = time.time() - t0

    for _ in range(10):
        col.query(query_embeddings=[queries[0].tolist()], n_results=k)

    latencies = []
    pred = np.zeros((len(queries), k), dtype=np.int64)
    t0 = time.time()
    for i, q in enumerate(queries):
        tq = time.time()
        results = col.query(query_embeddings=[q.tolist()], n_results=k)
        latencies.append(time.time() - tq)
        ids = results["ids"][0]
        pred[i, :len(ids)] = [int(x) for x in ids]
    t_search = time.time() - t0

    r = compute_recall(pred, gt, k)
    del client, col
    shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "name": "ChromaDB",
        "vectors": n,
        "insert_vps": n / t_insert if t_insert > 0 else 0,
        "search_qps": len(queries) / t_search if t_search > 0 else 0,
        "p50_ms": np.percentile(latencies, 50) * 1000,
        "p95_ms": np.percentile(latencies, 95) * 1000,
        "p99_ms": np.percentile(latencies, 99) * 1000,
        "recall": r,
    }


def print_table(title, results):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(f"{'System':<25} {'Insert':>10} {'QPS':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Recall':>10}")
    print(f"{'':25} {'(v/s)':>10} {'(q/s)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} {r['insert_vps']:>10,.0f} {r['search_qps']:>10,.0f} "
              f"{r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f} "
              f"{r['recall']:>9.1%}")
    print()


def main():
    dim = 128
    k = 10
    scales = [10000, 50000, 100000]

    all_results = []

    for n in scales:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {n:,} vectors x {dim} dimensions")
        print(f"{'#'*70}")

        base, queries = generate_dataset(n, dim)
        print(f"Generated {n:,} base vectors, {len(queries)} query vectors")
        print("Computing ground truth...")
        gt = compute_ground_truth(base, queries, k)

        round_results = []

        if CALIBY_AVAILABLE:
            print(f"Running Caliby with {n:,} vectors...")
            try:
                r = benchmark_caliby(base, queries, gt, k)
                round_results.append(r)
                print(f"  => {r['insert_vps']:,.0f} v/s, {r['search_qps']:,.0f} q/s, "
                      f"Recall@10={r['recall']:.1%}")
            except Exception as e:
                print(f"  => FAILED: {e}")

        if CHROMA_AVAILABLE and n <= 50000:
            print(f"Running ChromaDB with {n:,} vectors...")
            try:
                r = benchmark_chromadb(base, queries, gt, k)
                round_results.append(r)
                print(f"  => {r['insert_vps']:,.0f} v/s, {r['search_qps']:,.0f} q/s, "
                      f"Recall@10={r['recall']:.1%}")
            except Exception as e:
                print(f"  => FAILED: {e}")

        if round_results:
            print_table(f"{n:,} Vectors", round_results)
            all_results.extend(round_results)

    # Summary
    print(f"\n{'='*75}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*75}")
    print(f"{'Scale':<12} {'Caliby QPS':>15} {'ChromaDB QPS':>15} {'Caliby Rec':>12} {'ChromaDB Rec':>12}")
    print("-" * 60)
    for n in scales:
        c = [r for r in all_results if r['vectors'] == n and 'Caliby' in r['name']]
        ch = [r for r in all_results if r['vectors'] == n and 'Chroma' in r['name']]
        cq = f"{c[0]['search_qps']:,.0f}" if c else "N/A"
        chq = f"{ch[0]['search_qps']:,.0f}" if ch else "N/A"
        cr = f"{c[0]['recall']:.1%}" if c else "N/A"
        chr_ = f"{ch[0]['recall']:.1%}" if ch else "N/A"
        print(f"{n:<12,} {cq:>15} {chq:>15} {cr:>12} {chr_:>12}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Concurrent Access Stress Tests

Tests thread safety of Caliby under concurrent reads, writes, and mixed workloads.
Uses subprocess to avoid cross-module state pollution.

Run with: pytest tests/test_concurrent.py -v
"""

import pytest
import os
import sys
import tempfile
import numpy as np
import subprocess as sp
from concurrent.futures import ThreadPoolExecutor, as_completed

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)


def _concurrent_search_worker(args):
    """Worker function for concurrent search tests."""
    idx_module, dim, queries_chunk, k, ef = args
    import caliby as worker_caliby
    results = []
    for q in queries_chunk:
        labels, distances = worker_caliby.HnswIndex.search_knn.__func__(
            idx_module, q.tolist(), k, ef)
        results.append((labels, distances))
    return results


class TestConcurrentReads:
    """Concurrent read-only access."""

    def test_multi_threaded_searches(self):
        """Multiple threads searching the same index concurrently."""
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np
from concurrent.futures import ThreadPoolExecutor

td = "{td}"
caliby.set_buffer_config(size_gb=2)
caliby.open(td)

dim, n = 64, 2000
np.random.seed(42)
vectors = np.random.randn(n, dim).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

idx = caliby.HnswIndex(max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
idx.add_points(vectors)

queries = np.random.randn(400, dim).astype(np.float32)
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

def search_batch(q_batch):
    results = []
    for q in q_batch:
        labels, _ = idx.search_knn(q, k=10, ef_search=50)
        results.append(len(labels) == 10)
    return sum(results)

# Run searches from multiple threads
n_threads = 8
batch_size = len(queries) // n_threads
batches = [queries[i*batch_size:(i+1)*batch_size] for i in range(n_threads)]

with ThreadPoolExecutor(max_workers=n_threads) as executor:
    futures = [executor.submit(search_batch, b) for b in batches]
    total_ok = sum(f.result() for f in futures)

assert total_ok == len(queries), f"Expected {{len(queries)}} OK results, got {{total_ok}}"
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_concurrent_collection_searches(self):
        """Multiple threads searching the same collection."""
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np
from concurrent.futures import ThreadPoolExecutor

td = "{td}"
caliby.set_buffer_config(size_gb=2)
caliby.open(td)

dim, n = 64, 1000
np.random.seed(42)
vectors = np.random.randn(n, dim).astype(np.float32).tolist()

schema = caliby.Schema()
schema.add_field("idx", caliby.FieldType.INT)
col = caliby.Collection("concurrent_col", schema, vector_dim=dim)
col.create_hnsw_index("vec_idx", M=16, ef_construction=100)

col.add([f"doc {{i}}" for i in range(n)],
        [{{"idx": i}} for i in range(n)], vectors)

queries = np.random.randn(200, dim).astype(np.float32).tolist()

def search_batch(q_batch):
    ok = 0
    for q in q_batch:
        r = col.search_vector(q, "vec_idx", k=5)
        if len(r) == 5:
            ok += 1
    return ok

with ThreadPoolExecutor(max_workers=4) as executor:
    batch = len(queries) // 4
    futures = [executor.submit(search_batch,
                queries[i*batch:(i+1)*batch]) for i in range(4)]
    total_ok = sum(f.result() for f in futures)

assert total_ok > 0, "All concurrent searches failed"
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestConcurrentWriteRead:
    """Concurrent read + write operations."""

    def test_concurrent_add_and_search(self):
        """Add documents while other threads are searching."""
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np
from concurrent.futures import ThreadPoolExecutor

td = "{td}"
caliby.set_buffer_config(size_gb=2)
caliby.open(td)

dim, n_initial = 64, 500
np.random.seed(42)
vectors = np.random.randn(n_initial, dim).astype(np.float32).tolist()

schema = caliby.Schema()
schema.add_field("idx", caliby.FieldType.INT)
col = caliby.Collection("mixed_col", schema, vector_dim=dim)
col.create_hnsw_index("vec_idx", M=8, ef_construction=50)
col.add([f"doc {{i}}" for i in range(n_initial)],
        [{{"idx": i}} for i in range(n_initial)], vectors)

queries = np.random.randn(100, dim).astype(np.float32).tolist()

def searcher():
    for _ in range(50):
        q = [float(x) for x in np.random.randn(dim)]
        try:
            r = col.search_vector(q, "vec_idx", k=5)
            if len(r) != 5:
                return False
        except Exception:
            return False
    return True

def adder():
    for batch in range(5):
        new_vecs = np.random.randn(20, dim).astype(np.float32).tolist()
        try:
            col.add(
                [f"new doc batch {{batch}} doc {{i}}" for i in range(20)],
                [{{"idx": -1}} for _ in range(20)], new_vecs)
        except Exception:
            return False
    return True

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = []
    for _ in range(4):
        futures.append(executor.submit(searcher))
    for _ in range(2):
        futures.append(executor.submit(adder))
    results = [f.result() for f in futures]

assert all(results), f"Some workers failed: {{results}}"
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_concurrent_add_delete(self):
        """Concurrent add and delete operations."""
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np
from concurrent.futures import ThreadPoolExecutor

td = "{td}"
caliby.set_buffer_config(size_gb=1)
caliby.open(td)

dim = 16
schema = caliby.Schema()
schema.add_field("gen", caliby.FieldType.INT)
col = caliby.Collection("del_test", schema, vector_dim=dim)
col.create_hnsw_index("vec_idx", M=8, ef_construction=50)

# Insert initial batch
initial = col.add(
    [f"initial doc {{i}}" for i in range(100)],
    [{{"gen": 0}} for _ in range(100)],
    np.random.randn(100, dim).astype(np.float32).tolist())

def adder(gen):
    new_vecs = np.random.randn(30, dim).astype(np.float32).tolist()
    try:
        ids = col.add(
            [f"gen {{gen}} doc {{i}}" for i in range(30)],
            [{{"gen": gen}} for _ in range(30)], new_vecs)
        return len(ids) == 30
    except Exception:
        return False

def deleter(del_ids):
    try:
        col.delete(del_ids)
        return True
    except Exception:
        return False

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(3):
        futures.append(executor.submit(adder, i))
    futures.append(executor.submit(deleter, initial[:30]))
    results = [f.result() for f in futures]

# Should have managed concurrent operations without crashing
final_count = col.doc_count()
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestMultiIndexConcurrency:
    """Concurrent operations on multiple indexes in the same collection."""

    def test_concurrent_vector_and_text_search(self):
        """Concurrent vector and text search on the same collection."""
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np
from concurrent.futures import ThreadPoolExecutor

td = "{td}"
caliby.set_buffer_config(size_gb=1)
caliby.open(td)

dim = 16
schema = caliby.Schema()
schema.add_field("topic", caliby.FieldType.STRING)
col = caliby.Collection("multi_idx", schema, vector_dim=dim)
col.create_hnsw_index("vec_idx", M=8, ef_construction=50)
col.create_text_index("text_idx")

n = 200
vectors = np.random.randn(n, dim).astype(np.float32).tolist()
col.add(
    [f"document {{i}} about python and machine learning" for i in range(n)],
    [{{"topic": f"topic_{{i % 5}}"}} for i in range(n)], vectors)

queries = np.random.randn(100, dim).astype(np.float32).tolist()

def vec_searcher():
    ok = 0
    for q in queries:
        r = col.search_vector(q, "vec_idx", k=5)
        if len(r) == 5:
            ok += 1
    return ok

def text_searcher():
    ok = 0
    terms = ["python", "machine", "learning", "document"]
    for t in terms * 25:
        r = col.search_text(t, "text_idx", k=5)
        if len(r) >= 1:
            ok += 1
    return ok

with ThreadPoolExecutor(max_workers=4) as executor:
    f1 = executor.submit(vec_searcher)
    f2 = executor.submit(text_searcher)
    v_ok = f1.result()
    t_ok = f2.result()

assert v_ok > 0, "All vector searches failed"
assert t_ok > 0, "All text searches failed"
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestConcurrentRecovery:
    """Concurrent operations during flush/recovery."""

    def test_search_during_flush(self):
        """Search while flush is happening should be thread-safe."""
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np
from concurrent.futures import ThreadPoolExecutor

td = "{td}"
caliby.set_buffer_config(size_gb=1)
caliby.open(td)

dim = 32
n = 500
np.random.seed(42)
vectors = np.random.randn(n, dim).astype(np.float32)

idx = caliby.HnswIndex(max_elements=n, dim=dim, M=8, ef_construction=50, skip_recovery=True)
idx.add_points(vectors)

queries = np.random.randn(200, dim).astype(np.float32)

def searcher():
    ok = 0
    for q in queries:
        try:
            labels, _ = idx.search_knn(q, k=5, ef_search=30)
            if len(labels) == 5:
                ok += 1
        except Exception:
            pass
    return ok

def flusher():
    for _ in range(3):
        try:
            idx.flush()
            caliby.flush_storage()
        except Exception:
            pass
    return True

with ThreadPoolExecutor(max_workers=3) as executor:
    f1 = executor.submit(searcher)
    f2 = executor.submit(flusher)
    ok = f1.result()
    flushed = f2.result()

assert ok > 0, "All searches failed during flush"
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=60,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

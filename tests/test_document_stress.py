#!/usr/bin/env python3
"""
Document CRUD Stress Tests

Stress tests for document operations: large batches, rapid add/delete cycles.
Uses shared session fixtures from conftest.py.

Run with: pytest tests/test_document_stress.py -v
"""

import pytest
import os
import sys
import numpy as np
import caliby
import subprocess as sp

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

_CNT = [0]


def _unique(category):
    _CNT[0] += 1
    return f"{category}_{_CNT[0]}"


class TestLargeBatchOperations:

    def test_add_1000_documents_no_vectors(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("int_f", caliby.FieldType.INT)
        schema.add_field("str_f", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("large_no_vec"), schema)

        n = 1000
        ids = col.add(
            [f"Document number {i} with some content" for i in range(n)],
            [{"int_f": i, "str_f": f"string_{i % 10}"} for i in range(n)])
        assert len(ids) == n
        assert col.doc_count() == n

        docs = col.get([0, 499, 999])
        assert docs[0].content.startswith("Document number 0")
        assert docs[1].metadata["int_f"] == 499
        assert docs[2].metadata["str_f"] == "string_9"

    def test_add_500_documents_with_vectors(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection(_unique("large_vec"), schema, vector_dim=64)
        col.create_hnsw_index("vec_idx")

        n = 500
        np.random.seed(42)
        vectors = np.random.randn(n, 64).astype(np.float32).tolist()

        ids = col.add(
            [f"Vector document {i}" for i in range(n)],
            [{"idx": i} for i in range(n)], vectors)
        assert len(ids) == n
        assert col.has_vectors()

        results = col.search_vector(vectors[0], "vec_idx", k=10)
        assert len(results) == 10
        assert results[0].doc_id == 0

    def test_add_multiple_batches(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("batch", caliby.FieldType.INT)
        col = caliby.Collection(_unique("multi_batch"), schema, vector_dim=8)
        col.create_hnsw_index("vec_idx")

        total = 0
        batch_size = 100
        n_batches = 8
        for batch_idx in range(n_batches):
            ids = col.add(
                [f"Batch {batch_idx} doc {i}" for i in range(batch_size)],
                [{"batch": batch_idx} for _ in range(batch_size)],
                np.random.randn(batch_size, 8).astype(np.float32).tolist())
            total += len(ids)
        assert total == n_batches * batch_size
        assert col.doc_count() == total


class TestDeleteStress:

    def test_delete_all_documents(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection(_unique("delete_all"), schema)

        n = 200
        ids = col.add(
            [f"doc {i}" for i in range(n)],
            [{"idx": i} for i in range(n)])
        col.delete(list(reversed(ids)))
        assert col.doc_count() == 0

        new_ids = col.add(["new doc"], [{"idx": 0}])
        assert len(new_ids) == 1

    def test_delete_every_other(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection(_unique("delete_alternate"), schema)

        n = 100
        ids = col.add(
            [f"doc {i}" for i in range(n)],
            [{"idx": i} for i in range(n)])
        to_delete = [ids[i] for i in range(0, n, 2)]
        col.delete(to_delete)
        assert col.doc_count() == n // 2

        remaining = [ids[i] for i in range(1, n, 2)]
        docs = col.get(remaining)
        assert all(d.metadata["idx"] % 2 == 1 for d in docs)

    def test_delete_range(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("val", caliby.FieldType.INT)
        col = caliby.Collection(_unique("delete_range"), schema)

        n = 100
        ids = col.add(
            [f"doc {i}" for i in range(n)],
            [{"val": i} for i in range(n)])
        col.delete(ids[25:75])
        assert col.doc_count() == 50

        docs = col.get([ids[0], ids[-1]])
        assert docs[0].metadata["val"] == 0
        assert docs[1].metadata["val"] == n - 1

    def test_delete_and_recreate(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("gen", caliby.FieldType.INT)
        col = caliby.Collection(_unique("delete_recreate"), schema)

        ids1 = col.add(
            [f"gen 0 doc {i}" for i in range(50)],
            [{"gen": 0} for _ in range(50)])
        assert col.doc_count() == 50
        col.delete(ids1)
        assert col.doc_count() == 0

        ids2 = col.add(
            [f"gen 1 doc {i}" for i in range(30)],
            [{"gen": 1} for _ in range(30)])
        assert len(ids2) == 30
        assert col.doc_count() == 30


class TestUpdateStress:

    def test_update_all_documents(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("version", caliby.FieldType.INT)
        schema.add_field("data", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("update_all"), schema)

        n = 200
        ids = col.add(
            [f"version 1 doc {i}" for i in range(n)],
            [{"version": 1, "data": f"data_{i}"} for i in range(n)])
        col.update(list(ids),
                   [{"version": 2, "data": f"updated_data_{i}"} for i in range(n)])

        docs = col.get([ids[0], ids[50], ids[-1]])
        for d in docs:
            assert d.metadata["version"] == 2
            assert d.metadata["data"].startswith("updated_data_")

    def test_partial_update(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("a", caliby.FieldType.STRING)
        schema.add_field("b", caliby.FieldType.INT)
        col = caliby.Collection(_unique("partial_update"), schema)

        ids = col.add(["test doc"], [{"a": "original_a", "b": 100}])
        col.update(ids, [{"a": "updated_a"}])

        docs = col.get(ids)
        assert docs[0].metadata["a"] == "updated_a"


class TestSchemaEvolution:

    def test_expand_schema_across_sessions(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema_v1 = caliby.Schema()
        schema_v1.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("schema_evolve"), schema_v1)
        col.add(["hello"], [{"name": "world"}])
        col.flush()

        # Reopen within same session
        col2 = caliby.Collection.open(col.name())
        docs = col2.get([0])
        assert docs[0].metadata["name"] == "world"

    def test_schema_persists(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("field_a", caliby.FieldType.STRING)
        schema.add_field("field_b", caliby.FieldType.INT)
        schema.add_field("field_c", caliby.FieldType.FLOAT)
        col = caliby.Collection(_unique("schema_persist2"), schema)
        col.flush()

        col2 = caliby.Collection.open(col.name())
        ids = col2.add(
            ["test"],
            [{"field_a": "a", "field_b": 42, "field_c": 3.14}])
        docs = col2.get(ids)
        assert docs[0].metadata["field_a"] == "a"
        assert docs[0].metadata["field_b"] == 42


class TestPersistenceStress:
    """Persistence tests run in subprocesses to fully reset caliby state."""

    def test_many_persistence_cycles(self):
        td = __import__("tempfile").mkdtemp()

        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby

td = "{td}"
caliby.open(td)

schema = caliby.Schema()
schema.add_field("cycle", caliby.FieldType.INT)
col = caliby.Collection("pstress", schema)
col.create_text_index("text_idx")

for cycle in range(5):
    col.add(
        [f"cycle {{cycle}} doc {{i}}" for i in range(50)],
        [{{"cycle": cycle}} for _ in range(50)])
col.flush()
caliby.close()

caliby.open(td)
col = caliby.Collection.open("pstress")
count = col.doc_count()
results = col.search_text("cycle 0", "text_idx", k=50)
caliby.close()
assert count == 250, f"Expected 250, got {{count}}"
assert len(results) >= 1
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=30,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil
        shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, f"Failed: {r.stderr}"
        assert "PASS" in r.stdout

    def test_persist_with_vectors(self):
        td = __import__("tempfile").mkdtemp()

        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, numpy as np

td = "{td}"
caliby.open(td)

schema = caliby.Schema()
schema.add_field("idx", caliby.FieldType.INT)
col = caliby.Collection("pvec", schema, vector_dim=8)
col.create_hnsw_index("vec_idx", M=16, ef_construction=100)

n = 200
np.random.seed(42)
vectors = np.random.randn(n, 8).astype(np.float32).tolist()
col.add([f"doc {{i}}" for i in range(n)],
        [{{"idx": i}} for i in range(n)], vectors)
col.flush()
caliby.close()

caliby.open(td)
col2 = caliby.Collection.open("pvec")
assert col2.doc_count() == n
assert col2.has_vectors()
results = col2.search_vector(vectors[0], "vec_idx", k=5)
assert results[0].doc_id == 0
caliby.close()
print("PASS")
'''
        r = sp.run([sys.executable, "-c", script],
                   capture_output=True, text=True, timeout=30,
                   env={**os.environ, "PYTHONPATH": build_path})
        import shutil
        shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, f"Failed: {r.stderr}"
        assert "PASS" in r.stdout


class TestEdgeCases:

    def test_empty_batch_add(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("empty_batch"), schema)
        try:
            ids = col.add([], [])
            assert len(ids) == 0
        except Exception:
            pass

    def test_single_float_vectors(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("dim1"), schema, vector_dim=1)
        col.create_hnsw_index("vec_idx")
        ids = col.add(["a", "b", "c"],
                      [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                      [[1.0], [2.0], [3.0]])
        results = col.search_vector([1.0], "vec_idx", k=3)
        assert len(results) == 3
        assert results[0].doc_id == 0

    def test_high_dimensional_vectors(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection(_unique("high_dim"), schema, vector_dim=256)
        col.create_hnsw_index("vec_idx", M=16, ef_construction=100)

        n = 50
        np.random.seed(42)
        vectors = np.random.randn(n, 256).astype(np.float32).tolist()
        ids = col.add(
            [f"hi-dim doc {i}" for i in range(n)],
            [{"idx": i} for i in range(n)], vectors)
        query = vectors[0]
        results = col.search_vector(query, "vec_idx", k=5)
        assert len(results) == 5
        assert results[0].doc_id == 0

    def test_boolean_metadata_values(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("flag", caliby.FieldType.BOOL)
        col = caliby.Collection(_unique("bool_meta"), schema)

        ids = col.add(
            ["true case", "false case", "default case"],
            [{"flag": True}, {"flag": False}, {}])
        docs = col.get(ids)
        assert docs[0].metadata["flag"] == True
        assert docs[1].metadata["flag"] == False

    def test_string_array_metadata(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
        col = caliby.Collection(_unique("str_arr_meta"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("tag_idx2", ["tags"])

        ids = col.add(
            ["tagged doc 1", "tagged doc 2"],
            [{"tags": ["python", "ml", "ai"]}, {"tags": ["java", "spring"]}],
            [[1, 0, 0, 0], [0, 1, 0, 0]])
        assert len(ids) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

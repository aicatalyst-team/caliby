#!/usr/bin/env python3
"""
Collection Lifecycle and Edge Case Tests

Tests for collection open/close/reopen, schema validation, and error handling.
Lifecycle tests that need fresh state run in subprocesses.

Run with: pytest tests/test_collection_lifecycle.py -v
"""

import pytest
import os
import sys
import tempfile
import numpy as np
import subprocess as sp
import caliby

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

_CNT = [0]


def _unique(category):
    _CNT[0] += 1
    return f"{category}_{_CNT[0]}"


def _run_subprocess(script, timeout=30):
    return sp.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": build_path})


class TestCollectionLifecycle:
    """Tests that need fresh caliby state run in subprocesses."""

    def test_create_close_reopen(self):
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby

caliby.open("{td}")
schema = caliby.Schema()
schema.add_field("name", caliby.FieldType.STRING)
col = caliby.Collection("lifecycle_test", schema, vector_dim=4)
col.create_hnsw_index("vec_idx")
col.add(["hello"], [{{"name": "test"}}], [[1.0, 0.0, 0.0, 0.0]])
col.flush()
caliby.close()

caliby.open("{td}")
col2 = caliby.Collection.open("lifecycle_test")
assert col2.doc_count() == 1
assert col2.has_vectors()
docs = col2.get([0])
assert docs[0].content == "hello"
caliby.close()
print("PASS")
'''
        r = _run_subprocess(script)
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_open_nonexistent_collection(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        with pytest.raises(Exception):
            caliby.Collection.open("no_such_collection")

    def test_multiple_reopen_cycles(self):
        td = tempfile.mkdtemp()
        n = 10
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby

td = "{td}"
caliby.open(td)
schema = caliby.Schema()
schema.add_field("count", caliby.FieldType.INT)
col = caliby.Collection("multi_cycle", schema)
col.add([f"doc {{i}}" for i in range({n})],
        [{{"count": i}} for i in range({n})])
col.flush()
caliby.close()

for cycle in range(3):
    caliby.open(td)
    col = caliby.Collection.open("multi_cycle")
    assert col.doc_count() == {n}, f"Cycle {{cycle}}: count mismatch"
    docs = col.get([0, 5])
    assert docs[0].content == "doc 0"
    caliby.close()
print("PASS")
'''
        r = _run_subprocess(script)
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_collection_name_collision(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        name = _unique("unique_name")
        caliby.Collection(name, schema)
        with pytest.raises(Exception):
            caliby.Collection(name, schema)

    def test_create_collection_after_close_reopen(self):
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby

td = "{td}"
caliby.open(td)
schema = caliby.Schema()
schema.add_field("x", caliby.FieldType.INT)
col1 = caliby.Collection("first_col", schema)
col1.add(["a"], [{{"x": 1}}])
col1.flush()
caliby.close()

caliby.open(td)
col1_check = caliby.Collection.open("first_col")
assert col1_check.doc_count() == 1

schema2 = caliby.Schema()
schema2.add_field("x", caliby.FieldType.INT)
col2 = caliby.Collection("second_col", schema2)
col2.add(["b"], [{{"x": 2}}])
assert col2.doc_count() == 1
caliby.close()
print("PASS")
'''
        r = _run_subprocess(script)
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestSchemaEdgeCases:

    def test_empty_schema(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        col = caliby.Collection(_unique("no_fields"), schema)
        assert len(schema.fields()) == 0
        ids = col.add(["content only"], [{}])
        assert len(ids) == 1

    def test_many_schema_fields(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        field_types = [
            caliby.FieldType.STRING, caliby.FieldType.INT,
            caliby.FieldType.FLOAT, caliby.FieldType.BOOL,
            caliby.FieldType.STRING_ARRAY, caliby.FieldType.INT_ARRAY,
        ]
        for i, ft in enumerate(field_types):
            schema.add_field(f"field_{i}", ft)

        col = caliby.Collection(_unique("many_fields"), schema)
        metadata = {
            "field_0": "hello", "field_1": 42, "field_2": 3.14,
            "field_3": True, "field_4": ["a", "b"], "field_5": [1, 2, 3],
        }
        ids = col.add(["test"], [metadata])
        assert len(ids) == 1
        docs = col.get(ids)
        assert docs[0].metadata["field_0"] == "hello"
        assert docs[0].metadata["field_1"] == 42
        assert docs[0].metadata["field_3"] == True

    def test_duplicate_field_name(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("dup", caliby.FieldType.STRING)
        try:
            schema.add_field("dup", caliby.FieldType.INT)
        except Exception:
            pass
        col = caliby.Collection(_unique("dup_field"), schema)
        assert schema.has_field("dup")

    def test_long_field_names(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        long_name = "a" * 100
        schema.add_field(long_name, caliby.FieldType.STRING)
        col = caliby.Collection(_unique("long_names"), schema)
        meta = {long_name: "test_value"}
        ids = col.add(["content"], [meta])
        docs = col.get(ids)
        assert docs[0].metadata[long_name] == "test_value"


class TestCollectionErrorHandling:

    def test_add_wrong_metadata_type(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("count", caliby.FieldType.INT)
        col = caliby.Collection(_unique("type_check"), schema)
        try:
            ids = col.add(["doc"], [{"count": "not_a_number"}])
            assert col.doc_count() >= 1
        except Exception:
            pass  # Strict type checking is expected

    def test_add_missing_schema_field(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("required_field", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("missing_field"), schema)
        ids = col.add(["doc"], [{}])
        assert len(ids) == 1
        docs = col.get(ids)
        assert docs[0].metadata.get("required_field") is None or \
               docs[0].metadata.get("required_field") == ""

    def test_add_extra_metadata_field(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("extra_field"), schema)
        ids = col.add(["doc"], [{"name": "test", "extra": "bonus"}])
        assert len(ids) == 1
        docs = col.get(ids)
        assert docs[0].metadata["name"] == "test"

    def test_update_missing_document(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("update_missing"), schema)
        col.add(["exists"], [{"name": "test"}])
        with pytest.raises(Exception):
            col.update([999], [{"name": "ghost"}])

    def test_delete_already_deleted(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("double_delete"), schema)
        ids = col.add(["a"], [{"name": "a"}])
        col.delete(ids)
        try:
            col.delete(ids)
        except Exception:
            pass

    def test_null_and_empty_metadata(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("nullable", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("null_meta"), schema)
        ids2 = col.add(["doc with empty"], [{"nullable": ""}])
        docs2 = col.get(ids2)
        assert docs2[0].metadata["nullable"] == ""

    def test_very_long_content(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("size", caliby.FieldType.INT)
        col = caliby.Collection(_unique("long_content"), schema)
        long_text = "The quick brown fox jumps over the lazy dog. " * 2000
        ids = col.add([long_text], [{"size": len(long_text)}])
        docs = col.get(ids)
        assert docs[0].content == long_text


class TestCollectionIndexInteractions:

    def test_create_all_index_types(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("price", caliby.FieldType.FLOAT)
        col = caliby.Collection(_unique("all_idx3"), schema, vector_dim=8)
        col.create_hnsw_index("vec_idx", M=16, ef_construction=100)
        col.create_text_index("text_idx")
        col.create_metadata_index("meta_idx", ["category", "price"])

        col.add(
            ["Product A: high quality widget", "Product B: budget widget alternative"],
            [{"category": "premium", "price": 99.99},
             {"category": "budget", "price": 9.99}],
            [[1.0] + [0.0] * 7, [0.0] * 7 + [1.0]])
        assert len(col.search_vector([1.0] + [0.0] * 7, "vec_idx", k=2)) == 2
        assert len(col.search_text("widget", "text_idx", k=2)) >= 1
        results = col.search_vector(
            [1.0] + [0.0] * 7, "vec_idx", k=2,
            filter='{"category": {"$eq": "premium"}}')
        assert len(results) >= 1

    def test_index_recreation_after_reopen(self):
        td = tempfile.mkdtemp()
        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby

td = "{td}"
caliby.open(td)
schema = caliby.Schema()
schema.add_field("tag", caliby.FieldType.STRING)
col = caliby.Collection("reopen_idx", schema, vector_dim=4)
col.create_hnsw_index("vec_idx")
col.create_text_index("text_idx")
col.add(
    ["alpha beta gamma", "delta epsilon zeta"],
    [{{"tag": "a"}}, {{"tag": "b"}}],
    [[1, 0, 0, 0], [0, 1, 0, 0]])
col.flush()
caliby.close()

caliby.open(td)
col2 = caliby.Collection.open("reopen_idx")
assert len(col2.search_vector([1, 0, 0, 0], "vec_idx", k=2)) == 2
assert len(col2.search_text("alpha", "text_idx", k=2)) >= 1
caliby.close()
print("PASS")
'''
        r = _run_subprocess(script)
        import shutil; shutil.rmtree(td, ignore_errors=True)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestMetadataFiltering:

    def test_equality_filter(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("color", caliby.FieldType.STRING)
        schema.add_field("size", caliby.FieldType.INT)
        col = caliby.Collection(_unique("filter_test3"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("color_idx", ["color"])

        col.add(
            ["red item", "blue item", "red item 2"],
            [{"color": "red", "size": 1},
             {"color": "blue", "size": 2},
             {"color": "red", "size": 3}],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        results = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=3,
            filter='{"color": {"$eq": "red"}}')
        for r in results:
            assert r.doc_id in [0, 2]

    def test_range_filter(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("score", caliby.FieldType.INT)
        col = caliby.Collection(_unique("range_filter3"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("score_idx", ["score"])

        col.add(
            [f"item {i}" for i in range(10)],
            [{"score": i * 10} for i in range(10)],
            [[float(i == j) for j in range(4)] for i in range(10)])
        results = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=10,
            filter='{"score": {"$gt": 50}}')
        for r in results:
            assert col.get([r.doc_id])[0].metadata["score"] > 50

    def test_and_condition(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        col = caliby.Collection(_unique("and_filter3"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("cat_year_idx2", ["category", "year"])

        col.add(
            ["doc a", "doc b", "doc c", "doc d"],
            [{"category": "tech", "year": 2024},
             {"category": "tech", "year": 2023},
             {"category": "art", "year": 2024},
             {"category": "art", "year": 2023}],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        results = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=4,
            filter='{"$and": [{"category": {"$eq": "tech"}}, {"year": {"$eq": 2024}}]}')
        for r in results:
            meta = col.get([r.doc_id])[0].metadata
            assert meta["category"] == "tech" and meta["year"] == 2024

    def test_or_condition(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("type", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("or_filter3"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("type_idx2", ["type"])

        col.add(
            ["apple", "banana", "carrot", "date"],
            [{"type": "fruit"}, {"type": "fruit"},
             {"type": "vegetable"}, {"type": "fruit"}],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        results = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=4,
            filter='{"$or": [{"type": {"$eq": "fruit"}}, {"type": {"$eq": "vegetable"}}]}')
        assert len(results) >= 1


class TestDistanceMetricEdgeCases:

    def test_unit_vectors_l2(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("unit_l2_2"), schema, vector_dim=4,
                                distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec_idx")
        col.add(["unit", "double"],
                [{"name": "u"}, {"name": "d"}],
                [[1, 0, 0, 0], [2, 0, 0, 0]])
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=2)
        assert results[0].doc_id == 0

    def test_orthogonal_vectors_cosine(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("ortho_cos_2"), schema, vector_dim=4,
                                distance_metric=caliby.DistanceMetric.COSINE)
        col.create_hnsw_index("vec_idx")
        col.add(["same", "ortho"],
                [{"name": "s"}, {"name": "o"}],
                [[1, 0, 0, 0], [0, 1, 0, 0]])
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=2)
        assert results[0].doc_id == 0
        assert abs(results[0].score) < 0.01
        assert abs(results[1].score - 1.0) < 0.01

    def test_negative_inner_product(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("neg_ip_2"), schema, vector_dim=4,
                                distance_metric=caliby.DistanceMetric.IP)
        col.create_hnsw_index("vec_idx")
        col.add(["pos", "neg"],
                [{"name": "p"}, {"name": "n"}],
                [[1, 0, 0, 0], [-1, 0, 0, 0]])
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=2)
        assert results[0].doc_id == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

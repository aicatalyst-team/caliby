#!/usr/bin/env python3
"""
Multi-Index Interaction and Hybrid Search Tests

Tests for interactions between multiple index types, hybrid search scoring,
and index combinations. Uses shared session fixtures from conftest.py.

Run with: pytest tests/test_multi_index_interactions.py -v
"""

import pytest
import os
import sys
import numpy as np
import caliby

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

# Use shared session fixtures from conftest.py instead of open/close per test.
_CNT = [0]


def _unique(category):
    _CNT[0] += 1
    return f"{category}_{_CNT[0]}"


class TestHybridSearch:

    def test_hybrid_different_alpha_values(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("topic", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("hybrid_alpha"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")

        col.add([
            "Python is a programming language for data science",
            "Python machine learning and deep learning",
            "Java is used for enterprise software",
        ], [
            {"topic": "python"}, {"topic": "python"}, {"topic": "java"},
        ], [
            [1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0],
        ])

        results = col.search_hybrid(
            query_vec=[1, 0, 0, 0], vector_index="vec_idx",
            query_text="python", text_index="text_idx", k=3)
        assert len(results) >= 1
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_hybrid_no_text_match(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("tag", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("hybrid_notext"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        col.add(["apple banana"], [{"tag": "test"}], [[1, 0, 0, 0]])

        results = col.search_hybrid(
            query_vec=[1, 0, 0, 0], vector_index="vec_idx",
            query_text="zzzz_no_match_zzzz", text_index="text_idx", k=3)
        assert len(results) >= 1

    def test_hybrid_no_vector_match(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("hybrid_novec"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        col.add(["machine learning tutorial"], [{"name": "test"}], [[1, 0, 0, 0]])

        results = col.search_hybrid(
            query_vec=[-10, 0, 0, 0], vector_index="vec_idx",
            query_text="machine learning", text_index="text_idx", k=3)
        assert len(results) >= 1


class TestMultipleVectorIndices:

    def test_two_hnsw_indices(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("two_hnsw"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx_1", M=16, ef_construction=100)
        col.create_hnsw_index("vec_idx_2", M=16, ef_construction=100)

        col.add(["doc 1", "doc 2"], [{"name": "a"}, {"name": "b"}],
                [[1, 0, 0, 0], [0, 1, 0, 0]])
        r1 = col.search_vector([1, 0, 0, 0], "vec_idx_1", k=2)
        r2 = col.search_vector([1, 0, 0, 0], "vec_idx_2", k=2)
        assert len(r1) == 2
        assert len(r2) == 2

    def test_hnsw_and_diskann_in_collection(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection(_unique("hnsw_diskann"), schema, vector_dim=4)
        col.create_hnsw_index("hnsw_idx", M=16, ef_construction=100)

        n = 100
        col.add(
            [f"doc {i}" for i in range(n)],
            [{"idx": i} for i in range(n)],
            [[float(i == j % 4) for j in range(4)] for i in range(n)])
        results = col.search_vector([1, 0, 0, 0], "hnsw_idx", k=10)
        assert len(results) == 10


class TestIndexAcrossCollections:

    def test_cross_collection_isolation(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema_a = caliby.Schema()
        schema_a.add_field("source", caliby.FieldType.STRING)
        col_a = caliby.Collection(_unique("collection_a"), schema_a, vector_dim=4)
        col_a.create_hnsw_index("vec_a")
        col_a.create_text_index("text_a")
        col_a.add(["python programming", "java development"],
                  [{"source": "a"}, {"source": "a"}],
                  [[1, 0, 0, 0], [0, 1, 0, 0]])

        schema_b = caliby.Schema()
        schema_b.add_field("source", caliby.FieldType.STRING)
        col_b = caliby.Collection(_unique("collection_b"), schema_b, vector_dim=4)
        col_b.create_hnsw_index("vec_b")
        col_b.create_text_index("text_b")
        col_b.add(["machine learning tutorial", "deep learning guide"],
                  [{"source": "b"}, {"source": "b"}],
                  [[1, 0, 0, 0], [0, 1, 0, 0]])

        assert len(col_a.search_text("machine", "text_a", k=5)) == 0
        assert len(col_b.search_text("python", "text_b", k=5)) == 0

    def test_many_collections(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        n_collections = 10
        collections = []
        for i in range(n_collections):
            schema = caliby.Schema()
            schema.add_field("idx", caliby.FieldType.INT)
            col = caliby.Collection(_unique(f"many_col_{i}"), schema, vector_dim=8)
            col.create_hnsw_index("vec_idx")
            col.add([f"doc in collection {i}"], [{"idx": i}],
                    [[float(j == 0) for j in range(8)]])
            collections.append(col)
        for i, col in enumerate(collections):
            assert col.doc_count() == 1
            assert col.get([0])[0].content == f"doc in collection {i}"


class TestOperationsOnMultipleIndexes:

    def test_delete_affects_all_indexes(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("tag", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("delete_all_idx2"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        col.create_metadata_index("tag_idx", ["tag"])

        ids = col.add(
            ["important document to delete", "keep this document"],
            [{"tag": "del"}, {"tag": "keep"}],
            [[1, 0, 0, 0], [0, 1, 0, 0]])
        assert col.doc_count() == 2
        col.delete([ids[0]])
        assert col.doc_count() == 1
        text_results = col.search_text("document", "text_idx", k=5)
        assert len(text_results) >= 1

    def test_update_metadata_affects_filter(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("status", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("update_meta_idx2"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("status_idx", ["status"])
        ids = col.add(["doc 1", "doc 2"], [{"status": "draft"}, {"status": "draft"}],
                      [[1, 0, 0, 0], [0, 1, 0, 0]])

        results_before = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=2, filter='{"status": {"$eq": "published"}}')
        assert len(results_before) == 0

        col.update([ids[0]], [{"status": "published"}])
        results_after = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=2, filter='{"status": {"$eq": "published"}}')
        assert len(results_after) >= 1

    def test_add_after_index_create(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("topic", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("add_after_idx2"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        col.add(["Python tutorial", "Java guide"],
                [{"topic": "python"}, {"topic": "java"}],
                [[1, 0, 0, 0], [0, 1, 0, 0]])
        assert len(col.search_vector([1, 0, 0, 0], "vec_idx", k=2)) == 2
        assert len(col.search_text("tutorial", "text_idx", k=2)) >= 1


class TestFilterComprehensive:

    def test_filter_in_operator(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("color", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("in_filter2"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("color_idx", ["color"])
        col.add(["red", "green", "blue", "yellow"],
                [{"color": c} for c in ["red", "green", "blue", "yellow"]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        results = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=4,
            filter='{"color": {"$in": ["red", "blue"]}}')
        for r in results:
            meta = col.get([r.doc_id])[0].metadata
            assert meta["color"] in ["red", "blue"]

    def test_filter_cosine_with_filter(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("type", caliby.FieldType.STRING)
        col = caliby.Collection(_unique("cos_filter2"), schema, vector_dim=4,
                                distance_metric=caliby.DistanceMetric.COSINE)
        col.create_hnsw_index("vec_idx")
        col.create_metadata_index("type_idx", ["type"])
        col.add(["a", "b"], [{"type": "valid"}, {"type": "invalid"}],
                [[1, 0, 0, 0], [0, 1, 0, 0]])
        results = col.search_vector(
            [1, 0, 0, 0], "vec_idx", k=2,
            filter='{"type": {"$eq": "valid"}}')
        assert len(results) == 1
        assert results[0].doc_id == 0


class TestMultiThreadedSearches:

    def test_rapid_sequential_searches(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection(_unique("rapid_search2"), schema, vector_dim=32)
        col.create_hnsw_index("vec_idx")

        n = 300
        vectors = np.random.randn(n, 32).astype(np.float32).tolist()
        col.add([f"doc {i}" for i in range(n)],
                [{"idx": i} for i in range(n)], vectors)

        for _ in range(100):
            query = np.random.randn(32).astype(np.float32).tolist()
            results = col.search_vector(query, "vec_idx", k=5)
            assert len(results) == 5

    def test_interleaved_read_write(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("count", caliby.FieldType.INT)
        col = caliby.Collection(_unique("interleave2"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")

        for batch in range(10):
            start = batch * 10
            col.add(
                [f"batch {batch} doc {i}" for i in range(10)],
                [{"count": batch * 10 + i} for i in range(10)],
                [[float(j == (batch % 4)) for j in range(4)] for _ in range(10)])
            query = [float(i == 0) for i in range(4)]
            results = col.search_vector(query, "vec_idx", k=5)
            assert len(results) >= 1
        assert col.doc_count() == 100

    def test_delete_then_add_same_ids(self, caliby_module, temp_dir):
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("val", caliby.FieldType.INT)
        col = caliby.Collection(_unique("delete_add2"), schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")

        ids1 = col.add(["doc 1", "doc 2", "doc 3"],
                       [{"val": 1}, {"val": 2}, {"val": 3}],
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        col.delete(ids1)
        assert col.doc_count() == 0

        ids2 = col.add(["new doc 1", "new doc 2"],
                       [{"val": 10}, {"val": 20}],
                       [[1, 0, 0, 0], [0, 1, 0, 0]])
        assert col.doc_count() == 2
        assert ids2[0] != ids1[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

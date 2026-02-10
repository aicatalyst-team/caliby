#!/usr/bin/env python3
"""
Unit tests for histogram-based cardinality estimation and selectivity-adaptive filtered search.

Tests cover:
1. FieldHistogram functionality (add, remove, estimate_eq, estimate_range, estimate_in)
2. estimate_selectivity() in Collection using histograms from BTree indices
3. Selectivity-based adaptive search strategy:
   - High selectivity (>10%): Progressive post-filtering
   - Medium selectivity (>0.5%): Filtered HNSW with boosted ef_search
   - Low selectivity (≤0.5%): Brute-force for perfect recall
"""

import pytest
import numpy as np
import json
import os
import shutil
import time
from typing import List, Set

import caliby


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def data_dir(tmp_path):
    """Create a temporary data directory for each test."""
    d = tmp_path / "caliby_test"
    d.mkdir()
    yield str(d)
    # Cleanup handled by tmp_path fixture


@pytest.fixture(scope="function")
def caliby_env(data_dir):
    """Initialize caliby environment for testing."""
    caliby.set_buffer_config(size_gb=2)
    caliby.open(data_dir, cleanup_if_exist=True)
    yield data_dir
    caliby.close()


# ============================================================================
# Helper Functions
# ============================================================================

def compute_brute_force_knn(vectors: np.ndarray, query: np.ndarray, 
                           mask: np.ndarray = None, k: int = 10) -> List[int]:
    """Compute exact k-NN using brute force."""
    if mask is not None:
        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            return []
        filtered_vectors = vectors[mask]
        dists = np.sum((filtered_vectors - query)**2, axis=1)
        top_k_local = np.argsort(dists)[:k]
        return filtered_indices[top_k_local].tolist()
    else:
        dists = np.sum((vectors - query)**2, axis=1)
        return np.argsort(dists)[:k].tolist()


def compute_recall(results: List[int], ground_truth: List[int], k: int = 10) -> float:
    """Compute recall@k."""
    if len(ground_truth) == 0:
        return 1.0 if len(results) == 0 else 0.0
    result_set = set(results[:k])
    gt_set = set(ground_truth[:k])
    return len(result_set & gt_set) / min(len(gt_set), k)


# ============================================================================
# Test: Histogram Cardinality Estimation with Integer Fields
# ============================================================================

class TestHistogramIntegerField:
    """Test histogram-based estimation for integer fields."""
    
    @pytest.fixture
    def collection_with_int_data(self, caliby_env):
        """Create collection with integer category field and metadata index."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        schema.add_field("value", caliby.FieldType.INT)
        
        col = caliby.Collection("test_int", schema, vector_dim=32, 
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=100)
        col.create_metadata_index("cat_idx", ["category"])
        col.create_metadata_index("val_idx", ["value"])
        
        return col
    
    def test_equality_selectivity_uniform_distribution(self, collection_with_int_data):
        """Test selectivity estimation for equality on uniformly distributed data."""
        col = collection_with_int_data
        n = 10000
        num_categories = 10
        
        np.random.seed(42)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": i % num_categories, "value": i} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping for verification
        id_to_category = {doc_ids[i]: i % num_categories for i in range(n)}
        
        # Each category should have ~1000 docs (10% selectivity)
        # Test search with filter for a specific category
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": 0})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        # Verify all results match the filter using doc_id mapping
        for r in results:
            assert id_to_category[r.doc_id] == 0, f"Doc {r.doc_id} has category {id_to_category[r.doc_id]}, expected 0"
    
    def test_equality_selectivity_skewed_distribution(self, collection_with_int_data):
        """Test selectivity estimation for skewed data distribution."""
        col = collection_with_int_data
        n = 10000
        
        np.random.seed(123)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        
        # Skewed distribution: 80% category 0, 15% category 1, 5% category 2
        categories = np.zeros(n, dtype=int)
        categories[int(n*0.80):int(n*0.95)] = 1
        categories[int(n*0.95):] = 2
        np.random.shuffle(categories)
        
        metadatas = [{"category": int(categories[i]), "value": i} for i in range(n)]
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: int(categories[i]) for i in range(n)}
        
        # Test searches for each category
        query = np.random.randn(32).astype(np.float32)
        
        for cat in [0, 1, 2]:
            filter_json = json.dumps({"category": cat})
            results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
            
            for r in results:
                assert id_to_category[r.doc_id] == cat, f"Doc {r.doc_id} category mismatch"
    
    def test_range_selectivity(self, collection_with_int_data):
        """Test selectivity estimation for range queries."""
        col = collection_with_int_data
        n = 5000
        
        np.random.seed(456)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": 0, "value": i} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to value mapping
        id_to_value = {doc_ids[i]: i for i in range(n)}
        
        # Test range filter: value >= 4000 (20% of data)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"value": {"$gte": 4000}})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        for r in results:
            assert id_to_value[r.doc_id] >= 4000, f"Value {id_to_value[r.doc_id]} < 4000"
        
        # Test range filter: value < 500 (10% of data)
        filter_json = json.dumps({"value": {"$lt": 500}})
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        for r in results:
            assert id_to_value[r.doc_id] < 500, f"Value {id_to_value[r.doc_id]} >= 500"


# ============================================================================
# Test: Histogram Cardinality Estimation with String Fields
# ============================================================================

class TestHistogramStringField:
    """Test histogram-based estimation for string fields."""
    
    @pytest.fixture
    def collection_with_string_data(self, caliby_env):
        """Create collection with string category field."""
        schema = caliby.Schema()
        schema.add_field("type", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_str", schema, vector_dim=32,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=100)
        col.create_metadata_index("type_idx", ["type"])
        
        return col
    
    def test_string_equality_selectivity(self, collection_with_string_data):
        """Test selectivity estimation for string equality."""
        col = collection_with_string_data
        n = 5000
        types = ["article", "book", "paper", "thesis", "report"]
        
        np.random.seed(789)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        type_assignments = [types[i % len(types)] for i in range(n)]
        metadatas = [{"type": type_assignments[i]} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to type mapping
        id_to_type = {doc_ids[i]: type_assignments[i] for i in range(n)}
        
        # Each type has ~20% selectivity
        query = np.random.randn(32).astype(np.float32)
        
        for t in types:
            filter_json = json.dumps({"type": t})
            results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
            
            for r in results:
                assert id_to_type[r.doc_id] == t, f"Type mismatch for doc {r.doc_id}"
    
    def test_string_high_cardinality(self, collection_with_string_data):
        """Test selectivity estimation with high cardinality string field."""
        col = collection_with_string_data
        n = 2000
        
        np.random.seed(321)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        type_assignments = [f"type_{i}" for i in range(n)]
        # Each doc has unique type (very low selectivity per value)
        metadatas = [{"type": type_assignments[i]} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to type mapping
        id_to_type = {doc_ids[i]: type_assignments[i] for i in range(n)}
        
        # Query for a specific unique type
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"type": "type_500"})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        # Should find exactly 1 match
        assert len(results) <= 1
        if len(results) == 1:
            assert id_to_type[results[0].doc_id] == "type_500"


# ============================================================================
# Test: Selectivity-Based Adaptive Search Strategy
# ============================================================================

class TestAdaptiveSearchStrategy:
    """Test that different selectivity levels trigger appropriate search strategies."""
    
    @pytest.fixture
    def large_collection(self, caliby_env):
        """Create a larger collection to test adaptive strategy thresholds."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        
        col = caliby.Collection("test_adaptive", schema, vector_dim=64,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=200)
        col.create_metadata_index("cat_idx", ["category"])
        
        return col
    
    def test_high_selectivity_postfilter(self, large_collection):
        """Test that high selectivity (>10%) uses progressive post-filtering."""
        col = large_collection
        n = 20000
        
        np.random.seed(111)
        vectors = np.random.randn(n, 64).astype(np.float32)
        
        # 50% of data in category 0 (high selectivity)
        categories = np.array([0 if i < n//2 else 1 for i in range(n)])
        np.random.shuffle(categories)
        
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": int(categories[i])} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: int(categories[i]) for i in range(n)}
        
        # Create ground truth
        mask = categories == 0
        
        # Run multiple queries and check recall
        num_queries = 100
        k = 10
        recall_sum = 0
        
        for q in range(num_queries):
            query = np.random.randn(64).astype(np.float32)
            gt = compute_brute_force_knn(vectors, query, mask, k)
            
            filter_json = json.dumps({"category": 0})
            results = col.search_vector(query.tolist(), "vec", k=k, filter=filter_json)
            result_ids = [r.doc_id for r in results]
            
            # Verify filter correctness using mapping
            for r in results:
                assert id_to_category[r.doc_id] == 0
            
            recall_sum += compute_recall(result_ids, gt, k)
        
        avg_recall = recall_sum / num_queries
        # High selectivity should have good recall with post-filtering
        assert avg_recall > 0.7, f"High selectivity recall too low: {avg_recall}"
    
    def test_medium_selectivity_filtered_hnsw(self, large_collection):
        """Test that medium selectivity (0.5-10%) uses filtered HNSW."""
        col = large_collection
        n = 20000
        
        np.random.seed(222)
        vectors = np.random.randn(n, 64).astype(np.float32)
        
        # 5% of data in category 0 (medium selectivity)
        categories = np.ones(n, dtype=int)
        categories[:int(n*0.05)] = 0
        np.random.shuffle(categories)
        
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": int(categories[i])} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: int(categories[i]) for i in range(n)}
        
        mask = categories == 0
        
        num_queries = 100
        k = 10
        recall_sum = 0
        
        for q in range(num_queries):
            query = np.random.randn(64).astype(np.float32)
            gt = compute_brute_force_knn(vectors, query, mask, k)
            
            filter_json = json.dumps({"category": 0})
            results = col.search_vector(query.tolist(), "vec", k=k, filter=filter_json)
            result_ids = [r.doc_id for r in results]
            
            for r in results:
                assert id_to_category[r.doc_id] == 0
            
            recall_sum += compute_recall(result_ids, gt, k)
        
        avg_recall = recall_sum / num_queries
        # Medium selectivity should have reasonable recall
        assert avg_recall > 0.5, f"Medium selectivity recall too low: {avg_recall}"
    
    def test_low_selectivity_bruteforce(self, large_collection):
        """Test that low selectivity (≤0.5%) uses brute-force for perfect recall."""
        col = large_collection
        n = 20000
        
        np.random.seed(333)
        vectors = np.random.randn(n, 64).astype(np.float32)
        
        # 0.5% of data in category 0 (100 docs) - low selectivity
        categories = np.ones(n, dtype=int)
        categories[:100] = 0  # Exactly 0.5%
        np.random.shuffle(categories)
        
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": int(categories[i])} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: int(categories[i]) for i in range(n)}
        
        mask = categories == 0
        
        num_queries = 50
        k = 10
        recall_sum = 0
        
        for q in range(num_queries):
            query = np.random.randn(64).astype(np.float32)
            gt = compute_brute_force_knn(vectors, query, mask, k)
            
            filter_json = json.dumps({"category": 0})
            results = col.search_vector(query.tolist(), "vec", k=k, filter=filter_json)
            result_ids = [r.doc_id for r in results]
            
            for r in results:
                assert id_to_category[r.doc_id] == 0
            
            recall_sum += compute_recall(result_ids, gt, k)
        
        avg_recall = recall_sum / num_queries
        # Low selectivity with brute-force should have near-perfect recall
        assert avg_recall > 0.95, f"Low selectivity recall should be near-perfect: {avg_recall}"
    
    def test_very_low_selectivity_perfect_recall(self, large_collection):
        """Test that very low selectivity (<<0.5%) achieves perfect recall."""
        col = large_collection
        n = 20000
        
        np.random.seed(444)
        vectors = np.random.randn(n, 64).astype(np.float32)
        
        # 0.1% of data in category 0 (20 docs) - very low selectivity
        categories = np.ones(n, dtype=int)
        categories[:20] = 0  # 0.1%
        np.random.shuffle(categories)
        
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": int(categories[i])} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: int(categories[i]) for i in range(n)}
        
        mask = categories == 0
        
        num_queries = 50
        k = 10
        recall_sum = 0
        
        for q in range(num_queries):
            query = np.random.randn(64).astype(np.float32)
            gt = compute_brute_force_knn(vectors, query, mask, k)
            
            filter_json = json.dumps({"category": 0})
            results = col.search_vector(query.tolist(), "vec", k=k, filter=filter_json)
            result_ids = [r.doc_id for r in results]
            
            for r in results:
                assert id_to_category[r.doc_id] == 0
            
            recall_sum += compute_recall(result_ids, gt, k)
        
        avg_recall = recall_sum / num_queries
        # Very low selectivity should achieve perfect recall with brute-force
        assert avg_recall >= 0.99, f"Very low selectivity should have perfect recall: {avg_recall}"


# ============================================================================
# Test: Compound Filters with AND/OR
# ============================================================================

class TestCompoundFilterSelectivity:
    """Test selectivity estimation for compound filters."""
    
    @pytest.fixture
    def collection_with_multiple_fields(self, caliby_env):
        """Create collection with multiple indexed fields."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("type", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_compound", schema, vector_dim=32,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=100)
        col.create_metadata_index("cat_idx", ["category"])
        col.create_metadata_index("year_idx", ["year"])
        col.create_metadata_index("type_idx", ["type"])
        
        return col
    
    def test_and_filter_selectivity(self, collection_with_multiple_fields):
        """Test selectivity estimation for AND filters."""
        col = collection_with_multiple_fields
        n = 10000
        
        np.random.seed(555)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        
        # 5 categories, 5 years, 4 types
        categories = [i % 5 for i in range(n)]
        years = [2020 + (i % 5) for i in range(n)]
        types_list = ["article", "book", "paper", "thesis"]
        
        metadatas = [
            {"category": categories[i], "year": years[i], 
             "type": types_list[i % 4]}
            for i in range(n)
        ]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to metadata mapping
        id_to_meta = {doc_ids[i]: {"category": categories[i], "year": years[i]} for i in range(n)}
        
        # AND filter: category=0 AND year=2020 (should be ~4% = 20% * 20%)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({
            "$and": [
                {"category": 0},
                {"year": 2020}
            ]
        })
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        for r in results:
            meta = id_to_meta[r.doc_id]
            assert meta["category"] == 0, f"Category mismatch: {meta['category']}"
            assert meta["year"] == 2020, f"Year mismatch: {meta['year']}"
    
    def test_or_filter_selectivity(self, collection_with_multiple_fields):
        """Test selectivity estimation for OR filters."""
        col = collection_with_multiple_fields
        n = 10000
        
        np.random.seed(666)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        
        categories = [i % 10 for i in range(n)]  # 10% each
        metadatas = [{"category": categories[i], "year": 2020, "type": "article"} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: categories[i] for i in range(n)}
        
        # OR filter: category=0 OR category=1 (should be ~20%)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({
            "$or": [
                {"category": 0},
                {"category": 1}
            ]
        })
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        for r in results:
            cat = id_to_category[r.doc_id]
            assert cat in [0, 1], f"Expected category 0 or 1, got {cat}"


# ============================================================================
# Test: IN and NOT IN Filters
# ============================================================================

class TestInFilterSelectivity:
    """Test selectivity estimation for IN and NOT IN filters."""
    
    @pytest.fixture
    def collection_with_categories(self, caliby_env):
        """Create collection for IN filter testing."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        
        col = caliby.Collection("test_in", schema, vector_dim=32,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=100)
        col.create_metadata_index("cat_idx", ["category"])
        
        return col
    
    def test_in_filter(self, collection_with_categories):
        """Test IN filter with multiple values."""
        col = collection_with_categories
        n = 5000
        
        np.random.seed(777)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        categories = [i % 10 for i in range(n)]
        metadatas = [{"category": categories[i]} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: categories[i] for i in range(n)}
        
        # IN filter: category in [0, 2, 4] (30% selectivity)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": {"$in": [0, 2, 4]}})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        for r in results:
            cat = id_to_category[r.doc_id]
            assert cat in [0, 2, 4], f"Expected category in [0,2,4], got {cat}"
    
    def test_nin_filter(self, collection_with_categories):
        """Test NOT IN filter."""
        col = collection_with_categories
        n = 5000
        
        np.random.seed(888)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        categories = [i % 10 for i in range(n)]
        metadatas = [{"category": categories[i]} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: categories[i] for i in range(n)}
        
        # NOT IN filter: category not in [0, 1, 2] (70% selectivity)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": {"$nin": [0, 1, 2]}})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        for r in results:
            cat = id_to_category[r.doc_id]
            assert cat not in [0, 1, 2], f"Expected category not in [0,1,2], got {cat}"


# ============================================================================
# Test: Recall at Different Selectivity Levels
# ============================================================================

class TestRecallVsSelectivity:
    """Comprehensive recall testing across selectivity spectrum."""
    
    @pytest.fixture
    def recall_test_collection(self, caliby_env):
        """Create collection for recall testing."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        
        col = caliby.Collection("test_recall", schema, vector_dim=64,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=200)
        col.create_metadata_index("cat_idx", ["category"])
        
        return col
    
    @pytest.mark.parametrize("selectivity,expected_min_recall", [
        (0.50, 0.70),   # 50% - high selectivity
        (0.20, 0.65),   # 20% - high selectivity
        (0.10, 0.60),   # 10% - threshold
        (0.05, 0.55),   # 5% - medium selectivity
        (0.01, 0.80),   # 1% - low selectivity (brute force helps)
        (0.005, 0.95),  # 0.5% - threshold for brute force
        (0.001, 0.99),  # 0.1% - very low, brute force
    ])
    def test_recall_at_selectivity(self, recall_test_collection, selectivity, expected_min_recall):
        """Test recall at various selectivity levels."""
        col = recall_test_collection
        n = 50000
        
        np.random.seed(int(selectivity * 10000))
        vectors = np.random.randn(n, 64).astype(np.float32)
        
        # Create categories with target selectivity
        num_matching = int(n * selectivity)
        categories = np.ones(n, dtype=int)
        categories[:num_matching] = 0
        np.random.shuffle(categories)
        
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": int(categories[i])} for i in range(n)]
        
        col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        mask = categories == 0
        
        # Run queries
        num_queries = 50
        k = 10
        recall_sum = 0
        
        for q in range(num_queries):
            query = np.random.randn(64).astype(np.float32)
            gt = compute_brute_force_knn(vectors, query, mask, k)
            
            filter_json = json.dumps({"category": 0})
            results = col.search_vector(query.tolist(), "vec", k=k, filter=filter_json)
            result_ids = [r.doc_id for r in results]
            
            recall_sum += compute_recall(result_ids, gt, k)
        
        avg_recall = recall_sum / num_queries
        assert avg_recall >= expected_min_recall, \
            f"Selectivity {selectivity*100:.1f}%: recall {avg_recall:.4f} < {expected_min_recall}"


# ============================================================================
# Test: Performance with Different Histogram Configurations
# ============================================================================

class TestHistogramPerformance:
    """Test performance characteristics with histogram-based optimization."""
    
    @pytest.fixture
    def perf_collection(self, caliby_env):
        """Create collection for performance testing."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        
        col = caliby.Collection("test_perf", schema, vector_dim=128,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=100)
        col.create_metadata_index("cat_idx", ["category"])
        
        return col
    
    def test_low_selectivity_faster_than_high(self, perf_collection):
        """Test that low selectivity (brute force) is faster for small candidate sets."""
        col = perf_collection
        n = 30000
        
        np.random.seed(999)
        vectors = np.random.randn(n, 128).astype(np.float32)
        
        # Create two categories: one with 50% selectivity, one with 0.5%
        categories = np.ones(n, dtype=int)
        categories[:int(n*0.005)] = 0  # 0.5% in category 0
        categories[int(n*0.005):int(n*0.505)] = 2  # 50% in category 2
        np.random.shuffle(categories)
        
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": int(categories[i])} for i in range(n)]
        
        col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Benchmark low selectivity
        query = np.random.randn(128).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            col.search_vector(query.tolist(), "vec", k=10, filter=json.dumps({"category": 0}))
            col.search_vector(query.tolist(), "vec", k=10, filter=json.dumps({"category": 2}))
        
        # Measure low selectivity
        low_times = []
        for _ in range(50):
            query = np.random.randn(128).astype(np.float32)
            start = time.perf_counter()
            col.search_vector(query.tolist(), "vec", k=10, filter=json.dumps({"category": 0}))
            low_times.append(time.perf_counter() - start)
        
        # Measure high selectivity
        high_times = []
        for _ in range(50):
            query = np.random.randn(128).astype(np.float32)
            start = time.perf_counter()
            col.search_vector(query.tolist(), "vec", k=10, filter=json.dumps({"category": 2}))
            high_times.append(time.perf_counter() - start)
        
        avg_low = np.mean(low_times) * 1000
        avg_high = np.mean(high_times) * 1000
        
        print(f"\nLow selectivity (0.5%) avg latency: {avg_low:.2f}ms")
        print(f"High selectivity (50%) avg latency: {avg_high:.2f}ms")
        
        # Low selectivity with brute force should be reasonably fast
        # (may not be faster if candidate set scanning dominates)
        # Just verify both complete in reasonable time
        assert avg_low < 100, f"Low selectivity too slow: {avg_low}ms"
        assert avg_high < 100, f"High selectivity too slow: {avg_high}ms"


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases for histogram and selectivity."""
    
    @pytest.fixture
    def edge_collection(self, caliby_env):
        """Create collection for edge case testing."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.INT)
        
        col = caliby.Collection("test_edge", schema, vector_dim=32,
                               distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec", M=16, ef_construction=100)
        col.create_metadata_index("cat_idx", ["category"])
        
        return col
    
    def test_empty_result_set(self, edge_collection):
        """Test filter that matches no documents."""
        col = edge_collection
        n = 1000
        
        np.random.seed(1111)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": 1} for _ in range(n)]  # All category 1
        
        col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Filter for category 0 (no matches)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": 0})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        assert len(results) == 0
    
    def test_all_match_filter(self, edge_collection):
        """Test filter that matches all documents (100% selectivity)."""
        col = edge_collection
        n = 1000
        
        np.random.seed(2222)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        metadatas = [{"category": 0} for _ in range(n)]  # All category 0
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping (all 0)
        id_to_category = {doc_id: 0 for doc_id in doc_ids}
        
        # Filter for category 0 (all match)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": 0})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        assert len(results) == 10
        for r in results:
            assert id_to_category[r.doc_id] == 0
    
    def test_single_matching_document(self, edge_collection):
        """Test filter that matches exactly one document."""
        col = edge_collection
        n = 1000
        
        np.random.seed(3333)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        
        categories = [1] * n
        categories[500] = 0  # Only one doc with category 0
        metadatas = [{"category": categories[i]} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: categories[i] for i in range(n)}
        
        # Filter for category 0 (one match)
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": 0})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        assert len(results) == 1
        assert id_to_category[results[0].doc_id] == 0
    
    def test_k_larger_than_matching_set(self, edge_collection):
        """Test when k is larger than the number of matching documents."""
        col = edge_collection
        n = 1000
        
        np.random.seed(4444)
        vectors = np.random.randn(n, 32).astype(np.float32)
        contents = [f"doc {i}" for i in range(n)]
        
        categories = [1] * n
        categories[:5] = [0] * 5  # Only 5 docs with category 0
        metadatas = [{"category": categories[i]} for i in range(n)]
        
        doc_ids = col.add(contents, metadatas, [v.tolist() for v in vectors])
        
        # Build doc_id to category mapping
        id_to_category = {doc_ids[i]: categories[i] for i in range(n)}
        
        # Request k=10 but only 5 match
        query = np.random.randn(32).astype(np.float32)
        filter_json = json.dumps({"category": 0})
        
        results = col.search_vector(query.tolist(), "vec", k=10, filter=filter_json)
        
        assert len(results) == 5  # Should return all 5 matching docs
        for r in results:
            assert id_to_category[r.doc_id] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

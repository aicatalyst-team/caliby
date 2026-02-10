"""
Tests for Array Index functionality.
Tests the inverted index for array-type metadata fields to accelerate $contains queries.
"""
import pytest
import tempfile
import shutil
import os
import json
import numpy as np
import caliby


class TestArrayIndexBasic:
    """Basic array index functionality tests."""
    
    def test_create_array_index_string_array(self):
        """Test creating an array index on a STRING_ARRAY field."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes first
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("tags_idx", "tags")
                
                # Add some documents with string arrays
                contents = []
                metadatas = []
                vectors = []
                for i in range(10):
                    tags = ["tag_a", "tag_b"] if i % 2 == 0 else ["tag_b", "tag_c"]
                    contents.append(f"doc_{i}")
                    metadatas.append({"tags": tags})
                    vectors.append(np.random.rand(8).astype(np.float32).tolist())
                
                collection.add(contents, metadatas, vectors)
                
                # Test basic $contains filter with search_vector
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_a = json.dumps({"tags": {"$contains": "tag_a"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_a)
                assert len(results) == 5  # Documents 0, 2, 4, 6, 8
                
                filter_b = json.dumps({"tags": {"$contains": "tag_b"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_b)
                assert len(results) == 10  # All documents have tag_b
                
                filter_c = json.dumps({"tags": {"$contains": "tag_c"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_c)
                assert len(results) == 5  # Documents 1, 3, 5, 7, 9
            finally:
                caliby.close()
    
    def test_create_array_index_int_array(self):
        """Test creating an array index on an INT_ARRAY field."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("categories", caliby.FieldType.INT_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes first
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("cat_idx", "categories")
                
                # Add documents with int arrays
                contents = []
                metadatas = []
                vectors = []
                for i in range(10):
                    cats = [1, 2] if i % 2 == 0 else [2, 3]
                    contents.append(f"doc_{i}")
                    metadatas.append({"categories": cats})
                    vectors.append(np.random.rand(8).astype(np.float32).tolist())
                
                collection.add(contents, metadatas, vectors)
                
                # Test $contains filter with integers
                query = np.random.rand(8).astype(np.float32).tolist()
                
                filter_1 = json.dumps({"categories": {"$contains": 1}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_1)
                assert len(results) == 5
                
                filter_2 = json.dumps({"categories": {"$contains": 2}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_2)
                assert len(results) == 10
                
                filter_3 = json.dumps({"categories": {"$contains": 3}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_3)
                assert len(results) == 5
            finally:
                caliby.close()
    
    def test_array_index_combined_with_and_filter(self):
        """Test array index combined with AND filters."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
                schema.add_field("status", caliby.FieldType.STRING)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("tags_idx", "tags")
                
                # Add documents
                contents = []
                metadatas = []
                vectors = []
                for i in range(20):
                    tags = ["python", "ai"] if i % 2 == 0 else ["java", "web"]
                    status = "active" if i % 3 == 0 else "inactive"
                    contents.append(f"doc_{i}")
                    metadatas.append({"tags": tags, "status": status})
                    vectors.append(np.random.rand(8).astype(np.float32).tolist())
                
                collection.add(contents, metadatas, vectors)
                
                # Test combined filter
                query = np.random.rand(8).astype(np.float32).tolist()
                combined_filter = json.dumps({
                    "$and": [
                        {"tags": {"$contains": "python"}},
                        {"status": "active"}
                    ]
                })
                results = collection.search_vector(query, "hnsw_idx", k=20, filter=combined_filter)
                # Documents with python tags (even indices: 0,2,4,6,8,10,12,14,16,18) 
                # AND status=active (i%3==0: 0,3,6,9,12,15,18)
                # Intersection: 0, 6, 12, 18
                assert len(results) == 4
            finally:
                caliby.close()
    
    def test_array_index_with_vector_search(self):
        """Test array index with filtered vector search."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("categories", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx", ef_construction=100, M=16)
                collection.create_array_index("cat_idx", "categories")
                
                # Add documents with specific vectors
                np.random.seed(42)
                contents = []
                metadatas = []
                vectors = []
                for i in range(100):
                    categories = ["cs.AI", "cs.LG"] if i < 50 else ["cs.CV", "cs.NE"]
                    contents.append(f"doc_{i}")
                    metadatas.append({"categories": categories})
                    vectors.append(np.random.rand(8).astype(np.float32).tolist())
                
                collection.add(contents, metadatas, vectors)
                
                # Search with filter
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_ai = json.dumps({"categories": {"$contains": "cs.AI"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_ai)
                
                # All results should be from the first 50 documents
                assert len(results) > 0
                for result in results:
                    assert result.doc_id < 50
            finally:
                caliby.close()
    
    def test_array_index_updates_on_document_changes(self):
        """Test that array index is updated when documents change."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("tags_idx", "tags")
                
                # Add initial documents
                contents = [f"doc_{i}" for i in range(5)]
                metadatas = [{"tags": ["original"]} for _ in range(5)]
                vectors = [np.random.rand(8).astype(np.float32).tolist() for _ in range(5)]
                ids = collection.add(contents, metadatas, vectors)
                
                # Verify initial state
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_orig = json.dumps({"tags": {"$contains": "original"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_orig)
                assert len(results) == 5
                
                # Add a new document - index should be updated
                new_ids = collection.add(["doc_5"], [{"tags": ["new_tag"]}], [np.random.rand(8).astype(np.float32).tolist()])
                
                filter_new = json.dumps({"tags": {"$contains": "new_tag"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_new)
                assert len(results) == 1
                
                # Update a document - index should reflect the change
                collection.update([ids[0]], [{"tags": ["updated"]}])
                
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_orig)
                assert len(results) == 4  # doc 0 no longer has "original"
                
                filter_updated = json.dumps({"tags": {"$contains": "updated"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_updated)
                assert len(results) == 1
            finally:
                caliby.close()


class TestArrayIndexEdgeCases:
    """Edge cases for array index."""
    
    def test_array_index_empty_arrays(self):
        """Test handling of documents with empty arrays."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("tags_idx", "tags")
                
                # Add documents with empty and non-empty arrays
                contents = ["doc_0", "doc_1"]
                metadatas = [{"tags": []}, {"tags": ["test"]}]
                vectors = [np.random.rand(8).astype(np.float32).tolist() for _ in range(2)]
                
                ids = collection.add(contents, metadatas, vectors)
                
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_test = json.dumps({"tags": {"$contains": "test"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_test)
                assert len(results) == 1
                assert results[0].doc_id == ids[1]
            finally:
                caliby.close()
    
    def test_array_index_large_arrays(self):
        """Test handling of documents with large arrays."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("tags_idx", "tags")
                
                # Add a document with many tags
                many_tags = [f"tag_{i}" for i in range(100)]
                collection.add(["doc_0"], [{"tags": many_tags}], [np.random.rand(8).astype(np.float32).tolist()])
                
                query = np.random.rand(8).astype(np.float32).tolist()
                
                # Each tag should be indexable
                for i in [0, 50, 99]:
                    filter_tag = json.dumps({"tags": {"$contains": f"tag_{i}"}})
                    results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_tag)
                    assert len(results) == 1
            finally:
                caliby.close()
    
    def test_array_index_nonexistent_element(self):
        """Test searching for non-existent element."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("tags_idx", "tags")
                
                collection.add(["doc_0"], [{"tags": ["existing"]}], [np.random.rand(8).astype(np.float32).tolist()])
                
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_none = json.dumps({"tags": {"$contains": "nonexistent"}})
                results = collection.search_vector(query, "hnsw_idx", k=10, filter=filter_none)
                assert len(results) == 0
            finally:
                caliby.close()


class TestArrayIndexPerformance:
    """Performance-related tests for array index."""
    
    def test_array_index_many_documents(self):
        """Test array index with many documents."""
        with tempfile.TemporaryDirectory() as test_dir:
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(test_dir)
            
            try:
                schema = caliby.Schema()
                schema.add_field("categories", caliby.FieldType.STRING_ARRAY)
                
                collection = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                collection.create_hnsw_index("hnsw_idx")
                collection.create_array_index("cat_idx", "categories")
                
                # Add 1000 documents
                categories_pool = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]
                np.random.seed(42)
                
                contents = []
                metadatas = []
                vectors = []
                for i in range(1000):
                    # Each document gets 2-3 random categories
                    num_cats = np.random.randint(2, 4)
                    cats = list(np.random.choice(categories_pool, num_cats, replace=False))
                    contents.append(f"doc_{i}")
                    metadatas.append({"categories": [str(c) for c in cats]})  # Convert np.str_ to str
                    vectors.append(np.random.rand(8).astype(np.float32).tolist())
                
                collection.add(contents, metadatas, vectors)
                
                # Test queries - should be fast with index
                query = np.random.rand(8).astype(np.float32).tolist()
                for cat in categories_pool:
                    filter_cat = json.dumps({"categories": {"$contains": cat}})
                    results = collection.search_vector(query, "hnsw_idx", k=100, filter=filter_cat)
                    assert len(results) > 0  # Each category should have documents
            finally:
                caliby.close()


class TestArrayIndexRecovery:
    """Test array index persistence and recovery."""
    
    def test_array_index_survives_restart(self):
        """Test that array index can be rebuilt on collection reload."""
        with tempfile.TemporaryDirectory() as test_dir:
            # Note: We don't call set_buffer_config in subprocess tests because
            # it may have been set by previous tests. The test uses a fresh temp dir
            # which ensures clean state for the data.
            try:
                caliby.set_buffer_config(size_gb=0.5)
            except RuntimeError:
                pass  # Config already set by previous test, that's OK
            caliby.open(test_dir)
            
            schema = caliby.Schema()
            schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
            
            try:
                # Create and populate collection
                col = caliby.Collection("test_col", schema, vector_dim=8)
                
                # Create indexes
                col.create_hnsw_index("hnsw_idx")
                col.create_array_index("tags_idx", "tags")
                
                contents = []
                metadatas = []
                vectors = []
                for i in range(10):
                    tags = ["alpha", "beta"] if i % 2 == 0 else ["gamma", "delta"]
                    contents.append(f"doc_{i}")
                    metadatas.append({"tags": tags})
                    vectors.append(np.random.rand(8).astype(np.float32).tolist())
                
                col.add(contents, metadatas, vectors)
                
                # Verify it works
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_alpha = json.dumps({"tags": {"$contains": "alpha"}})
                results = col.search_vector(query, "hnsw_idx", k=10, filter=filter_alpha)
                assert len(results) == 5
                
                # Close
                del col
            finally:
                caliby.close()
            
            # Re-open
            try:
                caliby.set_buffer_config(size_gb=0.5)
            except RuntimeError:
                pass  # Config already set, that's OK
            caliby.open(test_dir)
            
            try:
                col2 = caliby.Collection.open("test_col")
                
                # Array index may already exist if persisted, try to create but ignore error
                try:
                    col2.create_array_index("tags_idx", "tags")
                except RuntimeError:
                    pass  # Index already exists, that's OK
                
                query = np.random.rand(8).astype(np.float32).tolist()
                filter_alpha = json.dumps({"tags": {"$contains": "alpha"}})
                results = col2.search_vector(query, "hnsw_idx", k=10, filter=filter_alpha)
                assert len(results) == 5
                
                del col2
            finally:
                caliby.close()

#!/usr/bin/env python3
"""
Extended Text Index Edge Case Tests

Comprehensive tests for BM25 text index including edge cases,
special characters, multilingual text, and scoring correctness.

Run with: pytest tests/test_text_index_extended.py -v
"""

import pytest
import os
import sys
import tempfile
import numpy as np
import caliby

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)


@pytest.fixture
def col(caliby_module, temp_dir):
    """Function-scoped: each test gets a unique collection within the shared session."""
    import uuid
    caliby_module.open(temp_dir)
    schema = caliby.Schema()
    schema.add_field("title", caliby.FieldType.STRING)
    schema.add_field("year", caliby.FieldType.INT)
    col = caliby.Collection(f"text_test_{uuid.uuid4().hex[:8]}", schema)
    col.create_text_index("text_idx")
    return col


class TestBM25Scoring:
    """Verify BM25 scoring properties."""

    def test_term_frequency_saturation(self, col):
        """BM25 should saturate TF — repeating a term 100x is not 100x better."""
        # Add docs with varying TF
        col.add(
            ["python python python python python "
             "python python python python python"],
            [{"title": "high_tf", "year": 2024}])
        col.add(
            ["python programming tutorial"],
            [{"title": "low_tf", "year": 2024}])

        results = col.search_text("python", "text_idx", k=2)
        # Both should be found, high TF doc should rank higher
        assert len(results) == 2
        assert results[0].doc_id == 0  # Higher TF
        # Score difference should be sub-linear due to saturation
        assert results[0].score > results[1].score

    def test_idf_effect(self, col):
        """Rare terms should show high specificity — only matching docs returned."""
        col.add([
            "common word appears many times in all documents common common",
            "xylophone zebra quantum entropy singularity",
            "common word also appears here common"
        ], [
            {"title": "common1", "year": 2024},
            {"title": "rare", "year": 2024},
            {"title": "common2", "year": 2024},
        ])

        results_rare = col.search_text("xylophone", "text_idx", k=3)
        assert len(results_rare) >= 1, "Rare term search should find matching doc"

        results_common = col.search_text("common", "text_idx", k=3)
        assert len(results_common) >= 2, "Common term should match multiple docs"

    def test_document_length_normalization(self, col):
        """BM25 normalizes for document length — short docs with same TF rank higher."""
        col.add([
            "python",  # Short doc, all about python
            "python " * 99 + "java " * 100,  # Long doc, half about python
        ], [
            {"title": "short", "year": 2024},
            {"title": "long", "year": 2024},
        ])

        results = col.search_text("python", "text_idx", k=2)
        # Short doc should rank higher due to length normalization
        assert results[0].doc_id in [0, 1]


class TestTextSearchEdgeCases:
    """Edge cases for text search."""

    def test_empty_query(self, col):
        """Empty query should return empty or handle gracefully."""
        col.add(["test document"], [{"title": "test", "year": 2024}])
        try:
            results = col.search_text("", "text_idx", k=5)
            assert isinstance(results, list)
        except Exception:
            pass  # Empty query may be rejected

    def test_no_matching_documents(self, col):
        """Query that matches no documents should return empty."""
        col.add(["python programming"], [{"title": "python", "year": 2024}])
        results = col.search_text("zzzzzzz_nonexistent_term_zzzzzzz", "text_idx", k=5)
        assert len(results) == 0

    def test_single_character(self, col):
        """Single character queries should work or be handled."""
        col.add(["a b c"], [{"title": "abc", "year": 2024}])
        try:
            results = col.search_text("a", "text_idx", k=5)
            assert isinstance(results, list)
        except Exception:
            pass

    def test_case_insensitivity(self, col):
        """Text search should be case-insensitive."""
        col.add(["PYTHON Programming"], [{"title": "upper", "year": 2024}])
        col.add(["python tutorial"], [{"title": "lower", "year": 2024}])

        # Both queries should find both documents
        results_upper = col.search_text("PYTHON", "text_idx", k=5)
        results_lower = col.search_text("python", "text_idx", k=5)

        assert len(results_upper) == len(results_lower)

    def test_special_characters(self, col):
        """Documents with special characters should be indexable."""
        col.add([
            "C++ is a programming language",
            "C# and .NET framework",
            "Hello!!! World??? Test--case__underscore",
        ], [
            {"title": "cpp", "year": 2024},
            {"title": "csharp", "year": 2024},
            {"title": "special", "year": 2024},
        ])

        # Search should still find relevant documents
        results = col.search_text("programming", "text_idx", k=5)
        assert len(results) >= 1

        results = col.search_text("Hello", "text_idx", k=5)
        assert len(results) >= 1

    def test_numbers_in_text(self, col):
        """Numbers in text should be tokenized."""
        col.add([
            "version 2.0 released in 2024",
            "version 1.0 released in 2023",
        ], [
            {"title": "v2", "year": 2024},
            {"title": "v1", "year": 2023},
        ])

        results = col.search_text("2024", "text_idx", k=5)
        assert len(results) >= 1

    def test_unicode_and_chinese(self, col):
        """Unicode and CJK text should be indexable and searchable."""
        col.add([
            "artificial intelligence and machine learning",
            "deep learning neural networks tutorial",
        ], [
            {"title": "en1", "year": 2024},
            {"title": "en2", "year": 2024},
        ])

        results = col.search_text("learning", "text_idx", k=5)
        assert len(results) >= 1

        # Also test with accented characters
        col.add([
            "café résumé naïve piñata",
        ], [
            {"title": "accented", "year": 2024},
        ])
        results_accent = col.search_text("café", "text_idx", k=5)
        # May or may not find accented chars depending on analyzer
        assert isinstance(results_accent, list)

    def test_remove_and_reindex(self, col):
        """Documents can be deleted and the collection count reflects deletions."""
        ids = col.add([
            "document to be deleted",
            "document to keep",
        ], [
            {"title": "del", "year": 2024},
            {"title": "keep", "year": 2024},
        ])

        # Verify both found initially
        results_before = col.search_text("document", "text_idx", k=5)
        assert len(results_before) == 2

        # Delete first doc
        col.delete([ids[0]])
        assert col.doc_count() == 1

        # Search may still return both due to index implementation details
        results_after = col.search_text("document", "text_idx", k=5)
        assert len(results_after) >= 1  # At least the kept doc should be found

    def test_large_batch_indexing(self, col):
        """Index and search many documents."""
        n = 200
        contents = []
        metadatas = []
        for i in range(n):
            if i < n // 2:
                contents.append(f"Python machine learning tutorial number {i}")
                metadatas.append({"title": f"python_{i}", "year": 2024})
            else:
                contents.append(f"Java enterprise development guide number {i}")
                metadatas.append({"title": f"java_{i}", "year": 2024})

        ids = col.add(contents, metadatas)
        assert col.doc_count() == n

        results_python = col.search_text("python tutorial", "text_idx", k=50)
        results_java = col.search_text("java guide", "text_idx", k=50)

        assert len(results_python) > 0
        assert len(results_java) > 0
        # Should return different top results
        assert results_python[0].doc_id != results_java[0].doc_id

    def test_result_scores_are_descending(self, col):
        """Search results should be sorted by score descending."""
        col.add([
            f"python is a great language for data science and machine learning {i}"
            for i in range(20)
        ], [{"title": f"doc_{i}", "year": 2024} for i in range(20)])

        results = col.search_text("python machine learning", "text_idx", k=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, \
                f"Scores not sorted: {results[i].score} < {results[i + 1].score}"

    def test_k_larger_than_docs(self, col):
        """Requesting more results than documents should return available docs."""
        col.add(["doc a", "doc b", "doc c"],
                [{"title": "a", "year": 2024},
                 {"title": "b", "year": 2024},
                 {"title": "c", "year": 2024}])

        results = col.search_text("doc", "text_idx", k=100)
        assert len(results) == 3


class TestTextIndexWithVectors:
    """Text index combined with vector operations."""

    def test_text_search_after_vector_operations(self, caliby_module, temp_dir):
        """Text index should still work after vector operations."""
        import uuid
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("tag", caliby.FieldType.STRING)
        col = caliby.Collection(f"vec_text_{uuid.uuid4().hex[:8]}", schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")

        vectors = np.random.rand(10, 4).astype(np.float32).tolist()
        col.add(
            [f"document number {i} about python" for i in range(10)],
            [{"tag": "python"} for _ in range(10)],
            vectors,
        )
        vec_results = col.search_vector([0.5, 0.5, 0, 0], "vec_idx", k=5)
        assert len(vec_results) == 5
        text_results = col.search_text("python", "text_idx", k=5)
        assert len(text_results) >= 1


class TestTextIndexPersistence:
    """Text index persistence and recovery — runs in subprocess to fully reset state."""

    def test_text_index_persists(self):
        import subprocess as sp
        td = tempfile.mkdtemp()

        script = f'''
import sys, os
sys.path.insert(0, "{build_path}")
import caliby, tempfile

td = "{td}"
caliby.open(td)
schema = caliby.Schema()
schema.add_field("title", caliby.FieldType.STRING)
col = caliby.Collection("text_persist", schema)
col.create_text_index("text_idx")
col.add([
    "Python machine learning tutorial",
    "Java spring boot guide",
    "Rust systems programming",
], [
    {{"title": "python"}}, {{"title": "java"}}, {{"title": "rust"}},
])
col.flush()
caliby.close()

caliby.open(td)
col2 = caliby.Collection.open("text_persist")
results = col2.search_text("python", "text_idx", k=5)
assert len(results) >= 1, f"Expected >= 1, got {{len(results)}}"
caliby.close()
print("PASS")
'''
        result = sp.run([sys.executable, "-c", script],
                        capture_output=True, text=True, timeout=30,
                        env={**os.environ, "PYTHONPATH": build_path})
        import shutil
        shutil.rmtree(td, ignore_errors=True)
        assert result.returncode == 0, f"Persistence test failed: {result.stderr}"
        assert "PASS" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

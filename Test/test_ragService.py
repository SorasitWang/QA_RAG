import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import MagicMock

from DatabaseService.RagService import RagService, _mmrSelect
from Dto.RetrievalResult import RetrievalResult
from Dto.Source import Source


def makeRaw(distances: list[float], embeddings: list[list[float]], docs=None):
    """Build a raw ChromaDB-style response dict for use in mock return values."""
    if docs is None:
        docs = [f"doc{i}" for i in range(len(distances))]
    return RetrievalResult(
        ids=        [f"id{i}" for i in range(len(distances))],
        documents=  docs,
        distances=  distances,
        metadatas=  [{} for _ in distances],
        embeddings= embeddings,
    )


class TestMmrSelect(unittest.TestCase):

    def test_returns_exactly_k_indices(self):
        distances  = [0.1, 0.2, 0.3, 0.4, 0.5]
        embeddings = [[1,0],[0,1],[1,1],[0.5,0.5],[-1,0]]
        result = _mmrSelect(distances, embeddings, k=3, lambda_=0.7)
        self.assertEqual(len(result), 3)

    def test_pure_relevance_first_is_most_similar(self):
        distances  = [0.1, 0.5, 0.9]
        embeddings = [[1,0],[0,1],[-1,0]]
        result = _mmrSelect(distances, embeddings, k=1, lambda_=1.0)
        self.assertEqual(result[0], 0)

    def test_diversity_prefers_diverse_over_redundant(self):
        # idx 0 and 1 are nearly identical; idx 2 is orthogonal
        distances  = [0.1, 0.15, 0.5]
        embeddings = [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]]
        result = _mmrSelect(distances, embeddings, k=2, lambda_=0.5)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 2)

    def test_fewer_candidates_than_k_returns_all(self):
        distances  = [0.1, 0.2]
        embeddings = [[1,0],[0,1]]
        result = _mmrSelect(distances, embeddings, k=5, lambda_=0.7)
        self.assertEqual(len(result), 2)

    def test_all_indices_are_unique(self):
        distances  = [0.1, 0.2, 0.3, 0.4]
        embeddings = [[1,0],[0,1],[-1,0],[0,-1]]
        result = _mmrSelect(distances, embeddings, k=4, lambda_=0.5)
        self.assertEqual(len(result), len(set(result)))


class TestAddSources(unittest.TestCase):

    def setUp(self):
        self.mock_repo = MagicMock()
        self.service   = RagService(self.mock_repo)

    def test_calls_repository_insert(self):
        source = Source("title", "text", "content", [])
        self.service.addSources([source], [0.1, 0.2])
        self.mock_repo.insert.assert_called_once()

    def test_document_metadata_contains_title_and_source_type(self):
        source = Source("myTitle", "news", "content", [])
        self.service.addSources([source], [0.1])
        inserted_doc = self.mock_repo.insert.call_args[0][0][0]
        self.assertEqual(inserted_doc.metadata['title'], "myTitle")
        self.assertEqual(inserted_doc.metadata['sourceType'], "news")

    def test_legacy_text_type_is_stored_as_file(self):
        self.service.addSources([Source("t", "text", "content", [])], [0.1])
        inserted_doc = self.mock_repo.insert.call_args[0][0][0]
        self.assertEqual(inserted_doc.metadata['sourceType'], "file")

    def test_optional_metadata_drops_none_values(self):
        source = Source("t", "news", "c", [], {"url": "https://x", "author": None, "publishedAt": 1700000000})
        self.service.addSources([source], [0.1])
        meta = self.mock_repo.insert.call_args[0][0][0].metadata
        self.assertEqual(meta["url"], "https://x")
        self.assertEqual(meta["publishedAt"], 1700000000)
        self.assertNotIn("author", meta)

    def test_multiple_sources_all_inserted(self):
        sources = [Source(f"t{i}", "text", "c", []) for i in range(3)]
        self.service.addSources(sources, [0.1])
        inserted_docs = self.mock_repo.insert.call_args[0][0]
        self.assertEqual(len(inserted_docs), 3)


class TestSourceExists(unittest.TestCase):

    def setUp(self):
        self.mock_repo = MagicMock()
        self.service   = RagService(self.mock_repo)

    def test_delegates_and_returns_true_when_found(self):
        self.mock_repo.existsByTitle.return_value = True
        self.assertTrue(self.service.sourceExists("title"))
        self.mock_repo.existsByTitle.assert_called_once_with("title")

    def test_returns_false_when_not_found(self):
        self.mock_repo.existsByTitle.return_value = False
        self.assertFalse(self.service.sourceExists("missing"))


class TestQuery(unittest.TestCase):

    def setUp(self):
        self.mock_repo = MagicMock()
        self.service   = RagService(self.mock_repo)

    def test_empty_distances_returns_empty(self):
        raw = makeRaw([], [])
        self.mock_repo.search.return_value = raw
        result = self.service.query("q")
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.documents, [])

    def test_threshold_filters_high_distance_chunks(self):
        # distances 0.5 and 0.8 pass (≤1.0), 1.5 does not
        raw = makeRaw([0.5, 0.8, 1.5], [[1,0],[0,1],[-1,0]])
        self.mock_repo.search.return_value = raw
        result = self.service.query("q", topK=3)
        self.assertNotIn("doc2", result.documents)

    def test_all_beyond_threshold_returns_empty(self):
        raw = makeRaw([1.5, 2.0], [[1,0],[0,1]])
        self.mock_repo.search.return_value = raw
        result = self.service.query("q")
        self.assertEqual(result.documents, [])

    def test_mmr_reduces_results_to_topK(self):
        embeddings = [[1,0],[0,1],[0.5,0.5],[0.3,0.7],[0.7,0.3]]
        raw = makeRaw([0.1,0.2,0.3,0.4,0.5], embeddings)
        self.mock_repo.search.return_value = raw
        result = self.service.query("q", topK=2)
        self.assertEqual(len(result.documents), 2)

    def test_returns_retrieval_result_with_all_fields(self):
        raw = makeRaw([0.3, 0.5], [[1,0],[0,1]])
        self.mock_repo.search.return_value = raw
        result = self.service.query("q", topK=2)
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(len(result.ids), len(result.documents))
        self.assertEqual(len(result.ids), len(result.distances))
        self.assertEqual(len(result.ids), len(result.metadatas))
        self.assertEqual(len(result.ids), len(result.embeddings))


if __name__ == '__main__':
    unittest.main()

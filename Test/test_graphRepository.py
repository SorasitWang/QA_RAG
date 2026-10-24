import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DatabaseRepository.GraphRepository import GraphRepository
from Dto.VectorDocument import VectorDocument


def _repo(tmpDir: str) -> GraphRepository:
    return GraphRepository(dbPath=os.path.join(tmpDir, "test_graph.sqlite3"))


def _doc(chunkId: str, text: str, title: str = "doc") -> VectorDocument:
    return VectorDocument(chunkId, text, [], {"title": title})


class TestGraphRepositorySchema(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.repo = _repo(self.tmpDir)

    def test_existsByTitle_returns_false_when_empty(self):
        self.assertFalse(self.repo.existsByTitle("anything"))

    def test_existsByTitle_consistent_with_entitychunks_rows(self):
        self.repo.insert([_doc("chunk_001", "Apple is a technology company.", "tech")])
        result = self.repo.existsByTitle("tech")
        with self.repo._connect() as conn:
            hasRows = conn.execute(
                "SELECT 1 FROM entityChunks WHERE title = 'tech' LIMIT 1"
            ).fetchone() is not None
        self.assertEqual(result, hasRows)

    def test_insert_is_idempotent(self):
        doc = _doc("chunk_dup", "Google is based in the United States.", "dup")
        self.repo.insert([doc])
        self.repo.insert([doc])  # second call must not raise

    def test_insert_skips_chunk_with_no_entities(self):
        # A chunk with no named entities produces no rows — existsByTitle stays False.
        self.repo.insert([_doc("chunk_noent", "the quick brown fox jumps", "noent")])
        self.assertFalse(self.repo.existsByTitle("noent"))


class TestGraphRepositorySearch(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.repo = _repo(self.tmpDir)
        docs = [
            _doc("chunk_001", "Google and Microsoft are technology companies.", "tech"),
            _doc("chunk_002", "Google acquired YouTube in 2006.", "tech"),
            _doc("chunk_003", "Microsoft released Windows 11.", "tech"),
            _doc("chunk_004", "Apple and Google compete in the smartphone market.", "tech"),
        ]
        self.repo.insert(docs)

    def test_search_returns_list(self):
        result = self.repo.search("Google technology", topK=5)
        self.assertIsInstance(result, list)

    def test_search_respects_topK(self):
        result = self.repo.search("Google technology", topK=1)
        self.assertLessEqual(len(result), 1)

    def test_search_empty_string_returns_empty(self):
        result = self.repo.search("", topK=5)
        self.assertEqual(result, [])

    def test_search_unknown_entity_returns_empty(self):
        result = self.repo.search("Zyxwvutsrqponm unknownxyz", topK=5)
        self.assertEqual(result, [])

    def test_search_returns_chunk_ids_as_strings(self):
        result = self.repo.search("Google", topK=5)
        for item in result:
            self.assertIsInstance(item, str)

    def test_hop_depth_zero_returns_only_seed_chunks(self):
        result = self.repo.search("Google", topK=5, hopDepth=0)
        self.assertIsInstance(result, list)
        # With hop depth 0, only chunks directly linked to Google seed entities are returned.
        # "chunk_003" (Microsoft/Windows only, no Google) must not appear.
        self.assertNotIn("chunk_003", result)


if __name__ == "__main__":
    unittest.main()

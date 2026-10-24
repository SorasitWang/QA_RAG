import os
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DatabaseRepository.GraphRepository import GraphRepository
from DatabaseRepository.TfidfRepository import TfidfRepository
from Dto.VectorDocument import VectorDocument


_KEEP = [
    VectorDocument("keep_1", "Google and Microsoft compete in cloud computing.", [0.1, 0.2], {"title": "keep"}),
    VectorDocument("keep_2", "Microsoft acquired GitHub to grow cloud developer tools.", [0.2, 0.1], {"title": "keep"}),
]
_DROP = [
    VectorDocument("drop_1", "Google and Apple build smartphones in the United States.", [0.3, 0.3], {"title": "drop"}),
    VectorDocument("drop_2", "Apple and Microsoft sell laptops and cloud storage.", [0.4, 0.1], {"title": "drop"}),
]


class _TempDirCase(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpDir, ignore_errors=True)


class TestTfidfDeleteByTitle(_TempDirCase):

    def _repo(self, name: str) -> TfidfRepository:
        return TfidfRepository(os.path.join(self.tmpDir, name, "tfidf.sqlite3"))

    def _stats(self, repo: TfidfRepository) -> tuple[dict[str, int], str, int]:
        with repo._connect() as conn:
            df = {r["term"]: r["df"] for r in conn.execute("SELECT term, df FROM documentFrequencies")}
            total = conn.execute("SELECT value FROM corpusStats WHERE key = 'totalDocs'").fetchone()["value"]
            tfRows = conn.execute("SELECT COUNT(*) FROM termFrequencies").fetchone()[0]
        return df, total, tfRows

    def test_stats_match_fresh_index_of_remaining_title(self):
        repo = self._repo("mixed")
        repo.insert(_KEEP + _DROP)
        removed = repo.deleteByTitle("drop")

        fresh = self._repo("fresh")
        fresh.insert(_KEEP)

        self.assertEqual(removed, 2)
        self.assertEqual(self._stats(repo), self._stats(fresh))
        self.assertFalse(repo.existsByTitle("drop"))
        self.assertTrue(repo.existsByTitle("keep"))

    def test_search_no_longer_returns_deleted_chunks(self):
        repo = self._repo("search")
        repo.insert(_KEEP + _DROP)
        repo.deleteByTitle("drop")
        self.assertNotIn("drop_1", repo.search("smartphones Apple", topK=10).ids)

    def test_unknown_title_is_noop(self):
        repo = self._repo("noop")
        repo.insert(_KEEP)
        self.assertEqual(repo.deleteByTitle("missing"), 0)
        self.assertEqual(self._stats(repo)[1], "2")


class TestGraphDeleteByTitle(_TempDirCase):

    def _repo(self, name: str) -> GraphRepository:
        return GraphRepository(os.path.join(self.tmpDir, name, "graph.sqlite3"))

    def _snapshot(self, repo: GraphRepository) -> tuple[set, set]:
        # Compare by entity name because autoincrement IDs differ between databases.
        with repo._connect() as conn:
            entities = {
                (r["name"], r["entityType"], r["chunkCount"])
                for r in conn.execute("SELECT name, entityType, chunkCount FROM entities")
            }
            edges = {
                (r["a"], r["b"], r["weight"])
                for r in conn.execute(
                    """SELECT ea.name AS a, eb.name AS b, c.weight FROM coOccurrences c
                       JOIN entities ea ON ea.id = c.entityIdA
                       JOIN entities eb ON eb.id = c.entityIdB"""
                )
            }
        return entities, {tuple(sorted(e[:2])) + (e[2],) for e in edges}

    def test_graph_matches_fresh_index_of_remaining_title(self):
        repo = self._repo("mixed")
        repo.insert(_KEEP + _DROP)
        removed = repo.deleteByTitle("drop")

        fresh = self._repo("fresh")
        fresh.insert(_KEEP)

        self.assertEqual(removed, 2)
        self.assertEqual(self._snapshot(repo), self._snapshot(fresh))
        self.assertFalse(repo.existsByTitle("drop"))

    def test_unknown_title_is_noop(self):
        repo = self._repo("noop")
        repo.insert(_KEEP)
        before = self._snapshot(repo)
        self.assertEqual(repo.deleteByTitle("missing"), 0)
        self.assertEqual(self._snapshot(repo), before)


class TestVectorDeleteByTitle(unittest.TestCase):

    def test_removes_only_matching_title(self):
        import chromadb
        from DatabaseRepository.VectorRepository import VectorRepository

        collection = chromadb.EphemeralClient().create_collection(f"test_{uuid.uuid4().hex}")
        repo = VectorRepository(collection)
        repo.insert([VectorDocument("keep", d.document, d.embeddings, d.metadata) for d in _KEEP])
        repo.insert([VectorDocument("drop", d.document, d.embeddings, d.metadata) for d in _DROP])

        self.assertEqual(repo.deleteByTitle("drop"), 2)
        self.assertFalse(repo.existsByTitle("drop"))
        self.assertEqual(collection.count(), 2)
        self.assertEqual(repo.deleteByTitle("drop"), 0)


if __name__ == '__main__':
    unittest.main()

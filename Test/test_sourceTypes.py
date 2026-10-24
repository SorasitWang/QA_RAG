import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Controller.SearchData import cleanFeedText
from DatabaseRepository.TfidfRepository import TfidfRepository
from DatabaseService.RagService import RagService
from Dto.RetrievalResult import RetrievalResult
from Dto.VectorDocument import VectorDocument


def _doc(chunkId: str, text: str, sourceType: str, **extra) -> VectorDocument:
    return VectorDocument(chunkId, text, [0.1, 0.2, 0.3], {"title": chunkId, "sourceType": sourceType, **extra})


class _TempDirCase(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.dbPath = os.path.join(self.tmpDir, "tfidf.sqlite3")

    def tearDown(self):
        shutil.rmtree(self.tmpDir, ignore_errors=True)


class TestTfidfSourceTypes(_TempDirCase):

    def setUp(self):
        super().setUp()
        self.repo = TfidfRepository(self.dbPath)
        self.repo.insert([
            _doc("file_1", "central banks raise interest rates to fight inflation", "file"),
            _doc("news_1", "breaking news central banks raise interest rates again", "news",
                 url="https://news.example/rates", publishedAt=1700000000),
            _doc("sns_1", "my take on interest rates this week", "sns"),
        ])

    def test_search_returns_stored_metadata(self):
        result = self.repo.search("interest rates", topK=10)
        meta = result.metadatas[result.ids.index("news_1")]
        self.assertEqual(meta["sourceType"], "news")
        self.assertEqual(meta["url"], "https://news.example/rates")
        self.assertEqual(meta["publishedAt"], 1700000000)

    def test_filter_restricts_to_requested_types(self):
        result = self.repo.search("interest rates", topK=10, sourceTypes=["news", "sns"])
        self.assertEqual(sorted(result.ids), ["news_1", "sns_1"])

    def test_filter_applies_before_topk_cut(self):
        # The file chunk scores highest, but with topK=1 a news filter must still return news.
        result = self.repo.search("central banks inflation", topK=1, sourceTypes=["news"])
        self.assertEqual(result.ids, ["news_1"])


class TestTfidfMigration(_TempDirCase):

    def test_old_schema_rows_become_file(self):
        # Create a database with the pre-sourceType schema and one row.
        with sqlite3.connect(self.dbPath) as conn:
            conn.executescript("""
                CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                                        docLength INTEGER NOT NULL, embedding BLOB NOT NULL);
                CREATE TABLE termFrequencies (docId TEXT NOT NULL, term TEXT NOT NULL, tf REAL NOT NULL,
                                              PRIMARY KEY (docId, term));
                CREATE TABLE documentFrequencies (term TEXT PRIMARY KEY, df INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE corpusStats (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO corpusStats VALUES ('totalDocs', '1');
                INSERT INTO documents VALUES ('old_1', 'old', 'inflation report', 2, X'00000000');
                INSERT INTO termFrequencies VALUES ('old_1', 'inflation', 0.5), ('old_1', 'report', 0.5);
                INSERT INTO documentFrequencies VALUES ('inflation', 1), ('report', 1);
            """)

        repo = TfidfRepository(self.dbPath)
        result = repo.search("inflation", topK=5, sourceTypes=["file"])
        self.assertEqual(result.ids, ["old_1"])
        self.assertEqual(result.metadatas[0], {"title": "old", "sourceType": "file"})


def _raw(ids: list[str], types: list[str], distances: list[float] | None = None) -> RetrievalResult:
    return RetrievalResult(
        ids=ids,
        documents=[f"doc {i}" for i in ids],
        distances=distances or [0.1 * (n + 1) for n in range(len(ids))],
        metadatas=[{"title": i, "sourceType": t} for i, t in zip(ids, types)],
        embeddings=[[1.0, float(n), 0.5] for n in range(len(ids))],
    )


class TestRagServiceSourceTypeFilter(unittest.TestCase):

    def setUp(self):
        self.vector = MagicMock()
        self.tfidf = MagicMock()
        self.graph = MagicMock()
        self.service = RagService(self.vector, self.tfidf, self.graph)

    def test_vector_mode_sends_chroma_where_clause(self):
        self.vector.search.return_value = _raw([], [])
        self.service.query("q", searchMode="vector", sourceTypes=["news", "sns"])
        where = self.vector.search.call_args.args[1]
        self.assertEqual(where, {"sourceType": {"$in": ["news", "sns"]}})

    def test_vector_mode_combines_tags_and_source_types_with_and(self):
        self.vector.search.return_value = _raw([], [])
        self.service.query("q", tags=["economics"], searchMode="vector", sourceTypes=["news"])
        where = self.vector.search.call_args.args[1]
        self.assertIn("$and", where)

    def test_no_filter_sends_empty_where(self):
        self.vector.search.return_value = _raw([], [])
        self.service.query("q", searchMode="vector")
        self.assertEqual(self.vector.search.call_args.args[1], {})

    def test_sparse_mode_passes_filter_to_tfidf(self):
        self.tfidf.search.return_value = _raw([], [])
        self.service.query("q", searchMode="sparse", sourceTypes=["blog"])
        self.assertEqual(self.tfidf.search.call_args.args[2], ["blog"])

    def test_hybrid_mode_filters_both_retrievers(self):
        self.vector.search.return_value = _raw([], [])
        self.tfidf.search.return_value = _raw([], [])
        self.service.query("q", searchMode="hybrid", sourceTypes=["news"])
        self.assertEqual(self.vector.search.call_args.args[1], {"sourceType": {"$in": ["news"]}})
        self.assertEqual(self.tfidf.search.call_args.args[2], ["news"])

    def test_graph_mode_filters_hydrated_chunks(self):
        self.graph.search.return_value = ["a", "b", "c"]
        self.vector.getByIds.return_value = _raw(["a", "b", "c"], ["file", "news", "sns"], [0.0, 0.0, 0.0])
        result = self.service.query("q", searchMode="graph", sourceTypes=["news"], topK=5)
        self.assertEqual(result.ids, ["b"])

    def test_graph_mode_widens_pool_when_filtering(self):
        self.graph.search.return_value = []
        self.service.query("q", searchMode="graph")
        unfiltered = self.graph.search.call_args.kwargs["topK"]
        self.service.query("q", searchMode="graph", sourceTypes=["news"])
        self.assertGreater(self.graph.search.call_args.kwargs["topK"], unfiltered)

    def test_graph_mode_treats_missing_source_type_as_file(self):
        self.graph.search.return_value = ["a"]
        raw = _raw(["a"], ["file"], [0.0])
        raw.metadatas = [{"title": "a"}]
        self.vector.getByIds.return_value = raw
        self.assertEqual(self.service.query("q", searchMode="graph", sourceTypes=["file"]).ids, ["a"])

    def test_unknown_source_type_raises(self):
        with self.assertRaises(ValueError):
            self.service.query("q", sourceTypes=["podcast"])


class TestVectorSourceTypes(unittest.TestCase):

    def setUp(self):
        import uuid
        import chromadb
        from DatabaseRepository.VectorRepository import VectorRepository
        self.collection = chromadb.EphemeralClient().create_collection(f"test_{uuid.uuid4().hex}")
        self.repo = VectorRepository(self.collection)

    def _metas(self) -> dict:
        page = self.collection.get(include=["metadatas"])
        return dict(zip(page["ids"], page["metadatas"]))

    def test_backfill_sets_file_only_where_missing(self):
        self.collection.add(ids=["old"], embeddings=[[0.1, 0.2]], documents=["d"], metadatas=[{"title": "old", "type": "text"}])
        self.collection.add(ids=["new"], embeddings=[[0.2, 0.1]], documents=["d"], metadatas=[{"title": "new", "sourceType": "news"}])

        self.assertEqual(self.repo.backfillLegacyMetadata(), 1)
        metas = self._metas()
        self.assertEqual(metas["old"]["sourceType"], "file")
        self.assertEqual(metas["new"]["sourceType"], "news")
        self.assertEqual(self.repo.backfillLegacyMetadata(), 0)

    def test_backfill_recovers_missing_title_so_reindex_replaces_old_chunks(self):
        # Very old indexes stored only {'type': 'text'}; the title survives in the chunk ID.
        legacyId = "./Test/RawTextFiles\\macro_economics_ec228267"
        self.collection.add(ids=[legacyId], embeddings=[[0.1, 0.2]], documents=["d"], metadatas=[{"type": "text"}])

        self.repo.backfillLegacyMetadata()

        self.assertEqual(self._metas()[legacyId]["title"], "./Test/RawTextFiles\\macro_economics")
        self.assertEqual(self.repo.deleteByTitle("./Test/RawTextFiles\\macro_economics"), 1)

    def test_where_clause_from_rag_service_is_valid_chroma_syntax(self):
        self.collection.add(
            ids=["a", "b"], embeddings=[[0.1, 0.2], [0.2, 0.1]], documents=["x", "y"],
            metadatas=[{"title": "a", "sourceType": "file"}, {"title": "b", "sourceType": "news"}],
        )
        where = RagService(self.repo)._buildFilters(None, ["news"])
        self.assertEqual(self.collection.get(where=where)["ids"], ["b"])


class TestCleanFeedText(unittest.TestCase):

    def test_keeps_short_post_and_strips_url(self):
        self.assertEqual(cleanFeedText("Rates up again! https://t.co/abc"), "Rates up again!")

    def test_strips_html_and_decodes_entities(self):
        text = "<p>Inflation &amp; growth</p><p>Second <b>para</b></p>"
        self.assertEqual(cleanFeedText(text), "Inflation & growth\nSecond para")

    def test_drops_lines_that_are_only_links(self):
        self.assertEqual(cleanFeedText("Headline\nhttps://example.com/x\n"), "Headline")


if __name__ == '__main__':
    unittest.main()

import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DatabaseRepository.IngestManifestRepository import IngestManifestRepository
from DatabaseService.IngestionService import IngestionService, IngestionWorker, isIgnoredFile
from Dto.Source import Source

_Labels = ['economics', 'science', 'sports', 'politics']
# Lines must exceed Config.Model.MinLineLength (20 chars) to survive cleanText.
_TextV1 = "Inflation rose sharply across the economy this year.\nCentral banks responded by raising interest rates."
_TextV2 = "The championship final drew a record crowd to the stadium.\nThe home team won after extra time."


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.encode."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[len(t) % 7 + 1.0, len(t) % 3 + 1.0, 1.0] for t in texts])


class FakeRagService:
    """Tracks chunks per title the way the real stores would."""

    def __init__(self):
        self.chunks: dict[str, list[str]] = {}
        self.tags: dict[str, list[str]] = {}
        self.deleted: list[str] = []

    def sourceExists(self, title: str) -> bool:
        return bool(self.chunks.get(title))

    def deleteSource(self, title: str) -> int:
        self.deleted.append(title)
        return len(self.chunks.pop(title, []))

    def addSources(self, sources: list[Source], embeddings: list[float]):
        for source in sources:
            self.chunks.setdefault(source.title, []).append(source.content)
            self.tags[source.title] = source.tags


class _IngestionCase(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.folder = os.path.join(self.tmpDir, "sources")
        os.makedirs(self.folder)
        self.rag = FakeRagService()
        self.manifest = IngestManifestRepository(os.path.join(self.tmpDir, "manifest.sqlite3"))
        self.service = IngestionService(self.rag, self.manifest, FakeModel(), self.folder, _Labels)

    def tearDown(self):
        shutil.rmtree(self.tmpDir, ignore_errors=True)

    def _write(self, name: str, text: str) -> str:
        path = os.path.join(self.folder, name)
        with open(path, 'w') as f:
            f.write(text)
        return path

    def _title(self, name: str) -> str:
        return os.path.splitext(os.path.join(self.folder, name))[0]


class TestIngestFile(_IngestionCase):

    def test_new_file_is_indexed(self):
        path = self._write("a.txt", _TextV1)
        self.assertEqual(self.service.ingestFile(path), 'indexed')
        self.assertTrue(self.rag.sourceExists(self._title("a.txt")))
        self.assertIsNotNone(self.manifest.getHash(self._title("a.txt")))

    def test_unchanged_file_is_skipped(self):
        path = self._write("a.txt", _TextV1)
        self.service.ingestFile(path)
        self.assertEqual(self.service.ingestFile(path), 'skipped')

    def test_edited_file_replaces_old_chunks(self):
        path = self._write("a.txt", _TextV1)
        self.service.ingestFile(path)
        self._write("a.txt", _TextV2)
        self.assertEqual(self.service.ingestFile(path), 'indexed')
        chunks = " ".join(self.rag.chunks[self._title("a.txt")])
        self.assertIn("championship", chunks)
        self.assertNotIn("Inflation", chunks)

    def test_deleted_file_is_removed(self):
        path = self._write("a.txt", _TextV1)
        self.service.ingestFile(path)
        os.remove(path)
        self.assertEqual(self.service.ingestFile(path), 'removed')
        self.assertFalse(self.rag.sourceExists(self._title("a.txt")))
        self.assertIsNone(self.manifest.getHash(self._title("a.txt")))

    def test_previously_indexed_without_manifest_is_reindexed_not_duplicated(self):
        # Indexes built before the manifest existed: chunks present, no hash recorded.
        path = self._write("a.txt", _TextV1)
        self.rag.chunks[self._title("a.txt")] = ["stale chunk"]
        self.service.ingestFile(path)
        self.assertNotIn("stale chunk", self.rag.chunks[self._title("a.txt")])

    def test_title_is_anchored_on_source_folder(self):
        # Watchers may report absolute paths; titles must still match fetchDataFromFolder.
        path = self._write("a.txt", _TextV1)
        self.service.ingestFile(os.path.abspath(path))
        self.assertIn(self._title("a.txt"), self.rag.chunks)

    def test_tags_are_assigned(self):
        path = self._write("a.txt", _TextV1)
        self.service.ingestFile(path)
        tags = self.rag.tags[self._title("a.txt")]
        self.assertEqual(len(tags), min(len(_Labels), 4))
        self.assertTrue(set(tags) <= set(_Labels))

    def test_file_with_no_usable_text_records_zero_chunks(self):
        path = self._write("short.txt", "too short\nalso short")
        self.assertEqual(self.service.ingestFile(path), 'indexed')
        self.assertFalse(self.rag.sourceExists(self._title("short.txt")))


class TestReconcile(_IngestionCase):

    def test_catches_changes_made_while_stopped(self):
        self._write("keep.txt", _TextV1)
        self._write("edit.txt", _TextV1)
        self._write("gone.txt", _TextV1)
        self.service.reconcile()

        # Simulate edits while the service was down.
        self._write("edit.txt", _TextV2)
        os.remove(os.path.join(self.folder, "gone.txt"))
        self._write("new.txt", _TextV2)

        counts = self.service.reconcile()
        self.assertEqual(counts, {'indexed': 2, 'skipped': 1, 'removed': 1})
        self.assertNotIn(self._title("gone.txt"), self.rag.chunks)
        self.assertIn(self._title("new.txt"), self.rag.chunks)
        self.assertEqual(sorted(self.manifest.listTitles()),
                         sorted(self._title(n) for n in ("keep.txt", "edit.txt", "new.txt")))

    def test_ignores_temp_files(self):
        self._write(".hidden.txt", _TextV1)
        self._write("draft.txt.swp", _TextV1)
        self.assertEqual(self.service.reconcile(), {'indexed': 0, 'skipped': 0, 'removed': 0})

    def test_short_sns_post_with_url_is_kept(self):
        # cleanText would drop this whole line (short + URL); feed items use cleanFeedText.
        self.service.ingestSource(Source("https://m.social/@a/1", "sns", "CPI up 3% https://t.co/x", []))
        self.assertEqual(self.rag.chunks["https://m.social/@a/1"], ["CPI up 3%"])

    def test_same_short_text_as_file_is_dropped(self):
        self.service.ingestSource(Source("short-file", "file", "CPI up 3% https://t.co/x", []))
        self.assertFalse(self.rag.sourceExists("short-file"))

    def test_api_sources_survive_reconcile(self):
        self.service.ingestSource(Source("api-doc", "text", _TextV1, []))
        self.service.reconcile()
        self.assertTrue(self.rag.sourceExists("api-doc"))


class TestIngestionWorker(_IngestionCase):

    def test_processes_jobs_and_survives_failures(self):
        worker = IngestionWorker(self.service)
        worker.start()
        try:
            worker.submit("file", self._write("a.txt", _TextV1))
            worker.submit("bogus", None)
            worker.submit("source", Source("https://n/1", "news", _TextV2, []))
            worker.submit("delete", "https://n/1")
            worker.submit("file", self._write("b.txt", _TextV2))
            worker.jobs.join()
        finally:
            worker.stop()

        status = worker.snapshot()
        self.assertEqual(status["processed"], 4)
        self.assertNotIn("https://n/1", self.rag.chunks)
        self.assertEqual(status["failed"], 1)
        self.assertIn("bogus", status["lastError"])
        self.assertIn(self._title("b.txt"), self.rag.chunks)


class TestIsIgnoredFile(unittest.TestCase):

    def test_patterns(self):
        for name in (".hidden", "~lock.txt", "a.swp", "a.tmp", "a.txt~", "x.part"):
            self.assertTrue(isIgnoredFile(os.path.join("dir", name)), name)
        self.assertFalse(isIgnoredFile(os.path.join("dir", "notes.txt")))


if __name__ == '__main__':
    unittest.main()

"""
IngestionService turns source files into indexed chunks and keeps the indexes in
sync with the source folder.

Pipeline per file:
  1. Load the file and hash its content
  2. Compare with the manifest: unchanged → skip; changed or new → (re)index
  3. Clean, split into overlapping chunks, embed, tag
  4. Delete any previous chunks for the title, then store the new ones in every index

IngestionWorker serialises all writes through one background thread so the folder
watcher, the HTTP endpoints and startup reconciliation never write concurrently.
"""

import hashlib
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config import Config
from Controller.CategoryTagger import assignTags, precomputeLabelEmbeddings
from Controller.SearchData import cleanFeedText, cleanText, loadSource, sourceTitle
from DatabaseRepository.IngestManifestRepository import IngestManifestRepository
from DatabaseService.RagService import RagService
from Dto.Source import Source, SourceType

logger = logging.getLogger(__name__)

# Editor swap/backup files and hidden files are never indexed.
_IgnoredPrefixes: tuple[str, ...] = ('.', '~')
_IgnoredSuffixes: tuple[str, ...] = ('.swp', '.swx', '.tmp', '~', '.part', '.crdownload')


def isIgnoredFile(path: str) -> bool:
    """Return True for hidden, temporary and editor swap files."""
    name = os.path.basename(path)
    return name.startswith(_IgnoredPrefixes) or name.endswith(_IgnoredSuffixes)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class IngestionService:
    """Indexes, re-indexes and removes sources while keeping the manifest up to date."""

    def __init__(
        self,
        ragService: RagService,
        manifestRepository: IngestManifestRepository,
        model,
        sourceFolder: str = Config.Ingestion.SourceFolder,
        labels: list[str] | None = None,
    ):
        """
        model is a SentenceTransformer (anything with .encode(list[str]) -> ndarray).
        It must match Config.Model.EmbeddingModel to keep the vector space consistent.
        labels defaults to Categories.CATEGORY_LABELS.
        """
        if labels is None:
            from Categories import CATEGORY_LABELS
            labels = CATEGORY_LABELS
        self.ragService: RagService = ragService
        self.manifest: IngestManifestRepository = manifestRepository
        self.model = model
        self.sourceFolder: str = sourceFolder
        self.labels: list[str] = labels
        self.labelEmbeddings = precomputeLabelEmbeddings(model, labels)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.Model.MaxTextChunkLength,
            chunk_overlap=Config.Model.OverLapChunkLength,
        )

    def _folderPath(self, path: str) -> str:
        """
        Re-anchor a path on sourceFolder so titles match fetchDataFromFolder's
        format regardless of how the watcher spelled the path.
        """
        return os.path.join(self.sourceFolder, os.path.basename(path))

    def _index(self, source: Source) -> int:
        """Chunk, embed, tag and store one source, replacing any earlier chunks. Returns chunk count."""
        if self.ragService.sourceExists(source.title):
            self.ragService.deleteSource(source.title)

        # Local files get aggressive noise removal; feed items are short and link-heavy,
        # so they use a gentler cleaner that keeps short lines.
        cleaner = cleanText if source.sourceType == SourceType.File.value else cleanFeedText
        chunkTextList: list[str] = self.splitter.split_text(cleaner(source.content))
        if not chunkTextList:
            return 0

        sentenceEmbeddings = self.model.encode(chunkTextList)
        # Average chunk embeddings to represent the whole source, then tag once
        source.tags = assignTags(
            sentenceEmbeddings.mean(axis=0), self.labelEmbeddings, self.labels, Config.Model.TagTopN
        )
        for sentence, embedding in zip(chunkTextList, sentenceEmbeddings):
            chunk = Source(source.title, source.sourceType, sentence, source.tags, source.optional)
            # ChromaDB expects plain lists, not numpy arrays
            self.ragService.addSources([chunk], embedding.tolist())
        return len(chunkTextList)

    def ingestFile(self, path: str) -> str:
        """
        Index a file from the source folder if it is new or its content changed.
        Returns 'skipped', 'indexed' or 'removed' (file vanished before it was read).
        """
        path = self._folderPath(path)
        if not os.path.isfile(path):
            return self.removeFile(path)

        source: Source = loadSource(path)
        contentHash = _sha256(source.content)
        if self.manifest.getHash(source.title) == contentHash:
            return 'skipped'

        chunkCount = self._index(source)
        self.manifest.upsert(source.title, contentHash, chunkCount)
        logger.info("Indexed '%s' (%d chunks)", source.title, chunkCount)
        return 'indexed'

    def removeFile(self, path: str) -> str:
        """Remove a deleted file's chunks from every index and forget it in the manifest."""
        title = sourceTitle(self._folderPath(path))
        removed = self.ragService.deleteSource(title)
        self.manifest.delete(title)
        logger.info("Removed '%s' (%d chunks)", title, removed)
        return 'removed'

    def ingestSource(self, source: Source) -> int:
        """
        Index a source pushed over the API (not backed by a file). Always replaces
        earlier chunks with the same title. It is not recorded in the manifest, so
        reconcile() never deletes it for being absent from the folder.
        """
        return self._index(source)

    def deleteSource(self, title: str) -> int:
        """Remove a source pushed over the API (e.g. an expired feed item). Returns chunks removed."""
        return self.ragService.deleteSource(title)

    def reconcile(self) -> dict[str, int]:
        """
        Bring the indexes in line with the source folder: index new and edited
        files, and remove files that were deleted while the service was down.
        """
        counts: dict[str, int] = {'indexed': 0, 'skipped': 0, 'removed': 0}
        os.makedirs(self.sourceFolder, exist_ok=True)

        presentTitles: set[str] = set()
        for name in sorted(os.listdir(self.sourceFolder)):
            path = os.path.join(self.sourceFolder, name)
            if not os.path.isfile(path) or isIgnoredFile(path):
                continue
            presentTitles.add(sourceTitle(path))
            counts[self.ingestFile(path)] += 1

        for title in self.manifest.listTitles():
            if title not in presentTitles:
                self.ragService.deleteSource(title)
                self.manifest.delete(title)
                counts['removed'] += 1
                logger.info("Removed '%s' (file no longer present)", title)

        return counts


@dataclass
class IngestionStatus:
    """Snapshot of the worker's progress, exposed via GET /ingest/status."""
    queueDepth: int = 0
    lastFile: str | None = None
    lastAction: str | None = None
    lastError: str | None = None
    lastUpdated: float | None = None
    processed: int = 0
    failed: int = 0
    lastReconcile: dict[str, int] = field(default_factory=dict)


class IngestionWorker:
    """
    Single background thread that applies ingestion jobs one at a time.
    Jobs: ('file', path) | ('remove', path) | ('reconcile', None) | ('source', Source) | ('delete', title).
    A failed job is logged and recorded in status; it never stops the worker.
    """

    def __init__(self, service: IngestionService):
        self.service: IngestionService = service
        self.jobs: queue.Queue = queue.Queue()
        self.status: IngestionStatus = IngestionStatus()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def submit(self, kind: str, payload=None) -> None:
        """Queue a job for the worker thread."""
        self.jobs.put((kind, payload))

    def start(self) -> None:
        """Start the worker thread (daemon, so it never blocks interpreter exit)."""
        self._thread = threading.Thread(target=self._run, name="ingestion-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to finish its current job and exit."""
        self._stop.set()
        self.jobs.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict:
        """Thread-safe copy of the current status as a dict."""
        with self._lock:
            self.status.queueDepth = self.jobs.qsize()
            return dict(self.status.__dict__)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.jobs.get()
            if job is None:
                break
            kind, payload = job
            label = payload.title if isinstance(payload, Source) else payload
            try:
                result = self._apply(kind, payload)
                with self._lock:
                    self.status.processed += 1
                    self.status.lastAction = f"{kind}:{result}" if isinstance(result, str) else kind
                    if kind == 'reconcile':
                        self.status.lastReconcile = result
            except Exception as exc:
                logger.exception("Ingestion job %s(%s) failed", kind, label)
                with self._lock:
                    self.status.failed += 1
                    self.status.lastAction = f"{kind}:failed"
                    self.status.lastError = f"{kind}({label}): {exc}"
            finally:
                with self._lock:
                    self.status.lastFile = label
                    self.status.lastUpdated = time.time()
                self.jobs.task_done()

    def _apply(self, kind: str, payload):
        if kind == 'file':
            return self.service.ingestFile(payload)
        if kind == 'remove':
            return self.service.removeFile(payload)
        if kind == 'reconcile':
            return self.service.reconcile()
        if kind == 'source':
            self.service.ingestSource(payload)
            return 'indexed'
        if kind == 'delete':
            self.service.deleteSource(payload)
            return 'deleted'
        raise ValueError(f"Unknown ingestion job kind: {kind}")

"""
FolderWatcher watches the source folder and turns file-system events into
ingestion jobs.

Events are debounced per file: a job is submitted only after the file has been
quiet for DebounceSeconds, because editors usually write a file several times
per save. Every event becomes a single 'file' job; IngestionService.ingestFile
works out whether that means index, re-index, skip or remove.

Use polling (Config.Ingestion.UsePolling) when the folder lives on /mnt/c under
WSL2 or on a Windows bind mount in Docker; native inotify events never fire there.
"""

import logging
import os
import threading
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from Config import Config
from DatabaseService.IngestionService import isIgnoredFile

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Collapses bursts of events on the same file into one callback."""

    def __init__(self, folder: str, onChange: Callable[[str], None], debounceSeconds: float):
        super().__init__()
        self.folder: str = os.path.normcase(os.path.abspath(folder))
        self.onChange: Callable[[str], None] = onChange
        self.debounceSeconds: float = debounceSeconds
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _isWatchedFile(self, path: str) -> bool:
        # Only direct children of the folder are sources (matches fetchDataFromFolder).
        parent = os.path.normcase(os.path.dirname(os.path.abspath(path)))
        return parent == self.folder and not isIgnoredFile(path)

    def _schedule(self, path: str) -> None:
        if not self._isWatchedFile(path):
            return
        key = os.path.normcase(os.path.abspath(path))
        with self._lock:
            existing = self._timers.get(key)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self.debounceSeconds, self._fire, args=(key, path))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _fire(self, key: str, path: str) -> None:
        with self._lock:
            self._timers.pop(key, None)
        try:
            self.onChange(path)
        except Exception:
            logger.exception("FolderWatcher callback failed for %s", path)

    def cancelAll(self) -> None:
        """Drop any pending debounced callbacks."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory or event.event_type in ('opened', 'closed_no_write'):
            return
        self._schedule(os.fsdecode(event.src_path))
        destPath = getattr(event, 'dest_path', None)
        if destPath:
            self._schedule(os.fsdecode(destPath))


class FolderWatcher:
    """Starts a watchdog observer on one folder and reports changed file paths."""

    def __init__(
        self,
        folder: str,
        onChange: Callable[[str], None],
        debounceSeconds: float = Config.Ingestion.DebounceSeconds,
        usePolling: bool = Config.Ingestion.UsePolling,
        pollIntervalSeconds: float = Config.Ingestion.PollIntervalSeconds,
    ):
        """onChange(path) is called once per file after its events settle."""
        self.folder: str = folder
        self.usePolling: bool = usePolling
        self.handler = _DebouncedHandler(folder, onChange, debounceSeconds)
        self.observer = PollingObserver(timeout=pollIntervalSeconds) if usePolling else Observer()

    def start(self) -> None:
        """Begin watching (non-recursive) in a background thread."""
        os.makedirs(self.folder, exist_ok=True)
        self.observer.schedule(self.handler, self.folder, recursive=False)
        self.observer.daemon = True
        self.observer.start()
        logger.info("Watching %s (%s)", self.folder, "polling" if self.usePolling else "native events")

    def stop(self) -> None:
        """Stop watching and discard pending callbacks."""
        self.handler.cancelAll()
        self.observer.stop()
        self.observer.join(timeout=5)

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileMovedEvent, DirCreatedEvent

from Controller.FolderWatcher import FolderWatcher, _DebouncedHandler


class _Recorder:
    def __init__(self):
        self.paths: list[str] = []
        self.event = threading.Event()

    def __call__(self, path: str):
        self.paths.append(os.path.basename(path))
        self.event.set()


class TestDebouncedHandler(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.recorder = _Recorder()
        self.handler = _DebouncedHandler(self.folder, self.recorder, debounceSeconds=0.2)

    def tearDown(self):
        self.handler.cancelAll()
        shutil.rmtree(self.folder, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.folder, name)

    def test_burst_of_events_fires_once(self):
        for _ in range(5):
            self.handler.on_any_event(FileModifiedEvent(self._path("a.txt")))
            time.sleep(0.02)
        time.sleep(0.5)
        self.assertEqual(self.recorder.paths, ["a.txt"])

    def test_different_files_fire_separately(self):
        self.handler.on_any_event(FileCreatedEvent(self._path("a.txt")))
        self.handler.on_any_event(FileCreatedEvent(self._path("b.txt")))
        time.sleep(0.5)
        self.assertEqual(sorted(self.recorder.paths), ["a.txt", "b.txt"])

    def test_ignores_temp_files_directories_and_subfolders(self):
        self.handler.on_any_event(FileModifiedEvent(self._path(".a.txt.swp")))
        self.handler.on_any_event(DirCreatedEvent(self._path("sub")))
        self.handler.on_any_event(FileCreatedEvent(os.path.join(self.folder, "sub", "x.txt")))
        time.sleep(0.5)
        self.assertEqual(self.recorder.paths, [])

    def test_move_fires_for_source_and_destination(self):
        self.handler.on_any_event(FileMovedEvent(self._path("old.txt"), self._path("new.txt")))
        time.sleep(0.5)
        self.assertEqual(sorted(self.recorder.paths), ["new.txt", "old.txt"])

    def test_move_out_of_folder_only_fires_for_source(self):
        outside = os.path.join(tempfile.gettempdir(), "elsewhere.txt")
        self.handler.on_any_event(FileMovedEvent(self._path("old.txt"), outside))
        time.sleep(0.5)
        self.assertEqual(self.recorder.paths, ["old.txt"])


class TestFolderWatcherPolling(unittest.TestCase):

    def test_detects_new_file_with_polling_observer(self):
        folder = tempfile.mkdtemp()
        recorder = _Recorder()
        watcher = FolderWatcher(folder, recorder, debounceSeconds=0.1, usePolling=True, pollIntervalSeconds=0.2)
        watcher.start()
        try:
            with open(os.path.join(folder, "new.txt"), "w") as f:
                f.write("hello")
            self.assertTrue(recorder.event.wait(5), "watcher did not report the new file")
            self.assertIn("new.txt", recorder.paths)
        finally:
            watcher.stop()
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Dto.RetrievalResult import RetrievalResult
from Dto.Source import Source
from retrievalApp import createApp


class TestRetrievalApp(unittest.TestCase):

    def setUp(self):
        self.ragService = MagicMock()
        self.ragService.query.return_value = RetrievalResult(
            ids=["c1"], documents=["chunk"], distances=[0.3], metadatas=[{"title": "t"}], embeddings=[[0.1]],
        )
        self.worker = MagicMock()
        self.worker.snapshot.return_value = {"processed": 2}
        self.client = createApp(self.ragService, self.worker).test_client()

    def test_query_returns_serialised_result(self):
        response = self.client.post("/query", json={"query": "q", "mode": "hybrid", "topK": 4})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["ids"], ["c1"])
        self.assertNotIn("embeddings", response.get_json())
        self.ragService.query.assert_called_once_with(
            "q", tags=None, topK=4, searchMode="hybrid", sourceTypes=None
        )

    def test_query_passes_source_types(self):
        self.client.post("/query", json={"query": "q", "sourceTypes": ["news", "sns"]})
        self.assertEqual(self.ragService.query.call_args.kwargs["sourceTypes"], ["news", "sns"])

    def test_query_rejects_unknown_source_type(self):
        response = self.client.post("/query", json={"query": "q", "sourceTypes": ["podcast"]})
        self.assertEqual(response.status_code, 400)

    def test_ingest_delete_queues_each_title(self):
        response = self.client.post("/ingest/delete", json={"titles": ["https://a", "https://b"]})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            [c.args for c in self.worker.submit.call_args_list],
            [("delete", "https://a"), ("delete", "https://b")],
        )

    def test_ingest_delete_rejects_bad_payload(self):
        self.assertEqual(self.client.post("/ingest/delete", json={"titles": []}).status_code, 400)
        self.assertEqual(self.client.post("/ingest/delete", json={"titles": [""]}).status_code, 400)

    def test_ingest_rejects_unknown_source_type(self):
        response = self.client.post("/ingest", json={"sources": [
            {"title": "a", "content": "x", "sourceType": "podcast"},
        ]})
        self.assertEqual(response.status_code, 400)

    def test_query_validates_input(self):
        self.assertEqual(self.client.post("/query", json={"query": ""}).status_code, 400)
        self.assertEqual(self.client.post("/query", json={"query": "q", "mode": "bogus"}).status_code, 400)
        self.assertEqual(self.client.post("/query", json={"query": "q", "topK": 99}).status_code, 400)

    def test_ingest_queues_each_source(self):
        response = self.client.post("/ingest", json={"sources": [
            {"title": "a", "content": "x"}, {"title": "b", "content": "y"},
        ]})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"queued": 2})
        kinds = [c.args[0] for c in self.worker.submit.call_args_list]
        self.assertEqual(kinds, ["source", "source"])
        self.assertIsInstance(self.worker.submit.call_args_list[0].args[1], Source)

    def test_ingest_rejects_bad_payload(self):
        self.assertEqual(self.client.post("/ingest", json={}).status_code, 400)
        self.assertEqual(self.client.post("/ingest", json={"sources": [{"title": "a"}]}).status_code, 400)
        self.worker.submit.assert_not_called()

    def test_rescan_queues_reconcile(self):
        self.assertEqual(self.client.post("/ingest/rescan").status_code, 202)
        self.worker.submit.assert_called_once_with("reconcile")

    def test_status_and_health(self):
        self.assertEqual(self.client.get("/ingest/status").get_json(), {"processed": 2})
        self.assertEqual(self.client.get("/health").get_json(), {"ok": True})


if __name__ == '__main__':
    unittest.main()

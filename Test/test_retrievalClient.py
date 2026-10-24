import os
import sys
import unittest
from unittest.mock import MagicMock

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Controller.RetrievalClient import RetrievalClient
from Dto.Source import Source


def _response(json: dict, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.json.return_value = json
    response.status_code = status
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return response


class TestRetrievalClientQuery(unittest.TestCase):

    def setUp(self):
        self.client = RetrievalClient(baseUrl="http://retrieval:5001/", timeoutSeconds=3)
        self.client.session = MagicMock()

    def test_parses_successful_response(self):
        self.client.session.post.return_value = _response({
            "ids": ["x"], "documents": ["doc"], "distances": [0.2], "metadatas": [{"title": "t"}],
        })
        result = self.client.query("hello", topK=3, searchMode="hybrid")
        self.assertEqual(result.ids, ["x"])
        self.assertEqual(result.documents, ["doc"])

    def test_sends_query_mode_and_topk(self):
        self.client.session.post.return_value = _response({"ids": []})
        self.client.query("hello", topK=3, searchMode="sparse")
        url = self.client.session.post.call_args.args[0]
        kwargs = self.client.session.post.call_args.kwargs
        self.assertEqual(url, "http://retrieval:5001/query")
        self.assertEqual(kwargs["json"], {"query": "hello", "mode": "sparse", "topK": 3})
        self.assertEqual(kwargs["timeout"], 3)

    def test_connection_error_returns_empty_result(self):
        self.client.session.post.side_effect = requests.ConnectionError("refused")
        self.assertTrue(self.client.query("hello").isEmpty())

    def test_server_error_returns_empty_result(self):
        self.client.session.post.return_value = _response({}, status=500)
        self.assertTrue(self.client.query("hello").isEmpty())


class TestRetrievalClientIngest(unittest.TestCase):

    def test_ingest_posts_sources_and_returns_count(self):
        client = RetrievalClient(baseUrl="http://retrieval:5001")
        client.session = MagicMock()
        client.session.post.return_value = _response({"queued": 1}, status=202)
        queued = client.ingest([Source("t", "text", "content", [])])
        self.assertEqual(queued, 1)
        payload = client.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["sources"][0]["title"], "t")


if __name__ == '__main__':
    unittest.main()

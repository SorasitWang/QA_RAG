import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Dto.RetrievalResult import RetrievalResult
from Dto.Source import Source


def _result(hybrid: bool = False) -> RetrievalResult:
    return RetrievalResult(
        ids=["a", "b"],
        documents=["doc a", "doc b"],
        distances=[0.1, 0.4],
        metadatas=[{"title": "ta"}, {"title": "tb"}],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        vectorDistances=[0.2, None] if hybrid else None,
        tfidfDistances=[None, 0.5] if hybrid else None,
    )


class TestRetrievalResultSerialisation(unittest.TestCase):

    def test_round_trip_preserves_fields(self):
        restored = RetrievalResult.fromDict(_result().toDict())
        self.assertEqual(restored.ids, ["a", "b"])
        self.assertEqual(restored.documents, ["doc a", "doc b"])
        self.assertEqual(restored.distances, [0.1, 0.4])
        self.assertEqual(restored.metadatas, [{"title": "ta"}, {"title": "tb"}])

    def test_round_trip_preserves_hybrid_distances(self):
        restored = RetrievalResult.fromDict(_result(hybrid=True).toDict())
        self.assertEqual(restored.vectorDistances, [0.2, None])
        self.assertEqual(restored.tfidfDistances, [None, 0.5])

    def test_embeddings_omitted_by_default(self):
        self.assertNotIn("embeddings", _result().toDict())

    def test_embeddings_included_on_request(self):
        restored = RetrievalResult.fromDict(_result().toDict(includeEmbeddings=True))
        self.assertEqual(restored.embeddings, [[0.1, 0.2], [0.3, 0.4]])

    def test_missing_embeddings_keep_lists_parallel(self):
        restored = RetrievalResult.fromDict(_result().toDict())
        self.assertEqual(len(restored.embeddings), len(restored.ids))

    def test_empty(self):
        self.assertTrue(RetrievalResult.empty().isEmpty())


class TestSourceFromDict(unittest.TestCase):

    def test_defaults(self):
        source = Source.fromDict({"title": "t", "content": "c"})
        self.assertEqual((source.title, source.sourceType, source.content, source.tags), ("t", "file", "c", []))

    def test_legacy_type_key_is_accepted(self):
        self.assertEqual(Source.fromDict({"title": "t", "content": "c", "type": "news"}).sourceType, "news")

    def test_unknown_source_type_raises(self):
        with self.assertRaises(ValueError):
            Source.fromDict({"title": "t", "content": "c", "sourceType": "podcast"})

    def test_missing_content_raises(self):
        with self.assertRaises(KeyError):
            Source.fromDict({"title": "t"})


if __name__ == '__main__':
    unittest.main()

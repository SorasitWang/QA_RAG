import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch

from Controller.LLMController import LLMController, FullReplyTurns, TruncateReplyChars


class TestGenerateQaPrompt(unittest.TestCase):

    def setUp(self):
        self.controller = LLMController()

    def test_empty_retrieved_data_returns_raw_query(self):
        result = self.controller.generateQAPrompt([], "What is AI?")
        self.assertEqual(result, "What is AI?")

    def test_with_data_includes_each_chunk_and_question(self):
        result = self.controller.generateQAPrompt(["chunk1", "chunk2"], "What is AI?")
        self.assertIn("Data 1:\nchunk1", result)
        self.assertIn("Data 2:\nchunk2", result)
        self.assertIn("Question: What is AI?", result)


class TestGenerateSummaryPrompt(unittest.TestCase):

    def setUp(self):
        self.controller = LLMController()

    def test_includes_all_content_items(self):
        result = self.controller.generateSummaryPrompt(["item1", "item2"])
        self.assertIn("Content 1:\nitem1", result)
        self.assertIn("Content 2:\nitem2", result)

    def test_empty_list_returns_base_prompt_only(self):
        result = self.controller.generateSummaryPrompt([])
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestHistoryLength(unittest.TestCase):

    def setUp(self):
        self.controller = LLMController()

    def test_empty_history_returns_zero(self):
        self.assertEqual(self.controller._historyLength(), 0)

    def test_with_exchanges_sums_both_sides(self):
        self.controller.exchanges = [("hello", "world"), ("foo", "bar")]
        self.assertEqual(self.controller._historyLength(), 16)


class TestCompactHistory(unittest.TestCase):

    def setUp(self):
        self.controller = LLMController()
        self.controller.exchanges = [("Q1", "A1"), ("Q2", "A2")]

    @patch('Controller.LLMController.chat')
    def test_resets_exchanges_to_single_summary(self, mock_chat):
        mock_chat.return_value = "summary"
        self.controller._compactHistory()
        self.assertEqual(len(self.controller.exchanges), 1)
        self.assertEqual(self.controller.exchanges[0][1], "summary")

    @patch('Controller.LLMController.chat')
    def test_summary_prompt_contains_conversation_content(self, mock_chat):
        mock_chat.return_value = "summary"
        self.controller._compactHistory()
        called_msg = mock_chat.call_args[0][0][0]
        self.assertIn("Q1", called_msg.userMessage)
        self.assertIn("A1", called_msg.userMessage)


class TestBuildChatMessages(unittest.TestCase):

    def setUp(self):
        self.controller = LLMController(systemMessage="sys")

    def test_no_exchanges_systemMessage_on_current_turn(self):
        messages = self.controller._buildChatMessages("Q1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].systemMessage, "sys")
        self.assertEqual(messages[0].userMessage, "Q1")

    def test_with_exchanges_systemMessage_on_first_exchange(self):
        self.controller.exchanges = [("Q1", "A1")]
        messages = self.controller._buildChatMessages("Q2")
        self.assertEqual(messages[0].systemMessage, "sys")
        self.assertIsNone(messages[1].systemMessage)

    def test_current_turn_has_no_assistant_message(self):
        self.controller.exchanges = [("Q1", "A1")]
        messages = self.controller._buildChatMessages("Q2")
        self.assertIsNone(messages[-1].assisstantMessage)

    def test_old_exchanges_truncated(self):
        long_reply = "x" * (TruncateReplyChars + 50)
        for i in range(FullReplyTurns + 2):
            self.controller.exchanges.append((f"Q{i}", long_reply))
        messages = self.controller._buildChatMessages("current")
        self.assertTrue(messages[0].assisstantMessage.endswith("..."))
        self.assertEqual(len(messages[0].assisstantMessage), TruncateReplyChars + 3)

    def test_recent_exchanges_not_truncated(self):
        long_reply = "x" * (TruncateReplyChars + 50)
        for i in range(FullReplyTurns + 1):
            self.controller.exchanges.append((f"Q{i}", long_reply))
        messages = self.controller._buildChatMessages("current")
        self.assertFalse(messages[-2].assisstantMessage.endswith("..."))


class TestGenerateResponse(unittest.TestCase):

    def setUp(self):
        self.controller = LLMController()

    @patch('Controller.LLMController.chat')
    def test_returns_llm_reply(self, mock_chat):
        mock_chat.return_value = "reply"
        result = self.controller.generateResponse("Q", [])
        self.assertEqual(result, "reply")

    @patch('Controller.LLMController.chat')
    def test_stores_plain_query_in_exchanges(self, mock_chat):
        mock_chat.return_value = "reply"
        self.controller.generateResponse("plain question", ["some data"])
        self.assertEqual(self.controller.exchanges[0][0], "plain question")

    @patch('Controller.LLMController.chat')
    def test_triggers_compaction_when_over_threshold(self, mock_chat):
        mock_chat.return_value = "reply"
        big_str = "x" * 3000
        self.controller.exchanges = [(big_str, big_str)]
        with patch.object(self.controller, '_compactHistory') as mock_compact:
            self.controller.generateResponse("Q", [])
            mock_compact.assert_called_once()

    @patch('Controller.LLMController.chat')
    def test_no_compaction_below_threshold(self, mock_chat):
        mock_chat.return_value = "reply"
        self.controller.exchanges = [("short", "short")]
        with patch.object(self.controller, '_compactHistory') as mock_compact:
            self.controller.generateResponse("Q", [])
            mock_compact.assert_not_called()


if __name__ == '__main__':
    unittest.main()

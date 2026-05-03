import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch, MagicMock


class TestChatSession(unittest.TestCase):

    @patch('DatabaseService.ChatService.LLMController')
    def test_ask_delegates_to_controller_generate_response(self, MockLLMController):
        mock_controller = MagicMock()
        MockLLMController.return_value = mock_controller
        mock_controller.generateResponse.return_value = "reply"

        from DatabaseService.ChatService import ChatSession
        session = ChatSession()
        result = session.ask("hello")

        mock_controller.generateResponse.assert_called_once_with("hello", [])
        self.assertEqual(result, "reply")

    @patch('DatabaseService.ChatService.LLMController')
    def test_init_passes_systemMessage_to_controller(self, MockLLMController):
        from DatabaseService.ChatService import ChatSession
        ChatSession(systemMessage="be concise")
        MockLLMController.assert_called_once_with(systemMessage="be concise")

    @patch('DatabaseService.ChatService.LLMController')
    def test_init_with_no_systemMessage_passes_none(self, MockLLMController):
        from DatabaseService.ChatService import ChatSession
        ChatSession()
        MockLLMController.assert_called_once_with(systemMessage=None)


if __name__ == '__main__':
    unittest.main()

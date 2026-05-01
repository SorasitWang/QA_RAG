"""
ChatService provides a simple session-based chat interface.
It delegates to LLMController so all history management and LLM interaction
happen in one place; ChatSession is purely a convenience wrapper.
"""

from Controller.LLMController import LLMController


class ChatSession:
    """
    Stateful chat session for plain (non-RAG) conversations.
    Each instance maintains its own history through the underlying LLMController.
    For RAG-augmented conversations use LLMController directly and pass
    retrieved chunks to generateResponse().
    """

    def __init__(self, systemMessage: str | None = None):
        self.controller = LLMController(systemMessage=systemMessage)

    def ask(self, message: str) -> str:
        """Sends a message and returns the LLM reply. History is maintained automatically."""
        return self.controller.generateResponse(message, [])

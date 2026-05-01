"""
ChatMessage is the unit of conversation passed to the LLM API wrapper.
One ChatMessage groups together the optional system instruction, the user
turn, and the assistant's reply to that turn, matching the structure that
ollama.py flattens into the role-based message list.
"""

from dataclasses import dataclass


@dataclass
class ChatMessage:
    """
    Represents one exchange in the conversation history.

    systemMessage   — instruction prompt; only used on the first message of a session.
    userMessage     — the user's input or an augmented RAG prompt for the current turn.
    assisstantMessage — the LLM's reply to the previous turn (note: field name typo
                        kept intentionally to match existing serialisation across the app).
    """
    systemMessage: str | None = None
    userMessage: str = ''
    assisstantMessage: str | None = None

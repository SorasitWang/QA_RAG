"""
Thin wrapper around the Ollama Python SDK.
Converts the internal ChatMessage list into the role-based message format
that Ollama's /api/chat endpoint expects, then returns the plain text reply.
"""

from ollama import chat as ollama_chat
from ollama import ChatResponse

from Dto.ChatMessage import ChatMessage
from Config import Config


def chat(chatMessageList: list[ChatMessage]) -> str:
    """
    Sends a conversation history to the configured Ollama model and returns
    the assistant's reply as a plain string.

    Each ChatMessage maps to up to three Ollama messages in this order:
        [system (optional)] → user → assistant (optional)
    The system message is only expected on the first ChatMessage in the list.
    The assistant message carries the reply from the previous turn so the model
    understands the conversation so far.
    """
    messages = []

    for chatMessage in chatMessageList:
        if chatMessage.systemMessage is not None:
            messages.append({
                'role': 'system',
                'content': chatMessage.systemMessage
            })

        messages.append({
            'role': 'user',
            'content': chatMessage.userMessage
        })

        if chatMessage.assisstantMessage is not None:
            messages.append({
                'role': 'assistant',
                'content': chatMessage.assisstantMessage
            })

    response: ChatResponse = ollama_chat(model=Config.Model.LLMModel, messages=messages)

    return response.message.content

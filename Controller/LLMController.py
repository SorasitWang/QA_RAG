"""
LLMController orchestrates the full RAG chat pipeline:
  - builds prompts (QA and summary)
  - maintains per-session conversation history
  - truncates old assistant replies to keep the context focused
  - compacts history with a summary when it grows too large
"""

from Config import Config
from Dto.ChatMessage import ChatMessage
from LLMApi.ollama import chat


# How many of the most recent exchanges keep their full assistant reply.
# Older ones are truncated to reduce noise in the LLM context.
FullReplyTurns     = 2
TruncateReplyChars = 200


class LLMController:
    """
    Manages conversation state and delegates LLM calls.
    Each exchange stores (query, reply, retrieved_chunks, distances) so the
    debug view can inspect what the retrieval pipeline actually sent.
    """

    def __init__(self, systemMessage: str | None = None):
        self.systemMessage = systemMessage
        # Each entry: (plain_query, reply, chunks_passed_to_llm, distances)
        # We store the plain query (not the augmented prompt) so history stays
        # concise and previous RAG data blocks don't repeat on every turn.
        self.exchanges: list[tuple[str, str, list[str], list[float]]] = []
        # Set to True on turns where _compactHistory() fires; reset each turn.
        self.lastCompacted: bool = False
        self.lastCompactedSummary: str | None = None

    def generateQAPrompt(self, retrievedData: list[str], userQuery: str) -> str:
        """
        Wraps the user query with retrieved context chunks.
        Returns the raw query unchanged when no chunks are available,
        letting the LLM fall back to its own knowledge.
        """
        if not retrievedData:
            return userQuery
        prompt = "You are a helpful assistant. Use the following retrieved data to answer the question.\n\n"
        for i, data in enumerate(retrievedData):
            prompt += f"Data {i+1}:\n{data}\n\n"
        prompt += f"Question: {userQuery}"
        return prompt

    def generateSummaryPrompt(self, contentList: list[str]) -> str:
        """Builds a prompt that asks the LLM to summarise a list of content blocks."""
        prompt = "You are a helpful assistant. Summarize the following content:\n\n"
        for i, content in enumerate(contentList):
            prompt += f"Content {i+1}:\n{content}\n\n"
        return prompt

    def _historyLength(self) -> int:
        """Total character count of all stored queries and replies."""
        return sum(len(u) + len(a) for u, a, *_ in self.exchanges)

    def _compactHistory(self):
        """
        Replaces the full exchange list with a single LLM-generated summary.
        Called when history exceeds MaxChatHistoryChars to keep prompt size
        within reasonable bounds without losing conversation context entirely.
        """
        turns = [
            f"User: {userMsg}\nAssistant: {assistantMsg}"
            for userMsg, assistantMsg, *_ in self.exchanges
        ]
        summaryPrompt = self.generateSummaryPrompt(turns)
        summary = chat([ChatMessage(userMessage=summaryPrompt)])
        # Reset to a single placeholder exchange carrying the summary as the reply
        self.exchanges = [("Summary of previous conversation:", summary, [], [])]
        self.lastCompacted = True
        self.lastCompactedSummary = summary

    def _buildChatMessages(self, currentUserMsg: str) -> list[ChatMessage]:
        """
        Converts stored exchanges into the ChatMessage list expected by ollama.chat().
        The structure per ChatMessage is [system?] → user → assistant?, which the
        ollama wrapper flattens into the correct alternating role sequence.

        Older exchanges (beyond FullReplyTurns from the end) have their assistant
        replies truncated so the LLM focuses on the current question rather than
        being distracted by verbose earlier answers.
        """
        messages = []
        cutoff = len(self.exchanges) - FullReplyTurns
        for i, (userMsg, assistantMsg, *_) in enumerate(self.exchanges):
            # Truncate assistant reply for exchanges outside the recent window
            if i < cutoff and len(assistantMsg) > TruncateReplyChars:
                assistantMsg = assistantMsg[:TruncateReplyChars] + "..."
            msg = ChatMessage()
            # System message is attached only to the first message of the conversation
            if i == 0 and self.systemMessage:
                msg.systemMessage = self.systemMessage
            msg.userMessage = userMsg
            msg.assisstantMessage = assistantMsg
            messages.append(msg)

        # Current (unanswered) turn — no assistant message yet
        current = ChatMessage()
        if not self.exchanges and self.systemMessage:
            current.systemMessage = self.systemMessage
        current.userMessage = currentUserMsg
        messages.append(current)
        return messages

    def rewriteQuery(self, query: str) -> str:
        """
        Rewrites a vague or referential query into a standalone search string
        using the last two user turns from history.
        Only called when initial retrieval returns no results.
        Returns the original query unchanged if there is no history to draw from.
        """
        if not self.exchanges:
            return query

        recentTurns = self.exchanges[-2:]
        historyText: str = '\n'.join(f"User: {u}" for u, *_ in recentTurns)
        prompt = (
            f"Conversation so far:\n{historyText}\n\n"
            f"Rewrite this follow-up question as a concise standalone search query. "
            f"Output only the rewritten query, nothing else.\n"
            f"Follow-up: {query}"
        )
        return chat([ChatMessage(userMessage=prompt)])

    def generateResponse(self, query: str, retrievedData: list[str], distances: list[float] = None) -> str:
        """
        Main entry point for a single conversation turn.
        Compacts history if needed, builds the augmented prompt, calls the LLM,
        and records the exchange (with raw query, not the augmented prompt, so
        history stays compact across turns).
        """
        self.lastCompacted = False
        self.lastCompactedSummary = None
        if self._historyLength() > Config.Model.MaxChatHistoryChars:
            self._compactHistory()

        augmented = self.generateQAPrompt(retrievedData, query)
        reply = chat(self._buildChatMessages(augmented))
        self.exchanges.append((query, reply, retrievedData, distances or []))
        return reply

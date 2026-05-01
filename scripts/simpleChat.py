"""
simpleChat — minimal interactive chat bot without RAG.
Useful for testing the LLM connection and conversation history in isolation
before adding retrieval into the loop.
"""

from DatabaseService.ChatService import ChatSession


def main():
    chatSession = ChatSession()

    print("Chat bot ready. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            userInput = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not userInput:
            continue

        if userInput.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        reply = chatSession.ask(userInput)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate


class Chatbot:
    def __init__(self, model: Ollama, system: ChatPromptTemplate):
        self.Model = model
        self.System = system
        self.History: list[dict[str, str]] = ChatMessageHistory()
        self.continue_chat = True

    def get_completion(self, prompt: str):
        """Gets a basic chat completion with chat history"""

        print("Thinking...")
        chain = self.System | self.Model
        self.History.add_user_message(prompt)

        model_input = {"messages": self.History.messages}

        completion: str = chain.invoke(model_input)
        self.History.add_ai_message(completion)

        return completion

    def get_history(self) -> list[dict[str, str]]:
        """Returns chat history"""
        return self.History.messages

    def run(self):
        print("Type 'goodbye' to end the chat.")

        while self.continue_chat:
            prompt = input("Please enter your prompt:")
            if prompt.strip().lower() != "goodbye":
                completion = self.get_completion(prompt)
                print(f"Output: {completion}")

            else:
                self.continue_chat = False
                log = self.get_history()
                print("The chat has ended.")
                print("Chat log:")
                print(f"Length of chat: {len(log)}")
                print(log)

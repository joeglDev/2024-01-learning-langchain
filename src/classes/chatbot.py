from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory


class Chatbot:
    def __init__(self, model: Ollama, system: ChatPromptTemplate):
        self.Model = model
        self.System = system
        self.History: list[dict[str, str]] = ChatMessageHistory()

    def get_completion(self, prompt: str):
        """Gets a basic chat completion with chat history"""

        chain = self.System | self.Model
        self.History.add_user_message(prompt)

        model_input = {"messages": self.History.messages}

        completion: str = chain.invoke(model_input)
        self.History.add_ai_message(completion)

        return completion

    def get_history(self) -> list[dict[str, str]]:
        """Returns chat history"""
        return self.History.messages

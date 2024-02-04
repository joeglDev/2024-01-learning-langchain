from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


class Chatbot:
    def __init__(self, model: Ollama, system: ChatPromptTemplate):
        self.Model = model
        self.System = system
        self.History: list[dict[str, str]] = []

    def get_completion(self, prompt: str):
        """Gets a basic chat completion with chat history"""
        if len(self.History) > 0:
            model_input = [
                HumanMessage(content=prompt),
            ]
            for chat_round in self.History:
                ai = AIMessage(content=chat_round["completion"])
                human = HumanMessage(content=chat_round["prompt"])
                model_input.append(ai)
                model_input.append(human)

            completion: str = self.Model.invoke(model_input)
            self.create_chat_history(prompt, completion)
            print(f"Output: {completion}")

        else:
            model_input = [
                HumanMessage(content=prompt),
            ]

            completion: str = self.Model.invoke(model_input)
            self.create_chat_history(prompt, completion)
            print(f"Output: {completion}")

    def create_chat_history(self, prompt: str, completion: str):
        """Appends prompt and completion to chat history"""
        chat_round = {"prompt": prompt, "completion": completion}
        self.History.append(chat_round)

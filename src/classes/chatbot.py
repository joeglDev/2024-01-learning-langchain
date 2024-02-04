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
        chain = self.System | self.Model
        model_input = {
            "messages": [
                HumanMessage(content=prompt),
            ]
        }

        if len(self.History) > 0:
            for chat_round in self.History:
                chat_history = model_input["messages"]
                ai = AIMessage(content=chat_round["completion"])
                human = HumanMessage(content=chat_round["prompt"])
                chat_history.append(ai)
                chat_history.append(human)

            completion: str = chain.invoke(model_input)
            self.create_chat_history(prompt, completion)
            print(f"Output: {completion}")

        else:
            completion: str = chain.invoke(model_input)
            self.create_chat_history(prompt, completion)
            print(f"Output: {completion}")

    def create_chat_history(self, prompt: str, completion: str):
        """Appends prompt and completion to chat history"""
        chat_round = {"prompt": prompt, "completion": completion}
        self.History.append(chat_round)

    def get_history(self) -> list[dict[str, str]]:
        """Returns chat history"""
        return self.History

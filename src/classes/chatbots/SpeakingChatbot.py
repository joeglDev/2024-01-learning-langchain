from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.classes.chatbots.chatbot import Chatbot


class SpeakingChatbot(Chatbot):
    def __init__(self, model: Ollama, system: ChatPromptTemplate):
        super().__init__(model, system)


from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from src.classes.chatbots.SpeakingChatbot import SpeakingChatbot

# setup large language model parameters
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are not a helpful ai.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chat = SpeakingChatbot(model=Ollama(model="syx"), system=system_prompt)
chat.run()

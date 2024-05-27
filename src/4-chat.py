from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from src.classes.chatbot import Chatbot

# setup large language model parameters
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your name is Anakin Skywalker.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chat = Chatbot(model=Ollama(model="orca-mini"), system=system_prompt)
chat.run()

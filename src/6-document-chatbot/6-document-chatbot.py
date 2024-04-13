from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from src.classes.chatbot import Chatbot
from prompts import initial_prompt

# Create LLM
llm = Ollama(model='orca-mini')

system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            initial_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chat = Chatbot(model=llm, system=system_prompt)




from langchain_community.llms.ollama import Ollama

from src.classes.chatbot import Chatbot

chat = Chatbot(model=Ollama(model="orca-mini"))

# first completion so no history
chat.get_completion()

# second completion so has history
chat.get_completion()

# third completion so has history
chat.get_completion()

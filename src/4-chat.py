from langchain_community.llms.ollama import Ollama

from src.classes.chatbot import Chatbot

chat = Chatbot(model=Ollama(model="orca-mini"))
CONTINUE_CHAT = True

while CONTINUE_CHAT:
    should_continue_chat = input("Y = continue chat, N = end chat")
    if should_continue_chat.lower() == "y":
        prompt = input("Please enter a prompt:")
        chat.get_completion(prompt)
    else:
        CONTINUE_CHAT = False
        print("The chat has ended.")

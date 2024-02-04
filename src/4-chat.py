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

CONTINUE_CHAT = True

while CONTINUE_CHAT:
    should_continue_chat = input("Y = continue chat, N = end chat")
    if should_continue_chat.strip().lower() == "y":
        prompt = input("Please enter a prompt:")
        chat.get_completion(prompt)
    else:
        CONTINUE_CHAT = False
        log = chat.get_history()
        print("The chat has ended.")
        print("Chat log:")
        print(f"Length of chat: {len(log)}")
        print(log)

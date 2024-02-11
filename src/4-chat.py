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
chat = Chatbot(model=Ollama(model="mistral"), system=system_prompt)

CONTINUE_CHAT = True

print("Type 'goodbye' to end the chat.")

while CONTINUE_CHAT:
    prompt = input("Please enter your prompt:")
    if prompt.strip().lower() != "goodbye":
        completion = chat.get_completion(prompt)
        print(f"Output: {completion}")

    else:
        CONTINUE_CHAT = False
        log = chat.get_history()
        print("The chat has ended.")
        print("Chat log:")
        print(f"Length of chat: {len(log)}")
        print(log)

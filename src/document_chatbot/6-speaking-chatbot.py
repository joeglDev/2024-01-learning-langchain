from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from src.classes.chatbots.SpeakingChatbot import SpeakingChatbot
from src.classes.chatbots.prompts import SYX_SYSTEM_PROMPT

# setup large language model parameters
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYX_SYSTEM_PROMPT,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chat = SpeakingChatbot(model=Ollama(model="mistral"), system=system_prompt)
chat.run()

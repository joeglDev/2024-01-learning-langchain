from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from prompts import SYSTEM_TEMPLATE
from src.classes.document_qa_chatbot import DocumentQaChatbot


llm = Ollama(model="orca-mini")

messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)

file_path = "/../data/fantasy-one-page-rules-rulebook.pdf"

chatbot = DocumentQaChatbot(llm=llm, file_path=file_path, prompt=prompt)
chatbot.run()

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from prompts import SYSTEM_TEMPLATE, CONTEXTUALIZED_QA_SYSTEM_PROMPT
from src.classes.document_qa_chatbot import DocumentQaChatbot


llm = Ollama(model="mistral")

messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

contextualized_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZED_QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

file_path = "/../data/fantasy-one-page-rules-rulebook.pdf"

chatbot = DocumentQaChatbot(
    llm=llm, file_path=file_path, prompt=prompt, qa_prompt=contextualized_qa_prompt
)
chatbot.run()

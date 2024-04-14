import os
from typing import Tuple

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.classes.document_loader import DocumentLoader


class DocumentQaChatbot:
    def __init__(self, llm: Ollama, file_path: str, prompt: ChatPromptTemplate):
        self.llm = llm
        self.file_path: str = file_path
        self.prompt: ChatPromptTemplate = prompt
        self.History: ChatMessageHistory = ChatMessageHistory()
        self.CONTINUE_CHAT = True
        self.document_retriever: VectorStoreRetriever | None = None
        self.vectorstore: Chroma | None = None
        self.chain: RunnableSerializable[str, str] | None = None

    def load_document(self) -> list[Document]:
        print("Reading document")

        file_path = os.path.abspath(os.getcwd()) + self.file_path
        loader = DocumentLoader(file_path=file_path)
        docs = loader.run()
        return docs

    def create_vectorstore(
        self, docs: list[Document]
    ) -> Tuple[VectorStoreRetriever, Chroma]:
        print("Creating vectorstore")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embedding_function
        )
        retriever = vectorstore.as_retriever()

        return retriever, vectorstore

    def format_docs(self, docs: list[Document]) -> str:
        page_content = "\n\n".join(doc.page_content for doc in docs)
        return page_content

    def get_history(self) -> ChatMessageHistory:
        """Returns chat history"""
        return self.History.messages

    def get_completion(self, prompt: str):
        """Gets a basic chat completion with chat history"""

        print("Generating response...")
        self.History.add_user_message(prompt)

        completion: str = self.chain.invoke(prompt)
        self.History.add_ai_message(completion)

        return completion

    def chat(self):
        print("Type 'exit' and press enter key to quit this application.")
        while self.CONTINUE_CHAT:
            question = input("Please enter a prompt and press enter key:")
            if question.strip().lower() == "exit":
                self.CONTINUE_CHAT = False
                log = self.get_history()
                print("The chat has ended.")
                print("Chat log:")
                print(f"Length of chat: {len(log)}")
                print(log)
            else:
                completion = self.get_completion(question)
                print(f"Output: {completion}")

    def run(self):
        docs = self.load_document()
        retriever, vectorstore = self.create_vectorstore(docs)
        self.document_retriever = retriever
        self.vectorstore = vectorstore

        chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        self.chain = chain

        self.chat()

        # cleanup
        print("Cleaning up vectorstore")
        vectorstore.delete_collection()

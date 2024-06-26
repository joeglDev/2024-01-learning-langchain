import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


from src.classes.document_loader import DocumentLoader
from src.document_chatbot.prompts import QA_SYSTEM_PROMPT


class DocumentQaChatbot:
    def __init__(
        self,
        llm: Ollama,
        file_path: str,
        prompt: ChatPromptTemplate,
        qa_prompt: ChatPromptTemplate,
    ):
        self.llm = llm
        self.file_path: str = file_path
        self.prompt: ChatPromptTemplate = prompt
        self.qa_prompt: ChatPromptTemplate = qa_prompt
        self.History: ChatMessageHistory = ChatMessageHistory()
        self.CONTINUE_CHAT = True
        self.chain: RunnableSerializable[str, str] | None = None
        self.retriever: VectorStoreRetriever | None = None

    def load_document(self) -> list[Document]:
        print("Reading document")

        file_path = os.path.abspath(os.getcwd()) + self.file_path
        loader = DocumentLoader(file_path=file_path)
        docs = loader.run()
        return docs

    def create_vectorstore(self, docs: list[Document]) -> Chroma:
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

        return vectorstore

    def format_docs(self, docs: list[Document]) -> str:
        page_content = "\n\n".join(doc.page_content for doc in docs)
        return page_content

    def get_history(self) -> ChatMessageHistory:
        """Returns chat history"""
        return self.History.messages

    def get_completion(self, question: str) -> dict:
        """Gets a basic chat completion with chat history"""

        print("Generating response...")
        self.History.add_user_message(question)

        completion: dict = self.chain.invoke(
            {"input": question, "chat_history": self.get_history()}
        )
        self.History.add_ai_message(completion["answer"])

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
                print(f"Output: {completion['answer']}")
                print()
                for doc in completion["context"]:
                    print(f"Source: {doc}")
                print()

    def create_chain(self):
        # handle question answer history
        # prepends a rephrasing of the input query to our retriever, so that the retrieval incorporates the context of the conversation
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.qa_prompt
        )

        # create qa chain
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.chain = chain

    def run(self):
        docs = self.load_document()
        vectorstore = self.create_vectorstore(docs)
        self.retriever = vectorstore.as_retriever()
        self.create_chain()

        self.chat()

        # cleanup
        print("Cleaning up vectorstore")
        vectorstore.delete_collection()

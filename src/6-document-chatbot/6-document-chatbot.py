import os

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import SYSTEM_TEMPLATE
from src.classes.document_loader import DocumentLoader

# Create LLM
llm = Ollama(model="orca-mini")

# Extract document content
print("Reading document")
file_path = (
    os.path.abspath(os.getcwd()) + "/../data/fantasy-one-page-rules-rulebook.pdf"
)
loader = DocumentLoader(file_path=file_path)
docs = loader.run()

# Create vectorstore
print("Creating vectorstore")
# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed the chunks
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Store the embeddings in a vector database
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

# Create a retriever from the vector database
retriever = vectorstore.as_retriever()


def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


# Create an LLM chain
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
question = "Give me the rules on mobility."

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print('Generating LLM response')
response = chain.invoke(question)
print(response)
# cleanup
vectorstore.delete_collection()

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.classes.chatbot import Chatbot
from src.classes.webscrapers import Webscrapers


url = input("Please enter a website url here:")
query = input("Please enter your question here:")
print(f"Query: {url} -> {query}")

web = Webscrapers()
web_content = web.extract_text_from(url)

# chat
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"The following text was scraped from the websites {url}: {web_content}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chat = Chatbot(model=Ollama(model="mistral"), system=system_prompt)

completion = chat.get_completion(query)
print(f"Output: {completion}")

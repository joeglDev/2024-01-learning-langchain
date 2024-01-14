from langchain_community.llms import Ollama

llm = Ollama(model="orca-mini")


def get_response_from_prompt() -> str:
    query: str = input("Please input a prompt here:")
    response: str = llm.invoke(query)

    return response


print(get_response_from_prompt())

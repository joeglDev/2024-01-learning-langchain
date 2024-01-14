from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Using orca-mini for testing purposes as fast.
# Usually I prefer Mistral with a complex system prompt 2024-01-14.
llm = Ollama(model="orca-mini")


def get_response_from_prompt() -> str:
    """Returns a simple completion from a prompt"""

    query: str = input("Please input a prompt here:")
    response: str = llm.invoke(query)

    return response


def get_response_from_prompt_with_system_prompt() -> str:
    """Adds a system prompt to a returned response."""

    query: str = input("Please input a prompt here:")
    prompt_with_system_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are the jedi Obi-Won Kenobi."), ("user", "{input}")]
    )

    chain = prompt_with_system_prompt | llm
    response: str = chain.invoke({"input": f"{query}"})

    return response


def response_with_output_parser() -> str:
    """Output parser formats the streamed output of the LLM e.g. as a string"""
    output_parser = StrOutputParser()

    query: str = input("Please input a prompt here:")
    prompt_with_system_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are the grey jedi Anakin Skywalker."), ("user", "{input}")]
    )

    chain = prompt_with_system_prompt | llm | output_parser
    response: str = chain.invoke({"input": f"{query}"})

    return response


print(response_with_output_parser())

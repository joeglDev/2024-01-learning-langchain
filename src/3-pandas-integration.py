import os
import pandas as pd

from langchain_community.llms import Ollama  # pylint: disable=E0611
from langchain_experimental.agents import create_pandas_dataframe_agent

DATAFRAME_PATH = "/data/example_csv.csv"
llm = Ollama(model="mistral")  # needs a more powerful model to work correctly


def get_pandas_dataframe_from_csv(dataframe_path: str) -> pd.DataFrame:
    path = os.path.abspath(os.getcwd()) + dataframe_path
    example_df = pd.read_csv(path)
    print(example_df)
    return example_df


def analyse_dataframe(df: pd.DataFrame, prompt: str):
    pd_agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, handle_parsing_errors=True
    )
    pd_agent.invoke(prompt)


analyse_dataframe(
    get_pandas_dataframe_from_csv(DATAFRAME_PATH),
    "Please find each jedi with a power level of less than 10.",
)

"""
Notes:
List of prompts and answers
1. How many rows are there in the dataframe? -> 3
2. Which jedi have a blue lightsaber_colour? -> The jedi with a blue lightsaber is Obi-Wan Kenobi.
3. Which jedi has a value of blue in previous_lightsaber_colour? -> # Note: Failed as as colour vs color was an issue.
4. Which jedi has a value of blue in previous_lightsaber_colour? -> # Note: failed to find the column.
5. Which jedi previously had a lightsaber of colour blue? -> # Note: Failed as as colour vs color was an issue.
6. Please find each jedi with a power level of less than 10. -> The Jedi with a power level of less than 10 are Obi Won Kenobi and Ahsoka Tano.
"""

from dataclasses import dataclass

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama


@dataclass
class RecordData:
    catalog_number: str
    artist: str
    title: str
    released: int
    genre: str


# load csv data
# example row: "V 3139, 00602547476241",Chvrches,Every Open Eye,"Virgin EMI Records, Goodbye Records (2)","LP, Album, Whi",,2015,7492694,Uncategorized,2025-02-17 06:15:48,Very Good (VG),Good (G),Joe,Pop
loader = CSVLoader(file_path="./data/discogs.csv")

data = loader.load()
record_context = []

# generate context string
for row in data:
    page_content = row.page_content.split("\n")
    page_content_data = [item.split(":")[1].strip() for item in page_content]

    record = RecordData(
        catalog_number=page_content_data[0],
        artist=page_content_data[1],
        title=page_content_data[2],
        released=page_content_data[6],
        genre=page_content_data[13],
    )
    record_context.append(record)


context_list = [
    f"Catalog number: {record.catalog_number} artist: {record.artist} title: {record.title} released: {record.released} genre: {record.genre}"
    for record in record_context
]
context_string = "\n".join(context_list)

# prompt
prompt = f"""
You are a audiophile and vinyl record collector familiar with many genres of music.
Please comment upon this individuals collection of music records. 
Each line represents a different LP.
After commenting on the composition of the collection as a whole please comment on some specific albums from it:
{context_string}
"""

# llm
llm = Ollama(model="mistral")
completion = llm.invoke(prompt)
print(completion)

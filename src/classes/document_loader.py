import mimetypes
from enum import Enum

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document


class MimeTypes(Enum):
    PDF = "application/pdf"


class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.docs: list[Document] | None = None

    def get_file_type(self) -> str | None:
        """
        Checks the document type of file.

        Parameters:
        file_path (str): The path to the file.

        Returns:
        str: The MIME type of the document, or None if the type cannot be determined.
        """
        mime_type, _ = mimetypes.guess_type(self.file_path)
        return mime_type

    def extract_pdf_content(self) -> list[Document]:
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()

        return pages

    def run(self) -> list[Document]:
        file_type = self.get_file_type()
        if file_type == MimeTypes.PDF.value:
            pages = self.extract_pdf_content()
            self.docs = pages

        return self.docs

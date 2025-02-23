from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader


class Webscrapers:
    def extract_text_from(self, url: str) -> str:
        print(f"Fetching web content from {url}")
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        content = docs_transformed[0].page_content[0:3000]
        return content

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 400):
        """
        Initialize the document processor

        Args:
                chunk_size (int, optional): Defaults to 1000. The chunk size to use.
                chunk_overlap (int, optional): Defaults to 200. The chunk overlap to use.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # separators=["\n\n", "\n", " ", ""],
        )

    def load_documents(self, path: str) -> List[Document]:
        """
        Load the documents from the directory

        Args:
            directory (str): The directory to load the documents from
        Returns:
            List[Document]: The list of documents
        """
        # Load txt files
        loader = TextLoader(path)
        documents = loader.load()

        # Add boardgame name in metadata
        for doc in documents:
            doc.metadata['game'] = os.path.basename(path).strip(".txt")

        print(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split the documents into chunks

        Args:
            documents (List[Document]): The list of documents to split
        Returns:
            List[Document]: The list of documents split into chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def process_documents(self, path:str) -> List[Document]:
        """
        Complete pipeline: load and split documents.

        Args:
            path: Path to directory containing text files

        Returns:
            List of processed document chunks
        """
        documents = self.load_documents(path)
        chunks = self.split_documents(documents)
        return chunks
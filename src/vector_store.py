import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import chromadb
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

class VectorStoreManager:
    def __init__(self, persist_directory: str, embeddings_model: str = 'text-embedding-3-small'):
        """
        Initialize the vector store manager.

        Args
            persist_directory (str): The directory to save the vector store
            embeddings_model (str): The model to create/load the embeddings from.
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.vectorstore: Dict[str, Chroma] = {}

        # Initialize a chroma db client for checking collections
        self.client = chromadb.PersistentClient(path=persist_directory)

    def _get_game_persist_path(self, game_name: str) -> str:
        """Get the persist path for a specific game collection."""
        return os.path.join(self.persist_directory, game_name)

    def _collection_exists(self, game_name: str) -> bool:
        try:
            self.client.get_collection(game_name)
            return True
        except Exception:
            return False

    def _delete_collection(self, game_name:str) -> None:
        try:
            self.client.delete_collection(game_name)
            print(f"Collection for {game_name} deleted")

            if game_name in self.vectorstore:
                del self.vectorstore[game_name]
        except Exception as e:
            print(f"Error deleting collection for {game_name} : {e}")

    def list_collections(self) -> List[str]:
        """
        List all existing game collections.

        Returns:
            List of game names with collections
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            return []

    def create_vectorstore(self, game_name:str, documents: List[Document], force_rebuild: bool = False) -> Chroma:
        """
        Create the vector store

        Args
            game_name (str): The name of the game to create the vector store for
            documents (List[Document]): The documents to create the vector store.
            force_rebuild: If True, delete existing collection before creating new one
        Returns
            Chroma (Chroma): The vector store with collections based on the boardgame name.
        """
        if self._collection_exists(game_name):
            if force_rebuild:
                print(f"Collection for {game_name} exists. Deleting and rebuilding...")
                self._delete_collection(game_name)
            else:
                print(f"Collection for {game_name} already exists. Loading existing collection...")
                print("Use 'force_rebuild=True' to force rebuilding.")
                return self.load_vectorstore(game_name)

        print(f"Creating vector store for {game_name} with {len(documents)} documents...")

        game_path = self._get_game_persist_path(game_name)

        vectorstore = Chroma.from_documents(
            persist_directory=self.persist_directory,
            collection_name=game_name,
            documents=documents,
            embedding=self.embeddings
        )

        self.vectorstore[game_name] = vectorstore

        print(f"Vector store for {game_name} created and persisted to {game_path}")
        return vectorstore

    def load_vectorstore(self, game_name:str) -> Chroma:
        """
        Load the vector store

        Returns:
            Chroma vector store instance
        """
        game_path = self._get_game_persist_path(game_name)

        if not self._collection_exists(game_name):
            raise ValueError(f"No existing collection for {game_name} in {game_path}. Create one first using create_vectorstore()")

        # Check if already loaded in memory
        if game_name in self.vectorstore:
            print(f"Vector store for '{game_name}' already loaded in memory")
            return self.vectorstore[game_name]

        print(f"Loading vector store for {game_name}...")

        vectorstore = Chroma(
            collection_name=game_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        self.vectorstore[game_name] = vectorstore
        collection_count = vectorstore._collection.count()
        print(f"Vector store loaded with {collection_count} documents")
        return vectorstore

    def load_all_vectorstores(self):
        """
        Load all existing game collections into memory.
        """
        collections = self.list_collections()
        print(f"Loading all vector stores for {len(collections)} collections: {collections}")

        for game_name in collections:
            if game_name not in self.vectorstore:
                self.load_vectorstore(game_name)

    def add_documents(self, game_name:str, documents: List[Document]) -> None:
        """
            Add documents to existing collection

        Args:
            game_name (str): The name of the game to add documents to
            documents (List[Document]): The documents to add to the vector store
        """
        if game_name not in self.vectorstore:
            if self._collection_exists(game_name):
                self.load_vectorstore(game_name)
            else:
                raise ValueError(f"Vector store for '{game_name}' not initialized. Call create_vectorstore() first.")

        print(f"Adding {len(documents)} documents to {game_name} collection...")

        self.vectorstore[game_name].add_documents(documents)
        print(f"Documents added successfully to {game_name} collection")

    def similarity_search(self, game_name: str, query: str, k: int = 3) -> List[Tuple]:
        """
        Perform similarity search

        Args:
            game_name (str): The name of the game to search
            query (str): The query to search for
            k (int): The number of documents to return (default 3)

        Returns:
            A list of tuples containing the similarity score and document name
        """
        if game_name not in self.vectorstore:
            if self._collection_exists(game_name):
                self.load_vectorstore(game_name)

        results = self.vectorstore[game_name].similarity_search_with_score(query, k)

        return results

    def search_all_games(self, query: str, k: int = 3) ->  Dict[str, List[Tuple]]:
        """
        Search across all game collections.

        Args:
            query: Query string
            k: Number of results per game

        Returns:
            Dictionary mapping game names to their search results
        """
        self.load_all_vectorstores()

        all_results = {}
        for game_name in self.vectorstore.keys():
            result = self.similarity_search(game_name, query, k)
            if result:
                all_results[game_name] = result

        return all_results
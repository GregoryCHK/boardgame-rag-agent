import os

from dotenv import load_dotenv
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.vector_store import VectorStoreManager

load_dotenv()

system_prompt = """
You are a helpful board game rules assistant. Answer questions about board game rules based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Be clear and concise
4. Cite specific rules when relevant
5. If asked about a different game than the context, clarify that

Context from {game_name}: {context}
"""

user_prompt = "{question}"

class RAGAgent:
    def __init__(self, vector_manager: VectorStoreManager, llm_model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the RAG agent.

        Args:
            vector_manager: VectorStoreManager instance
            llm_model: OpenAI model to use
            temperature: Temperature for response generation
        """
        self.llm_model = llm_model
        self.vector_manager = vector_manager
        self.temperature = temperature
        self.llm_model = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(messages=[
            ("system", system_prompt),
            ("user", user_prompt),
        ])

    def _format_documents(self, documents: List[Document]) -> str:
        return "\n\n--\n\n".join([doc.page_content for doc in documents])

    def query(self, game_name:str, question: str, k: int = 4, return_resources: bool = True) -> Dict:
        """
        Query the RAG agent for a specific game

        Args:
            game_name: Name of the game
            question: Question to ask
            k: Number of documents to return
            return_resources: Whether to return sources for the documents
        """
        # Retrieve relevant documents using the implemented vector manager
        results = self.vector_manager.similarity_search(game_name=game_name, query=question, k=k)

        if not results:
            return {
                "answer" : f"No relevant documents found for {game_name}",
                "sources": [] if return_resources else None
            }

        # Separate documents from scores from the formatted result from vector manager
        documents = [doc for doc, score in results]
        scores = [score for doc, score in results]

        # Format context for the agent prompt
        context = self._format_documents(documents)

        # Prompt template
        messages = self.prompt_template.format_messages(
            game_name=game_name,
            context=context,
            question=question
        )

        response = self.llm_model.invoke(messages)

        result = {
            "answer": response.content,
            "game_name": game_name,
        }

        if return_resources:
            result["sources"] = [
                {
                "content" : doc.page_content,
                "metadata" : doc.metadata,
                "score" : score
                }
                for doc, score in zip(documents, scores)
            ]

        return result

    def query_all_games(self, question: str, k: int = 4) -> List[Dict]:
        """
        Query across all available games and return the most relevant answer.

        Args:
            question: User's question
            k: Number of documents per game to retrieve

        Returns:
            List of results from each game
        """

        all_results = self.vector_manager.search_all_games(query=question)

        responses = []

        for game_name, result in all_results.items():
            documents = [doc for doc, score in result]
            context = self._format_documents(documents)

            messages = self.prompt_template.format_messages(
                game_name=game_name,
                context=context,
                question=question
            )

            response = self.llm_model.invoke(messages)

            responses.append(
                {
                    "game_name": game_name,
                    "answer": response.content,
                    "relevance_score": result[0][1] if result else float('inf')
                }
            )

        # Sort by relevance (lower score = more relevant)
        responses.sort(key=lambda x: x["relevance_score"])

        return responses

    def chat(self, game_name:str, k: int = 4):
        """
        Chat with the RAG agent.

        Args:
            game_name: Name of the game
            k: Number of documents per game to retrieve
        """
        if not self.vector_manager._collection_exists(game_name=game_name):
            print(f"Game {game_name} not found.")
            print(f"Available games: {self.vector_manager.list_collections()}")
            return

        print(f"Chat with {game_name.upper()} Rules Assistant")
        print(f"{'=' * 60}")
        print("Type 'quit' to exit, 'sources' to toggle source display\n")

        show_sources = False

        while True:
            question = input(f"\n[{game_name.upper()}]  You> ").strip()

            if question.lower() == "quit":
                print("\nGoodbye!")
                break

            if question.lower() == "sources":
                show_sources = not show_sources
                print(f"Source Display: {'ON' if show_sources else 'OFF'}")
                continue

            if not question:
                continue

            result = self.query(game_name=game_name, question=question, k=k, return_resources=show_sources)

            print(f"\nAgent: {result['answer']}")

            if show_sources and result.get('sources'):
                print("\n--- Sources ---")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{i}. (Score: {source['score']:.4f})")
                    print(f"{source['content'][:200]}...")
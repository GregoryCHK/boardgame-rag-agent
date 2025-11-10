# test_rag_agent.py
from src.vector_store import VectorStoreManager
from src.rag_agent import RAGAgent
import os

# current file location
current_dir = os.path.dirname(os.path.abspath(__file__))
# go to parent of src
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
persist_directory = os.path.join(root_dir, "vector_db")
os.makedirs(persist_directory, exist_ok=True)

def test_single_game():
    """Test RAG agent with a single game."""
    vector_manager = VectorStoreManager(persist_directory=persist_directory)
    agent = RAGAgent(vector_manager)

    # List available games
    games = vector_manager.list_collections()
    print(f"Available games: {games}\n")

    if not games:
        print("No games found. Run process_games.py first.")
        return

    # Choose a game
    game_name = input("Enter game name: ").strip().lower()

    if game_name not in games:
        print(f"Game '{game_name}' not found.")
        return

    # Test queries
    test_questions = [
        "How do you win the game?",
        "How many players can play?",
        "What happens on your turn?",
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        print('=' * 60)

        result = agent.query(game_name, question, k=3, return_resources=True)

        print(f"\nAnswer: {result['answer']}")

        print("\n--- Top Sources ---")
        for i, source in enumerate(result['sources'][:2], 1):
            print(f"\n{i}. (Score: {source['score']:.4f})")
            print(f"{source['content'][:150]}...")


def test_all_games():
    vector_manager = VectorStoreManager(persist_directory=persist_directory)
    agent = RAGAgent(vector_manager)

    question = input("Enter a question no matter the game: ").strip().lower()

    print(f"\nSearching all games for: '{question}'")
    print("=" * 60)

    results = agent.query_all_games(question, k=3)

    for result in results[:3]:  # Show top 3
        print(f"\n{result['game_name'].upper()}")
        print(f"Relevance Score: {result['relevance_score']:.4f}")
        print(f"Answer: {result['answer']}")
        print("-" * 60)


def test_chat():
    vector_manager = VectorStoreManager(persist_directory=persist_directory)
    agent = RAGAgent(vector_manager)

    games = vector_manager.list_collections()
    print(f"Available games: {games}\n")

    game_name = input("Choose a game for chat: ").strip().lower()

    agent.chat(game_name, k=4)


if __name__ == "__main__":
    print("RAG Agent Tests")
    print("=" * 60)
    print("1. Test single game queries")
    print("2. Query all games")
    print("3. Interactive chat")

    choice = input("\nChoose option: ").strip()

    if choice == "1":
        test_single_game()
    elif choice == "2":
        test_all_games()
    elif choice == "3":
        test_chat()
    else:
        print("Invalid choice")
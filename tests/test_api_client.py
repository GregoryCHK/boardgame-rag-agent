import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("=== Testing health endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_list_games():
    """Test list games endpoint"""
    print("=== Testing listing all games endpoint ===")
    response = requests.get(f"{BASE_URL}/games")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    return response.json()["games"]

def test_query_game(game_name, question):
    """Test querying a specific game endpoint"""
    print(f"=== Testing querying game {game_name} ===")
    print(f"Question: {question}")

    payload = {
        "question": question,
        "k": 3,
        "return_resources": True,
    }
    response = requests.post(f"{BASE_URL}/games/{game_name}/query", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nAnswer: {result['answer']}")

        if result.get("sources"):
            print("\nSources:")
            for i, source in enumerate(result["sources"][:2], 1):
                print(f"{i}. Score: {source['score']}")
                print(f"{source['content'][:150]}...")
    else:
        print(json.dumps(response.json(), indent=2))
    print()

def test_query_all(question):
    """Test querying across all games endpoint"""
    print(f"=== Testing querying across all games ===")
    print(f"Question: {question}")

    payload = {
        "question": question,
        "k": 3,
    }

    response = requests.post(f"{BASE_URL}/query-all", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Found answers in {len(results)} games")
        for result in results:
            print(f"\n{result['game_name'].upper()}: {result['relevance_score']}")
            print(f"Answer: {result['answer']}")
    else:
        print(json.dumps(response.json(), indent=2))
    print()


if __name__ == "__main__":
    # Test health
    test_health()

    # List all games
    games = test_list_games()
    if not games:
        print("No games found")
        exit(1)

    # Test querying specific game
    test_query_game("monopoly", "How do i get out of jail?")

    # Test querying across all games
    test_query_all("How many players can play?")
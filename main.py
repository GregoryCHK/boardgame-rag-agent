import argparse
import sys
import os
from pathlib import Path

import uvicorn

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Constants
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "rules"
CHROMA_DB_DIR = PROJECT_ROOT / "vector_db"
ENV_FILE = PROJECT_ROOT / ".env"

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN} {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING} {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL} {text}{Colors.ENDC}")

def check_environment() -> bool:
    """
    Check if the environment is properly set up.

    Returns:
        bool: True if environment is ready, False otherwise
    """
    print_header("ENVIRONMENT CHECK")
    all_good = True

    # Check Python version
    if sys.version_info >= (3, 10):
        print_success(f"Python version: {sys.version.split()[0]}")
    else:
        print_error(f"Python version {sys.version.split()[0]} is too old. Need 3.10+")
        all_good = False

    # Check for .env file
    if ENV_FILE.exists():
        print_success(".env file found")

        # Check for OpenAI API key
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)

        if os.getenv("OPENAI_API_KEY"):
            print_success("OPENAI_API_KEY is set")

        else:
            print_error("OPENAI_API_KEY not found in .env file")
            all_good = False
    else:
        print_error(".env file not found")
        print_warning("Creating .env file with: OPENAI_API_KEY=your_api_key")

    # Check for rules files
    if DATA_DIR.exists():
        txt_files = list(DATA_DIR.glob("*.txt"))
        if txt_files:
            print_success(f"Data directory found with {len(txt_files)} game files")
            for file in txt_files:
                print(f"  • {file.name}")
        else:
            print_warning(f"Data directory exists but no .txt files found")
            print_warning(f"Add game rulebooks to: {DATA_DIR}")
            all_good = False
    else:
        print_error(f"Data directory not found: {DATA_DIR}")
        print_warning("Creating data directory...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {DATA_DIR}")
        print_warning("Please add .txt game rulebooks to this directory")
        all_good = False

    # Check required packages
    try:
        import langchain, langchain_chroma, langchain_community, langchain_core
        import chromadb
        import fastapi
        import openai
        print_success("All required packages are installed")
    except ImportError as e:
        print_error(f"Missing package: {e.name}")
        print_warning("Run: uv sync")
        all_good = False

    return all_good

def process_games(force_recreate: bool = False, interactive: bool = True):
    """
    Process game rulebooks and create/update vector store.

    Args:
        force_recreate: If True, recreate all collections
        interactive: If True, ask user for confirmation

    """
    print_header("PROCESSING GAME RULES")

    try:
        processor = DocumentProcessor()
        vector_manager = VectorStoreManager(str(CHROMA_DB_DIR))
        existing = vector_manager.list_collections()
        if existing:
            print_warning(f"Found {len(existing)} existing collections: {', '.join(existing)}")

        if interactive and not force_recreate:
            print("\nOptions")
            print("  1. Skip processing (use existing collections)")
            print("  2. Process only new games")
            print("  3. Recreate all games")

        choice = input("\nEnter choice (1/2/3) [1]: ").strip() or "1"

        if choice == "1":
            print_warning("Skipping processing, using existing collections")
            return True
        elif choice == "2":
            force_recreate = False
        elif choice == "3":
            force_recreate = True
        else:
            print_warning("Invalid choice, skipping processing")
            return True

        # Process games
        print_warning("Processing game rulebooks...")

        game_files = list(Path(DATA_DIR).glob("*.txt"))

        if not game_files:
            print_error(f"No .txt files found in {DATA_DIR}")
            return False
        print_warning(f"Found {len(game_files)} game files to process\n")

        for game_file in game_files:
            if not force_recreate and vector_manager._collection_exists(game_file.stem):
                print_warning(f"Skipping '{game_file.stem}' (already exists)")
                continue
            print(f"\n{'─' * 60}")
            print(f"Processing: {Colors.BOLD}{game_file.stem}{Colors.ENDC}")
            print(f"{'─' * 60}")

            try:
                documents = processor.process_documents(str(game_file))
                vector_manager.create_vectorstore(game_file.stem, documents, force_recreate)
                print_success(f"  Successfully processed '{game_file.stem}'")
            except Exception as e:
                print_error(f"  Error processing '{game_file.stem}': {str(e)}")
                continue

    except Exception as e:
        print_error(f"Error processing games: {str(e)}")
        return False


def check_vector_store() -> bool:
    """
    Check if vector store exists and has data.

    Returns:
        bool: True if vector store is ready
    """
    print_header("VECTOR STORE CHECK")

    if not CHROMA_DB_DIR.exists():
        print_warning("Vector store not found")
        return False

    try:
        vector_manager = VectorStoreManager(persist_directory=str(CHROMA_DB_DIR))
        collections = vector_manager.list_collections()

        if not collections:
            print_warning("Vector store exists but has no collections")
            return False

        print_success(f"Vector store ready with {len(collections)} collections")
        return True

    except Exception as e:
        print_error(f"Error checking vector store: {str(e)}")
        return False

def start_api_server(host: str = "localhost", port: int = 8000, reload: bool = True):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
    """
    print_header("STARTING API SERVER")

    print_warning(f"Starting server on http://{host}:{port}")
    print_warning(f"API Documentation: http://localhost:{port}/docs")
    print_warning(f"Alternative Docs: http://localhost:{port}/redoc")
    print_warning("Press CTRL+C to stop the server\n")

    try:
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            reload=reload,
            reload_dirs=[str(src_path)] if reload else None,
            log_level="info"
        )
    except KeyboardInterrupt:
        print_warning("\nServer stopped by user")
    except Exception as e:
        print_error(f"Error starting server: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the application"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Board Games RAG Agent - Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          python main.py                    # Run with interactive prompts
          python main.py --skip-processing  # Skip processing, just start API
          python main.py --force-recreate   # Recreate all collections
          python main.py --no-reload        # Start API without auto-reload
          python main.py --port 5000        # Use custom port
        """
    )

    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip game processing, just start the API server"
    )

    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of all vector collections"
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run without interactive prompts"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the API server to (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the API server to (default: 8000)"
    )

    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload for the API server"
    )

    args = parser.parse_args()

    # Print welcome banner
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║         🎲  BOARD GAMES RAG AGENT  🎲                     ║")
    print("║                                                            ║")
    print("║    Intelligent Question-Answering for Board Game Rules     ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    # Step 1: Check environment
    if not check_environment():
        print_error("\nEnvironment check failed. Please fix the issues above.")
        sys.exit(1)

    print_success("Environment check passed!")

    # Step 2: Process games (if needed)
    if not args.skip_processing:
        # Check if we need to process
        needs_processing = not check_vector_store()

        if needs_processing or args.force_recreate:
            success = process_games(
                force_recreate=args.force_recreate,
                interactive=not args.no_interactive
            )

            if not success:
                print_error("\nGame processing failed. Cannot start API server.")
                sys.exit(1)
        else:
            print_warning("Vector store already exists, skipping processing")
            print_warning("Use --force-recreate to rebuild collections")
    else:
        print_warning("Skipping game processing (--skip-processing flag)")

        # Still need to check if vector store exists
        if not check_vector_store():
            print_error("No vector store found. Remove --skip-processing flag to create one.")
            sys.exit(1)

    # Step 3: Start API server
    start_api_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )


if __name__ == "__main__":
    main()
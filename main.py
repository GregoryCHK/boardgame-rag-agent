from src.document_processor import DocumentProcessor
import os
from pprint import pprint

from src.rag_agent import RAGAgent
from src.vector_store import VectorStoreManager

# Files directory
data_dir = os.path.join(os.getcwd(), "rules")
file_path = os.path.join(data_dir, "codenames.txt")

persist_directory = os.path.join(os.getcwd(), "vector_db")
os.makedirs(persist_directory, exist_ok=True)

# Initialize the document processor
processor = DocumentProcessor()
#documents = processor.process_documents(file_path)


# Initialize Chroma manager
db_manager = VectorStoreManager(persist_directory=persist_directory)

# Create and load vector store
# db_manager.create_vectorstore(game_name='codenames', documents=documents)

query = "Explain me how to setup monopoly?"

# results = db_manager.similarity_search("codenames", query)

llm = RAGAgent(vector_manager=db_manager)

result = llm.query(game_name="monopoly", question=query)

print(result['answer'])
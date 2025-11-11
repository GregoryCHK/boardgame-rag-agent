from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.rag_agent import RAGAgent
from src.vector_store import VectorStoreManager

import os
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Initialize FastAPI app
app = FastAPI(
    title="Board Game Rules RAG API",
    version="1.0",
    description="API for querying board game rules using RAG",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chroma directory
persist_directory = os.path.join(os.getcwd(), "vector_db")
os.makedirs(persist_directory, exist_ok=True)

# Initialize vector store manager and RAG agent
vector_manager = VectorStoreManager(persist_directory=persist_directory)
rag_agent = RAGAgent(vector_manager=vector_manager)

# Define pydantic models for requests/responses
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question about the game rules")
    k: int = Field(default=4, description="Number of documents to retrieve")
    return_resources: bool = Field(default=False, description="Whether to return source documents")

class Source(BaseModel):
    content: str
    metadata: Dict
    score: float

class QueryResponse(BaseModel):
    answer: str
    game_name: str
    resources: Optional[List[Source]] = None

class MultiGameQueryRequest(BaseModel):
    question: str = Field(..., description="Question to search across all games")
    k: int = Field(default=4, description="Number of documents to retrieve per game")

class GameResult(BaseModel):
    game_name: str
    answer: str
    relevance_score: float

class MultiGameQueryResponse(BaseModel):
    results: List[GameResult]

class GamesListResponse(BaseModel):
    games: List[str]
    count: int

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Board Game Rules RAG API",
        "version": "1.0",
        "endpoint": {
            "GET /games": "List all available games",
            "POST /games/{game_name}/query": "Query a specific game",
            "POST /query-all": "Query across available games",
            "Get /health": "Health Check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint """
    try:
        games = vector_manager.list_collections()
        return {
            "status": "healthy",
            "available_games": len(games),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service Unhealthy: {str(e)}")

@app.get("/games")
async def list_games():
    """List all available games"""
    try:
        games = vector_manager.list_collections()
        return {
            "games": games,
            "count": len(games),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/games/{game_name}/query", response_model=QueryResponse)
async def query_game(game_name: str, request: QueryRequest):
    """Query a specific game"""
    try:
        if not vector_manager._collection_exists(game_name):
            raise HTTPException(status_code=404, detail=f"Game {game_name} not found. Available games: {vector_manager.list_collections()}")

        result = rag_agent.query(
            game_name=game_name,
            k=request.k,
            question=request.question,
            return_resources=request.return_resources,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-all",  response_model=MultiGameQueryResponse)
async def query_all_games(request: MultiGameQueryRequest):
    """Search across all available games"""
    try:
        games = vector_manager.list_collections()

        if not games:
            raise HTTPException(status_code=404, detail=f"No games found in database")

        results = rag_agent.query_all_games(
            question=request.question,
            k=request.k,
        )

        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
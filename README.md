# 🎲 Board Games Rules RAG Agent

An intelligent Retrieval-Augmented Generation (RAG) system that provides
accurate, context-aware answers to board game rules questions. Built
with **LangChain**, **ChromaDB**, and **FastAPI**, this project
demonstrates modern AI engineering practices for building
production-ready semantic search applications.

## 🎯 Project Overview

This project implements a complete RAG pipeline that ingests board game
rulebooks, creates semantic embeddings, stores them in a vector
database, and provides a REST API for querying rules using natural
language. The system combines information retrieval with large language
models to generate accurate answers while citing source material.

### Key Features

-   📚 Multi-game support with isolated vector collections\
-   🔍 Semantic search with relevance scoring\
-   🤖 LLM-powered natural language responses\
-   🚀 Production-ready REST API (FastAPI)\
-   📊 Source citation and transparency\
-   💾 Persistent vector storage with ChromaDB

## 🛠️ Technical Stack

### Core Technologies

-   Python 3.10+
-   LangChain
-   OpenAI GPT-4
-   ChromaDB
-   FastAPI
-   Pydantic
-   UV package manager

### Key Skills Demonstrated

-   ✅ RAG Architecture\
-   ✅ Vector Databases\
-   ✅ API Design\
-   ✅ LLM Integration\
-   ✅ Document Processing\
-   ✅ Async Programming\
-   ✅ Modular Software Architecture

## 📁 Project Structure

    boardgames-agent/
    ├── src/
    │   ├── document_processor.py
    │   ├── vector_store.py
    │   ├── rag_agent.py
    │   ├── api.py
    ├── tests/
    │   ├── test_rag_agent.py
    │   └── test_api_client.py
    ├── rules/
    │   └── monopoly.txt
    |   └── dixit.txt
    |   └── codenames.txt
    ├── vector_db/
    ├── main.py
    ├── pyproject.toml
    ├── uv.lock
    ├── .gitignore
    ├── .env
    └── README.md

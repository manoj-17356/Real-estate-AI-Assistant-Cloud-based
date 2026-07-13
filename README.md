# Real Estate AI Assistant

This project is a cloud-ready, retrieval-augmented generation (RAG) application for answering questions about real estate projects, builders, legal terms, payment plans, and customer support information.

It combines a Streamlit chat interface with a vector database built from a knowledge base of real estate documents stored in this repository.

## Overview

The application allows users to ask natural-language questions such as:

- What is the payment plan for a project?
- What are the terms and conditions?
- What is the possession guideline?
- What support information is available for a builder?

The system retrieves relevant documents from the knowledge base and uses an LLM to generate concise answers.

## Project Structure

- app.py — Streamlit web application for the chat interface
- src/ingestion/build_index.py — script to load documents, split them into chunks, and build the vector index
- data/ — raw data source files used for indexing
- chroma_db_v2/ — persistent Chroma vector database
- requirements.txt — Python dependencies

## Technologies Used

- Streamlit for the user interface
- LangChain for retrieval and prompt chaining
- Chroma for vector storage
- Hugging Face embeddings for semantic search
- Groq LLM for answer generation

## Dataset / Knowledge Base

The project includes a synthetic real-estate knowledge base with documents covering:

- Project brochures
- Builder websites and profile documents
- RERA-related information
- Privacy policy and terms documents
- FAQ content
- Payment plans and registration details
- Possession guidance
- Customer support and loan-related documentation

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer
- A Groq API key
- Internet access to download required packages and models

## Installation

1. Open the project folder
2. Install the required dependencies:

```bash
pip install -r requirements.txt

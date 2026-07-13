import argparse
import os
import shutil
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
DATA_DIR = "data/raw"
CHROMA_PATH = "chroma_db_v2"

def load_documents():
    print("Loading documents...")
    documents = []

    # Map file extensions to their respective loaders
    # Note: UnstructuredMarkdownLoader requires unstructured package which might be heavy.
    # We will use TextLoader for markdown as a lightweight alternative if unstructured is not installed.
    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.html", loader_cls=BSHTMLLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader),
    ]

    for loader in loaders:
        try:
            docs = loader.load()
            print(f"Loaded {len(docs)} documents using {loader.loader_cls.__name__}")
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading with {loader.loader_cls.__name__}: {e}")

    return documents

def chunk_documents(documents):
    print(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def store_embeddings(chunks):
    print("Generating embeddings and storing in ChromaDB...")
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store chunks in Chroma
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing existing database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
    else:
        print("No existing database to clear.")

def main():
    parser = argparse.ArgumentParser(description="Build RAG index for Real Estate data.")
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma database.")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    documents = load_documents()
    if not documents:
        print("No documents found. Exiting.")
        return

    chunks = chunk_documents(documents)
    store_embeddings(chunks)
    print("Index build complete!")

if __name__ == "__main__":
    main()

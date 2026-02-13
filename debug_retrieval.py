import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def debug_retrieval(query):
    embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    docs = vectorstore.similarity_search(query, k=5)
    print(f"\n--- DEBUG RETRIEVAL for: '{query}' ---")
    print(f"Found {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:500])
        print("-" * 20)

if __name__ == "__main__":
    debug_retrieval("What to do when approached by an emergency vehicle?")

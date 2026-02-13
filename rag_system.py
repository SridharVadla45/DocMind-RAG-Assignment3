import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables (for OpenAI API Key)
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment or .env file.")
    # In some environments, it might be named differently
    if os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        print("Using GEMINI_API_KEY as GOOGLE_API_KEY.")
    else:
        sys.exit(1)

def build_rag_system():
    # 1. Document Loading
    print("--- Loading Documents ---")
    data_path = "./data/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    loader = DirectoryLoader(data_path, glob="./*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Error: No PDF files found in data/ directory. Please add a PDF file and run again.")
        return None

    # 2. Text Splitting
    print("--- Splitting Documents ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embedding & Storage
    print("--- Creating Embeddings & Storing in ChromaDB ---")
    # Using Jina AI's open-source model via HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-en",
        model_kwargs={'trust_remote_code': True}
    )
    
    # Clean up existing DB to avoid duplicate/stale metadata
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

def main():
    vectorstore = build_rag_system()
    if not vectorstore:
        return

    # 4. RAG Chain
    # Try gemini-flash-latest
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    from langchain.prompts import PromptTemplate

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer or the information is not present in the context, just say that "I am sorry, but the document does not contain information about this." 
    Do not try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Use RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # 5. User Query & Answer Generation
    print("\nRAG System Ready! Type 'exit' to stop.")
    
    # Predefined queries for testing as per requirements
    test_queries = [
        "What is Crosswalk guards?",
        "What to do if moving through an intersection with a green signal?",
        "What to do when approached by an emergency vehicle?"
    ]
    
    output_file = "./output/results.txt"
    os.makedirs("./output", exist_ok=True)
    
    import time
    from google.api_core import exceptions

    with open(output_file, "w") as f:
        f.write("RAG SYSTEM TEST RESULTS\n")
        f.write("="*30 + "\n\n")
        
        for query in test_queries:
            print(f"\nProcessing Test Query: {query}")
            
            # Simple retry logic for rate limits (429)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = qa_chain.invoke({"query": query})
                    answer = response["result"]
                    source_docs = response.get("source_documents", [])
                    
                    f.write(f"QUERY: {query}\n")
                    f.write(f"ANSWER: {answer}\n")
                    
                    if source_docs:
                        f.write("SOURCES:\n")
                        for doc in source_docs:
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "Unknown")
                            f.write(f"  - File: {os.path.basename(source)}, Page: {page}\n")
                    
                    f.write("-" * 20 + "\n")
                    print(f"Answer: {answer}")
                    if source_docs:
                        print("Sources:")
                        for doc in source_docs[:2]:
                            source = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", "Unknown")
                            print(f"  - {source} (Page {page})")
                    
                    # Add a small delay between successful queries to avoid hitting rate limits
                    time.sleep(2)
                    break
                    
                except exceptions.ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        wait_time = 30 # Wait 30 seconds on quota error
                        print(f"Quota exceeded. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print("Error: Quota exceeded after multiple retries. Please wait a few minutes before running again.")
                        f.write(f"QUERY: {query}\nERROR: Quota exceeded. Please run again later.\n")
                        f.write("-" * 20 + "\n")

    # Interactive loop
    while True:
        user_query = input("\nAsk a question about your document (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        if not user_query.strip():
            continue
            
        print("Retrieving and generating answer...")
        response = qa_chain.invoke({"query": user_query})
        answer = response["result"]
        source_docs = response.get("source_documents", [])
        
        print(f"\nAnswer: {answer}")
        if source_docs:
            print("\nSource Citations:")
            unique_sources = set()
            for doc in source_docs:
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "Unknown")
                unique_sources.add(f"{source} (Page {page})")
            
            for s in sorted(list(unique_sources)):
                print(f"  - {s}")

if __name__ == "__main__":
    main()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_data(pdf_path="tariff.pdf", persist_directory="./chroma_db"):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    print(f"Created {len(splits)} chunks. Generating embeddings and storing in VectorDB...")
    
    # Using a fast, local embedding model that both Gemini and Ollama agents can share
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Successfully ingested {len(splits)} chunks into {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    # Ensure current working directory is the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if os.path.exists("./chroma_db"):
        print("VectorDB already exists at ./chroma_db. Skipping ingestion.")
    else:
        ingest_data()

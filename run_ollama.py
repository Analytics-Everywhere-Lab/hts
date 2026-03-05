import os
from graph import run_pipeline
from langchain_ollama import ChatOllama

def main():
    # Initialize Ollama LLM
    ollama_llm = ChatOllama(model="qwen3:4b")
    
    print("Welcome to the Ollama (Qwen) HTS Code Generator!")
    print("Note: Ensure you have run 'ollama serve' and have the model pulled.")
    print("Enter 'quit' to exit.")
    
    while True:
        desc = input("\nEnter item description: ")
        if desc.lower() in ["quit", "exit", "q"]:
            break
            
        print("Running pipeline (Search + RAG)...")
        try:
            result = run_pipeline(desc, ollama_llm)
            print("\n" + "="*50)
            print("REASONING STEPS & CONTEXT")
            print("="*50)
            print(f"Generated Search Queries: {result.get('search_queries', [])}")
            print("\n--- Websites Read (Search Context) ---")
            print(result.get("search_results", "None"))
            print("\n--- Tariff PDF (RAG Context) ---")
            print(result.get("rag_context", "None"))
            print("="*50)
            print(f"\n=> Predicted HTS Code: {result.get('final_hts_code', 'ERROR')}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

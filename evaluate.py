import os
import pandas as pd
from dotenv import load_dotenv
from graph import run_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def main():
    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    # Initialize Gemini LLM
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        disable_streaming=True
    )
    
    # Initialize Ollama LLM
    ollama_llm = ChatOllama(model="qwen3:4b")
    
    # Load dataset
    df = pd.read_csv("data.csv")
    
    results = []
    gemini_correct = 0
    ollama_correct = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        desc = str(row['desc'])
        label = str(row['label']).strip()
        print(f"\n[{idx+1}/{total}] Processing item: {desc}")
        print(f"Target Label: {label}")
        
        # Run Gemini
        try:
            gemini_result = run_pipeline(desc, gemini_llm)
            gemini_pred = gemini_result["final_hts_code"]
        except Exception as e:
            print(f"Gemini error: {e}")
            gemini_pred = "ERROR"
            
        print(f"Gemini Prediction: {gemini_pred}")
        
        # Run Ollama
        try:
            ollama_result = run_pipeline(desc, ollama_llm)
            ollama_pred = ollama_result["final_hts_code"]
        except Exception as e:
            print(f"Ollama error: {e}")
            ollama_pred = "ERROR"
            
        print(f"Ollama Prediction: {ollama_pred}")
        
        # Evaluation logic
        # Some HTS codes might have spaces or dashes, keep only digits
        clean_label = ''.join(filter(str.isdigit, label))
        
        gemini_match = (clean_label in gemini_pred) or (gemini_pred in clean_label)
        ollama_match = (clean_label in ollama_pred) or (ollama_pred in clean_label)
        
        if gemini_match and len(gemini_pred) >= 4:
            gemini_correct += 1
        if ollama_match and len(ollama_pred) >= 4:
            ollama_correct += 1
            
        results.append({
            "Description": desc,
            "Target_HTS": clean_label,
            "Gemini_Prediction": gemini_pred,
            "Gemini_Correct": gemini_match,
            "Ollama_Prediction": ollama_pred,
            "Ollama_Correct": ollama_match
        })
        
    print("\n--- Final Evaluation ---")
    print(f"Gemini Accuracy: {gemini_correct}/{total} ({gemini_correct/total*100:.2f}%)")
    print(f"Ollama Accuracy: {ollama_correct}/{total} ({ollama_correct/total*100:.2f}%)")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Saved results to evaluation_results.csv")

if __name__ == "__main__":
    main()

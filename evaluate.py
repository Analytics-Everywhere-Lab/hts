import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from graph import run_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def main():
    parser = argparse.ArgumentParser(description="Evaluate HTS Classification Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate (e.g., gemini-3.1-flash-lite-preview, qwen3:4b)")
    args = parser.parse_args()
    
    model_name = args.model

    load_dotenv()
    if "gemini" in model_name.lower():
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        print(f"Initializing Gemini pipeline with model: {model_name}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            disable_streaming=True
        )
    else:
        print(f"Initializing Ollama pipeline with model: {model_name}")
        llm = ChatOllama(model=model_name)
    
    # Load dataset
    df = pd.read_csv("data.csv")
    
    results = []
    element_correct = {
        "Chapter (first 2)": 0,
        "Heading (next 2)": 0,
        "Subheading 1 (next 2)": 0,
        "Subheading 2 (next 2)": 0,
        "Statistical Suffix (last 2)": 0,
        "Whole (10 digits)": 0
    }
    total = len(df)
    
    for idx, row in df.iterrows():
        desc = str(row['desc'])
        label = str(row['label']).strip()
        print(f"\n[{idx+1}/{total}] Processing item: {desc}")
        print(f"Target Label: {label}")
        
        try:
            result = run_pipeline(desc, llm, disable_escalation=True)
            pred = result["final_hts_code"]
        except Exception as e:
            print(f"Error ({model_name}): {e}")
            pred = "ERROR"
            
        print(f"Prediction: {pred}")
        
        # Evaluation logic
        # Some HTS codes might have spaces or dashes, keep only digits
        clean_label = ''.join(filter(str.isdigit, label)).ljust(10, '0')[:10]
        
        # Clean predictions to match the label format exactly (10 digits)
        clean_pred = ''.join(filter(str.isdigit, pred)).ljust(10, '0')[:10]
        
        match_chapter = clean_label[0:2] == clean_pred[0:2]
        match_heading = clean_label[2:4] == clean_pred[2:4]
        match_subheading1 = clean_label[4:6] == clean_pred[4:6]
        match_subheading2 = clean_label[6:8] == clean_pred[6:8]
        match_statistical_suffix = clean_label[8:10] == clean_pred[8:10]
        match_whole = clean_label == clean_pred
        
        if match_chapter: element_correct["Chapter (first 2)"] += 1
        if match_heading: element_correct["Heading (next 2)"] += 1
        if match_subheading1: element_correct["Subheading 1 (next 2)"] += 1
        if match_subheading2: element_correct["Subheading 2 (next 2)"] += 1
        if match_statistical_suffix: element_correct["Statistical Suffix (last 2)"] += 1
        if match_whole: element_correct["Whole (10 digits)"] += 1
            
        results.append({
            "Description": desc,
            "Target_HTS": clean_label,
            "Prediction": clean_pred,
            "Correct_Chapter": match_chapter,
            "Correct_Heading": match_heading,
            "Correct_Subheading_1": match_subheading1,
            "Correct_Subheading_2": match_subheading2,
            "Correct_Statistical_Suffix": match_statistical_suffix,
            "Correct_Whole": match_whole
        })
        
    print("\n--- Final Evaluation ---")
    print(f"Model: {model_name}")
    print(f"Total Items: {total}")
    for el, count in element_correct.items():
        print(f"{el} Accuracy: {count}/{total} ({count/total*100:.2f}%)")
    
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    output_filename = f"{safe_model_name}_evaluation_results.csv"
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main()

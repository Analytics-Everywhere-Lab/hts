import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from graph import run_pipeline, process_escalation_chat
from typing import List, Dict

# Initialize FastAPI
app = FastAPI(title="HTS Code Generator API")

# Load Env
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Mount static directory for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# LLM setup
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    disable_streaming=True
)

class ClassifyRequest(BaseModel):
    description: str

class EscalationRequest(BaseModel):
    description: str
    search_results: str
    rag_context: str
    chat_history: List[Dict[str, str]]

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/api/classify")
def classify_hts(request: ClassifyRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")
        
    try:
        # Run the LangGraph Pipeline
        result = run_pipeline(request.description, gemini_llm)
        return {
            "success": True,
            "final_hts_code": result.get("final_hts_code"),
            "element_confidences": result.get("element_confidences"),
            "escalation_needed": result.get("escalation_needed"),
            "escalation_question": result.get("escalation_question"),
            "reasoning_steps": result.get("reasoning_steps"),
            "search_results": result.get("search_results"),
            "rag_context": result.get("rag_context"),
            "search_queries": result.get("search_queries")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/escalation")
def handle_escalation(request: EscalationRequest):
    try:
        result = process_escalation_chat(
            request.description,
            request.search_results,
            request.rag_context,
            request.chat_history,
            gemini_llm
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
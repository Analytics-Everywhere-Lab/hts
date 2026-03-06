import os
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Any, Dict
from langgraph.graph import StateGraph, START, END
from duckduckgo_search import DDGS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import logging
import warnings

# Suppress harmless warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

class GraphState(TypedDict):
    item_description: str
    search_queries: List[str]
    search_results: str
    rag_context: str
    reasoning_steps: str
    final_hts_code: str
    element_confidences: Dict[str, Any]
    escalation_needed: bool
    escalation_question: str
    llm: Any

def search_node(state: GraphState):
    llm = state["llm"]
    item_desc = state["item_description"]
    
    # 1. Ask LLM to generate a search query
    sys_msg = SystemMessage(content="You are an expert search assistant. Given an item description, generate a concise DuckDuckGo search query to find its Canadian HTS code, material composition, or tariff classification. Output ONLY the query string. Do not include quotes or extra text.")
    user_msg = HumanMessage(content=item_desc)
    
    response = llm.invoke([sys_msg, user_msg])
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    query = content.strip().strip('"').strip("'")
    
    # 2. Search DDG
    ddgs = DDGS()
    try:
        results = list(ddgs.text(query + " Canadian HTS code", max_results=3))
    except Exception as e:
        results = []
        
    # 3. Scrape results
    scraped_content = ""
    for r in results:
        url = r.get("href")
        if not url: continue
        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.content, "html.parser")
            text = " ".join([p.text for p in soup.find_all("p")])
            scraped_content += f"\nSource ({url}):\n{text[:1000]}\n"
        except:
            scraped_content += f"\nSource ({url}): {r.get('body')}\n"
            
    if not scraped_content.strip():
        scraped_content = "No relevant web results found."
        
    return {"search_queries": [query], "search_results": scraped_content}

def rag_node(state: GraphState):
    item_desc = state["item_description"]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    docs = retriever.invoke(item_desc)
    rag_context = "\n\n".join([d.page_content for d in docs])
    
    return {"rag_context": rag_context}
    
def decision_node(state: GraphState):
    # Retrieve base LLM
    base_llm = state["llm"]
    item_desc = state["item_description"]
    search_results = state["search_results"]
    rag_context = state["rag_context"]
    
    sys_msg = SystemMessage(content="""You are an expert Canadian customs classification agent. 
Analyze the item description, web search context, and official Canadian Tariff PDF context to determine the 10-digit Canadian HTS code.
Output your response as STRICT JSON with the following structure:
{
  "reasoning": "Step-by-step logic for the classification...",
  "chapter": "First 2 digits",
  "heading": "Next 2 digits",
  "subheading": "Next 2 digits",
  "additional_subheading": "Next 2 digits",
  "statistical_suffix": "Last 2 digits"
}
Ensure all digits combined make exactly 10 digits. Do not include markdown formatting like ```json. Output ONLY the raw JSON object.
""")
    user_msg = HumanMessage(content=f"Item Description: {item_desc}\n\nWeb Search Context:\n{search_results}\n\nTariff Context:\n{rag_context}")
    
    # 3-Ensemble Voting for Self-Consistency
    # We will simulate perturbations using 3 different variations of the prompt slightly.
    perturbations = [
        "What is the HTS code for this item?",
        "Carefully re-evaluate the material composition and function. What is the 10-digit HTS code?",
        "Please provide the most accurate 10-digit Tariff classification based on the provided evidence."
    ]
    
    results_list = []
    first_reasoning = ""
    
    for i, pert in enumerate(perturbations):
        perturbed_user_msg = HumanMessage(content=f"{user_msg.content}\n\n{pert}")
        # Note: Ideally we'd vary temperature, but LangChain's ChatGoogleGenerativeAI might need re-instantiation for that. 
        # Using altered user prompts is a standard perturbation.
        try:
            response = base_llm.invoke([sys_msg, perturbed_user_msg])
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            
            # clean json
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                content = content[start:end]
            
            data = json.loads(content)
            if i == 0:
                first_reasoning = data.get("reasoning", "")
            results_list.append(data)
        except Exception as e:
            print(f"Ensemble run {i+1} failed to parse JSON: {e}")
            pass
            
    # Voting logic
    elements = ["chapter", "heading", "subheading", "additional_subheading", "statistical_suffix"]
    votes = {el: {} for el in elements}
    
    for res in results_list:
        if not isinstance(res, dict): continue
        for el in elements:
            val = res.get(el, "")
            # Ensure it's exactly 2 digits
            val = ''.join(filter(str.isdigit, str(val)))[:2].zfill(2)
            votes[el][val] = votes[el].get(val, 0) + 1
            
    final_elements = {}
    element_confidences = {}
    escalation_needed = False
    
    for el in elements:
        if not votes[el]:
            # Fallback if all failed
            final_elements[el] = "00"
            element_confidences[el] = {"value": "00", "confidence": "0% (0/3)", "score": 0.0}
            escalation_needed = True
            continue
            
        # Get max voted value
        top_val = max(votes[el], key=votes[el].get)
        vote_count = votes[el][top_val]
        total_runs = len(results_list) if len(results_list) > 0 else 3
        confidence_fraction = vote_count / total_runs
        
        final_elements[el] = top_val
        element_confidences[el] = {
            "value": top_val, 
            "confidence": f"{int(confidence_fraction*100)}% ({vote_count}/{total_runs})",
            "score": confidence_fraction
        }
        
        # If not unanimous or at least 2/3, we escalate
        if confidence_fraction < 0.6:  # Less than 2 out of 3
            escalation_needed = True
            
    final_hts_code = f"{final_elements['chapter']}{final_elements['heading']}.{final_elements['subheading']}.{final_elements['additional_subheading']}.{final_elements['statistical_suffix']}"
    
    escalation_question = ""
    if escalation_needed:
        # Generate the first escalation question
        esc_sys_msg = SystemMessage(content="You are an expert Canadian customs classification assistant. Our AI ensemble could not confidently classify an item. Given the item context and the elements we are struggling with, ask the user ONE targeted, clarifying question to help determine the correct 10-digit HTS code.")
        esc_user_msg = HumanMessage(content=f"Item: {item_desc}\n\nSearch Context: {search_results}\n\nTariff Context: {rag_context}\n\nCurrent Best Confidences:\n{json.dumps(element_confidences, indent=2)}\n\nWhat ONE question should I ask the user to clarify the classification?")
        try:
            esc_response = base_llm.invoke([esc_sys_msg, esc_user_msg])
            content = esc_response.content
            escalation_question = content[0].get("text", "") if isinstance(content, list) else str(content).strip()
        except:
            escalation_question = "Could you please provide more details about the material composition or specific function of this item?"

    return {
        "final_hts_code": final_hts_code,
        "reasoning_steps": first_reasoning,
        "element_confidences": element_confidences,
        "escalation_needed": escalation_needed,
        "escalation_question": escalation_question
    }
    
def process_escalation_chat(item_desc: str, search_results: str, rag_context: str, chat_history: List[Dict[str, str]], llm: Any):
    # Chat history is a list of dicts: [{"role": "user"|"assistant", "content": "..."}]
    
    sys_msg = SystemMessage(content="""You are an expert Canadian customs classification agent interacting with a human. 
Your goal is to determine the 10-digit Canadian HTS code.
If you are still unsure, ask ONE clarifying question.
If you are confident you know the 10-digit code, you MUST output a STRICT JSON payload with exactly this structure:
{
  "reasoning": "Step-by-step logic",
  "chapter": "First 2 digits",
  "heading": "Next 2 digits",
  "subheading": "Next 2 digits",
  "additional_subheading": "Next 2 digits",
  "statistical_suffix": "Last 2 digits",
  "final_hts_code": "XXXX.XX.XX.XX",
  "element_confidences": {
    "chapter": { "value": "XX", "confidence": "100% (Human Verified)", "score": 1.0 },
    "heading": { "value": "XX", "confidence": "100% (Human Verified)", "score": 1.0 },
    "subheading": { "value": "XX", "confidence": "100% (Human Verified)", "score": 1.0 },
    "additional_subheading": { "value": "XX", "confidence": "100% (Human Verified)", "score": 1.0 },
    "statistical_suffix": { "value": "XX", "confidence": "100% (Human Verified)", "score": 1.0 }
  },
  "is_final": true
}
Do NOT output JSON until you are absolutely confident. Otherwise, just output conversational text asking the clarifying question.
""")
    
    messages = [sys_msg]
    messages.append(HumanMessage(content=f"Initial Item Description: {item_desc}\n\nWeb Search Context:\n{search_results}\n\nTariff Context:\n{rag_context}"))
    
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=f"Assistant previously said: {msg['content']}")) # Use SystemMessage or AIMessage. AIMessage is better if available. Doing SystemMessage for simplicity/safety

    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        
    content = content.strip()
    
    # Try parsing as JSON
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            data = json.loads(json_str)
            if data.get("is_final") or data.get("final_hts_code"):
                return {"is_final": True, "data": data}
    except:
        pass
        
    # If not JSON or failed to parse, it's a question
    return {"is_final": False, "message": content}

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("search", search_node)
    builder.add_node("rag", rag_node)
    builder.add_node("decision", decision_node)
    
    builder.add_edge(START, "search")
    builder.add_edge("search", "rag")
    builder.add_edge("rag", "decision")
    builder.add_edge("decision", END)
    
    return builder.compile()

# Instantiate the executable graph
graph = build_graph()

def run_pipeline(item_description: str, llm: Any):
    initial_state = {"item_description": item_description, "llm": llm, "search_queries": [], "search_results": "", "rag_context": "", "reasoning_steps": "", "final_hts_code": "", "element_confidences": {}, "escalation_needed": False, "escalation_question": ""}
    result = graph.invoke(initial_state)
    return result



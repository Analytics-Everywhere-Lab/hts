import os
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END
from duckduckgo_search import DDGS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

class GraphState(TypedDict):
    item_description: str
    search_queries: List[str]
    search_results: str
    rag_context: str
    final_hts_code: str
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
    llm = state["llm"]
    item_desc = state["item_description"]
    search_results = state["search_results"]
    rag_context = state["rag_context"]
    
    sys_msg = SystemMessage(content="You are an expert Canadian customs classification agent. Given an item description, web search context, and official Canadian Tariff PDF context, output ONLY the 10-digit Canadian HTS code (e.g., 9506620090). Do not include any other text, explanation, or punctuation.")
    user_msg = HumanMessage(content=f"Item Description: {item_desc}\n\nWeb Search Context:\n{search_results}\n\nTariff Context:\n{rag_context}\n\nWhat is the 10-digit HTS code?")
    
    response = llm.invoke([sys_msg, user_msg])
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    final_code = content.strip()
    # clean up code: just keep digits
    final_code = ''.join(filter(str.isdigit, final_code))
    
    return {"final_hts_code": final_code}
    
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
    initial_state = {"item_description": item_description, "llm": llm}
    result = graph.invoke(initial_state)
    return result


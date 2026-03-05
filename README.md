# Multi-Agent Canadian HTS Classification System

This repository contains a multimodal, multi-agent AI pipeline designed to intelligently predict Canadian Harmonized Tariff Schedule (HTS) codes from plain-text item descriptions. The system leverages LangGraph, LangChain, and Chroma vector databases.

There are two LLM pipelines supported out-of-the-box:
1. **Gemini**: Using Google's `gemini-3.1-flash-lite-preview` via API.
2. **Ollama**: Using the open-source `qwen3:4b` running locally.

## Architecture

The workflow is orchestrated via **LangGraph** through three core agent nodes:

1. **Search Agent (DuckDuckGo + BeautifulSoup):**
   - Receives the raw item description.
   - Leverages an LLM to formulate an optimized search query.
   - Searches DuckDuckGo for the query.
   - Scrapes the text from the top web results, returning detailed context (material composition, alternative names, commonly cited HTS codes online).
2. **RAG Agent (ChromaDB + PyPDF):**
   - Computes local embeddings using `all-MiniLM-L6-v2`.
   - Queries a pre-built Chroma vector store populated with extracts from the official Canadian `tariff.pdf` schedule.
   - Retrieves the top matches based on cosine similarity to the item description.
3. **Decision Agent (LLM classification):**
   - Takes the user's item description, the aggregated context from live websites, and the official excerpts from the tariff schedule.
   - Synthesizes all information to deduce the authoritative 10-digit Canadian HTS code.

---

## Setup & Installation

### 1. Requirements

Ensure you have Python 3.11+ and a `.venv` (virtual environment) configured.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root of the project with your Gemini API Key:

```env
GEMINI_API_KEY=AIzaSyA...
```

*(Note: The Ollama pipeline does not require an API key but requires the `ollama` daemon to be active locally).*

### 3. Data Ingestion (VectorDB)

Before running the agents, you must parse your local `tariff.pdf` and build the retrieval database.

```bash
python data_ingestion.py
```
*This will create a `chroma_db/` directory containing the vectorized chunks of the document.*

---

## Running the Application

### Option A: Interactive Wrappers

Test individual item descriptions iteratively and view the full inner reasoning (search queries, scraped URLs, RAG contexts, and final classification) in your terminal.

**For Gemini (Cloud):**
```bash
python run_gemini.py
```

**For Ollama (Local):**
```bash
# In an external terminal, ensure you have pulled the model and started the server:
# ollama pull qwen3:4b
# ollama serve
python run_ollama.py
```

### Option B: Batch Evaluation

To iterate over the provided `data.csv` dataset, compare outputs from both pipelines against the `label` targets, and score the accuracy:

```bash
python evaluate.py
```
*This script will write results out to a file named `evaluation_results.csv` logging correctness and generated outputs row-by-row.*
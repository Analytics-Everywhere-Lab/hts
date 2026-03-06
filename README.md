# Multi-Agent Canadian HTS Classification System

This repository contains a sophisticated, multi-agent AI pipeline designed to intelligently predict Canadian Harmonized Tariff Schedule (HTS) codes from plain-text item descriptions. The system leverages **LangGraph**, **LangChain**, and **Chroma** vector databases to achieve high-accuracy classifications through a combination of live web research, official document retrieval, and self-consistency voting.

## 🚀 Key Features

*   **Multi-Agent Orchestration**: Powered by LangGraph for structured, stateful workflows.
*   **Search Agent**: Real-time web browsing using DuckDuckGo and BeautifulSoup to gather material composition and industry context.
*   **RAG Agent**: Contextual retrieval from the official 2024 Canadian Tariff Schedule (PDF) via ChromaDB.
*   **Self-Consistency Voting**: Uses a 3-instance ensemble with prompt perturbations to calculate confidence scores for each digit of the HTS code.
*   **Human-in-the-Loop Escalation**: Automatically triggers a clarification chat interface if the AI's confidence in any HTS element falls below 60%.
*   **Modern Web UI**: A glassmorphic, responsive web interface built with FastAPI and Vanilla CSS/JS for real-time interaction and reasoning visualization.
*   **Model Agnostic**: Supports Google Gemini (Cloud) and Ollama (Local) pipelines.

---

## 🏗️ Architecture

The classification workflow follows a directed acyclic graph (DAG):

1.  **Search Node**: Formulates optimized queries, scrapes results, and extracts relevant technical specifications.
2.  **RAG Node**: Vectorizes the item description to pull specific legal headings and subheadings from the local 1,500+ page tariff schedule.
3.  **Decision Node**:
    *   Synthesizes Web + PDF context.
    *   Runs 3 independent classification attempts.
    *   Performs element-wise voting to determine the final 10-digit code (`XXXX.XX.XX.XX`).
    *   Calculates a confidence score for each segment (Chapter, Heading, Subheading, etc.).
4.  **Escalation Logic**: If consensus is not reached, it prepares a targeted clarifying question for the user.

---

## 🛠️ Setup & Installation

### 1. Environment Configuration

Ensure you have Python 3.11+ installed. We recommend using a virtual environment.

```bash
# Install dependencies
pip install -r requirements.txt
# Note: Ensure fastapi, uvicorn, and duckduckgo-search (ddgs) are also installed.
```

### 2. API Credentials

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_ai_studio_api_key
```

### 3. Data Ingestion (First-Time Only)

Populate the Chroma vector store with extracts from the official `tariff.pdf`:

```bash
python data_ingestion.py
```
*This creates the `chroma_db/` directory.*

---

## 🖥️ Usage

### Option A: The Web Application (Recommended)

Start the interactive web interface to experience the human-escalation chatbot and visual reasoning chain.

```bash
python app.py
```
Access the UI at: `http://localhost:8001`

### Option B: Interactive CLI Wrappers

Test individual descriptions and view full internal logs in the terminal.

*   **Gemini (Cloud)**: `python run_gemini.py`
*   **Ollama (Local)**: `python run_ollama.py`

### Option C: Batch Evaluation

Benchmark the system against `data.csv` to calculate accuracy and generate performance reports.

```bash
python evaluate.py
```
*Results are saved to `evaluation_results.csv`.*

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | FastAPI application serving the Web UI and API endpoints. |
| `graph.py` | Core LangGraph logic, agent node definitions, and escalation chat. |
| `data_ingestion.py` | PDF parsing and vector database (RAG) initialization. |
| `evaluate.py` | Metric generation and batch testing script. |
| `run_*.py` | CLI entry points for specific LLM backends. |
| `static/` | Frontend assets (HTML/CSS/JS) for the web interface. |
| `tariff.pdf` | The official reference document for Canadian HTS codes. |
| `requirements.txt` | Python dependencies. |

---

## ⚖️ License
This project is intended for research and internal logistical optimization. Always verify final HTS codes with official customs documentation for legal compliance.
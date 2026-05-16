# Capstone Writeup: Enabling AI-Powered Business Intelligence for Organizations

## Project Overview

This project implements an AI-powered business intelligence assistant named InsightForge. The system helps users analyze structured sales data, retrieve insights from business documents, and ask natural-language questions through a Gradio interface.

The final application uses Gradio instead of Streamlit. The main UI includes database status, document management, evaluation, and chat functionality in one place.

## Problem Statement

Organizations often store useful business knowledge across multiple formats, including CSV files, PDFs, and generated analysis reports. Decision makers need a way to combine these sources and ask business questions without manually searching through files or writing custom analysis code.

This project addresses that problem by combining:

- CSV business intelligence analysis.
- PDF and pickle document ingestion.
- Vector search over stored document chunks.
- LLM-based retrieval-augmented generation.
- Evaluation reporting for answer quality.

## Goals

The main goals of the project are:

- Analyze business sales data and generate useful insights.
- Produce visual output for sales analysis.
- Load PDF and analysis documents into a searchable vector database.
- Provide a chat interface for business questions.
- Support document reset and upload from the UI.
- Provide an evaluation workflow using QAEvalChain or an LLM evaluator.
- Save output artifacts for grading and review.

## Technology Stack

- Python
- Gradio
- Pandas
- NumPy
- Matplotlib
- Seaborn
- PostgreSQL
- pgvector
- LangChain
- LangChain Classic
- LangGraph
- Groq
- Sentence Transformers
- pypdf
- Docker and Docker Compose

## System Architecture

The project is organized into four main layers:

1. CSV analysis layer  
   Reads sales data, computes statistics, extracts business insights, and generates a dashboard image and pickle file.

2. Document processing layer  
   Loads PDF and pickle files, splits content into chunks, generates embeddings, and stores the results in PostgreSQL with pgvector.

3. RAG/chat layer  
   Uses vector search to retrieve relevant document chunks and passes those results into the LangGraph-based conversational flow.

4. Gradio UI layer  
   Provides database status, document management, evaluation, and chat controls for the user.

## Main Features

### CSV Business Intelligence Analysis

The CSV analysis workflow processes sales data and generates business insights such as:

- Sales trends.
- Product performance.
- Regional performance.
- Customer-level analysis.
- Statistical summaries.
- Dashboard visualization.

Generated artifacts include:

```text
output/csv_analysis_dashboard.png
output/csv_analysis_results.pkl
```

### Document Management

The Gradio interface includes document management tools that allow the user to:

- Upload PDF files.
- Upload `.pkl` or `.pickle` analysis files.
- Reset the database.
- Reload the default PDF and pickle files.
- Refresh database/document status.

This supports repeatable demonstrations and makes it easier for graders to test the document ingestion workflow.

### Chat Interface

The chat interface allows users to ask questions about indexed documents and analysis results. The system retrieves relevant chunks from PostgreSQL/pgvector and uses the LLM to produce an answer grounded in the retrieved context.

The application appends one source and score block to each answer:

```text
Source: <document name>
Score: <retrieval score>
```

The prompt also prevents the LLM from generating duplicate source/score text.

### Evaluation

The project includes an evaluation workflow that can be run from either the UI or the command line.

The preferred evaluator is:

```text
QAEvalChain
```

Because modern LangChain versions moved older evaluation APIs, the project uses:

```text
langchain-classic==1.0.3
```

The import path is:

```python
from langchain_classic.evaluation.qa import QAEvalChain
```

If QAEvalChain cannot run, the evaluation system falls back to an LLM judge or a keyword-based heuristic.

The default evaluation report is written to:

```text
output/evaluation_report.json
```

## User Interface

The main interface is launched with:

```bash
make web
```

The app runs at:

```text
http://localhost:7860
```

The UI includes:

- Database Status
- Document Management
- Evaluation
- Chat

## Screenshots

Screenshots are stored in:

```text
docs/screenshots/
```

Included screenshots:

```text
docs/screenshots/home_page_chat.png
docs/screenshots/home_page_document_upload.png
docs/screenshots/home_page_eval.png
docs/screenshots/csv_analysis_dashboard.png
```

## How To Run

The project is intended to run in WSL with Ubuntu or Ubuntu-24.04.

Start the database:

```bash
make start-db
```

Run CSV analysis:

```bash
make csv
```

Process default documents:

```bash
PRE_LOAD_DATA=true make process
```

Run evaluation:

```bash
make eval
```

Start the Gradio UI:

```bash
make web
```

## Environment Variables

The project uses a `.env` file for configuration.

Example:

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/bi_db
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=1024
PRE_LOAD_DATA=false
LOG_LEVEL=INFO
WEB_HOST=0.0.0.0
WEB_PORT=7860
```

## Capstone Requirement Coverage

| Requirement Area | Implementation |
| --- | --- |
| Data ingestion | CSV, PDF, and pickle ingestion |
| Business analysis | Sales statistics, trends, product, region, and customer insights |
| Visual reporting | Dashboard PNG generated in the output folder |
| AI-powered insights | RAG chat over indexed business documents |
| Recommendations | Recommendation-style answers from retrieved analysis context |
| Memory/context | Gradio chat history passed through the conversation flow |
| Prompt chaining/orchestration | LangGraph-based flow |
| Evaluation | QAEvalChain/LLM evaluation with JSON report |
| User interface | Gradio UI with status, document management, evaluation, and chat |
| Persistence | PostgreSQL with pgvector |

## Output Artifacts

The main output artifacts are:

```text
output/csv_analysis_dashboard.png
output/csv_analysis_results.pkl
output/evaluation_report.json
```

These files can be used by the grader to verify that the analysis and evaluation workflows were executed.

## Conclusion

InsightForge demonstrates how AI can support business intelligence workflows by combining structured data analysis, document retrieval, vector search, and LLM-based question answering. The final Gradio interface gives users a single place to manage documents, inspect database status, run evaluation, and ask business questions.

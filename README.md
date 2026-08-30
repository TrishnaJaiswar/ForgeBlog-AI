# ForgeBlog-AI

<img width="1358" height="1358" alt="image" src="https://github.com/user-attachments/assets/8b89614d-4c5a-4c41-af61-82e3f2b55d5b" />


> Autonomous multi-agent technical blog generation using LangGraph, OpenRouter, Tavily & Hugging Face FLUX.

## Overview

ForgeBlog AI is an end-to-end AI content generation system where multiple specialized agents collaborate to produce publication-ready technical blogs. Instead of relying on a single prompt, the application plans, researches, writes, generates diagrams, and exports Markdown automatically.

## Key Features

* Multi-agent architecture built with LangGraph
* Automatic topic routing (Closed Book / Hybrid / Open Book)
* Real-time web research using Tavily Search
* Parallel section generation for faster writing
* AI-generated technical diagrams using FLUX.1-dev
* Automatic Markdown export with embedded images
* Streamlit frontend with interactive workflow

## Architecture

```text
                User Topic
                    │
                    ▼
              Router Agent
        (Research Decision)
                    │
        ┌───────────┴───────────┐
        │                       │
   Closed Book            Hybrid/Open
        │                       │
        ▼                       ▼
                 Tavily Research Agent
                         │
                         ▼
                  Orchestrator Agent
                  (Blog Planning)
                         │
                         ▼
            Parallel Writer Agents
      Section 1 · Section 2 · Section N
                         │
                         ▼
                  Content Merger
                         │
                         ▼
                 Image Planning Agent
                         │
                         ▼
          Hugging Face FLUX Generator
                         │
                         ▼
             Markdown + Images Export
```

## Tech Stack

| Layer            | Technology              |
| ---------------- | ----------------------- |
| Frontend         | Streamlit               |
| Workflow         | LangGraph               |
| LLM              | OpenRouter              |
| Research         | Tavily Search           |
| Image Generation | Hugging Face FLUX.1-dev |
| Language         | Python                  |
| Output           | Markdown + PNG          |

## Project Structure

```text
ForgeBlog-AI/
│
├── backend.py                # LangGraph agents
├── frontend.py               # Streamlit UI
├── images/                   # Generated diagrams
├── outputs/                  # Exported blogs
├── .env
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ForgeBlog-AI.git
cd ForgeBlog-AI
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
HF_TOKEN=your_huggingface_token
```

### 5. Launch the application

```bash
streamlit run frontend.py
```

## How It Works

### Step 1 — Topic Input

The user enters a technical topic in the Streamlit interface.

### Step 2 — Router Agent

Determines whether the topic requires web research.

* Closed Book
* Hybrid
* Open Book

### Step 3 — Research Agent

For Hybrid/Open Book topics, Tavily collects authoritative sources and converts them into structured evidence.

### Step 4 — Orchestrator

Creates:

* Blog title
* Audience
* Tone
* Section plan
* Writing constraints

### Step 5 — Parallel Writers

Multiple LangGraph worker agents generate different sections simultaneously.

### Step 6 — Image Planning

The editor agent inserts placeholders such as:

```text
[[IMAGE_1]]
[[IMAGE_2]]
```

and creates detailed prompts for each technical illustration.

### Step 7 — FLUX Image Generation

Hugging Face FLUX.1-dev generates diagrams which are saved inside:

```text
images/
```

### Step 8 — Export

The final blog is exported as:

```text
outputs/
    Self_Attention_in_Transformer.md
```

with embedded PNG images.

## Example Output

```markdown
# Self Attention in Transformer Architecture

## What is Self Attention?

Self attention allows every token to attend to every other token.

![QKV Flow](images/qkv_flow.png)
*Query, Key and Value information flow.*
```

## Future Improvements

* PDF export
* DOCX export
* Mermaid diagram generation
* Notion publishing
* GitHub Pages deployment
* Vector memory for previous blogs

## Author

**Trishna Jaiswar**

GenAI • Agentic AI • LangGraph • RAG Developer

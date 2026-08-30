# ForgeBlog AI

> An autonomous Agentic AI platform that researches, writes, and illustrates technical blogs using LangGraph, OpenRouter, Tavily, and Hugging Face FLUX.1-dev.

![Architecture](architecture.png)

## What is ForgeBlog AI?

ForgeBlog AI is an end-to-end autonomous technical blog generation platform. A user enters a topic (e.g. **Self Attention in Transformer Architecture**), and a team of AI agents collaborates to:

1. Decide whether web research is needed.
2. Search authoritative sources using Tavily.
3. Create a structured blog outline.
4. Write each section in parallel.
5. Decide where diagrams improve understanding.
6. Generate images using Hugging Face FLUX.1-dev.
7. Export a complete Markdown blog with embedded images.

The architecture is built with **LangGraph**, making every stage modular, parallel, and observable.

---

## Architecture

```text
                    User Topic
                        │
                        ▼
                 Router Agent
        (Closed Book / Hybrid / Open Book)
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
      Skip Research          Tavily Research
            │                       │
            └───────────┬───────────┘
                        ▼
               Orchestrator Agent
            (Blog Planning & Tasks)
                        │
                        ▼
          Parallel Writer Agents (N)
      Section 1 · Section 2 · Section N
                        │
                        ▼
                Content Merger
                        │
                        ▼
               Image Planner Agent
                        │
                        ▼
     Hugging Face FLUX.1-dev Generator
                        │
                        ▼
          Markdown + Embedded Images
```

---

## Workflow

![Workflow](workflow.png)

The workflow begins with a user topic and progresses through routing, research, planning, parallel writing, image generation, and Markdown export.

---

## Agent Responsibilities

| Agent              | Responsibility                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| **Router Agent**   | Classifies the topic as Closed Book, Hybrid, or Open Book and decides whether research is required.  |
| **Research Agent** | Uses Tavily Search to collect authoritative web sources and structured evidence.                     |
| **Orchestrator**   | Creates the complete blog title, audience, tone, sections, and writing constraints.                  |
| **Writer Agents**  | Multiple LangGraph workers generate different blog sections simultaneously.                          |
| **Image Planner**  | Determines where diagrams improve understanding and inserts `[[IMAGE_1]]` placeholders.              |
| **FLUX Generator** | Generates technical illustrations using Hugging Face FLUX.1-dev and embeds them into the final blog. |

---

## Tech Stack

| Layer                | Technology              |
| -------------------- | ----------------------- |
| **Workflow Engine**  | LangGraph               |
| **LLM**              | OpenRouter              |
| **Research**         | Tavily Search           |
| **Image Generation** | Hugging Face FLUX.1-dev |
| **Frontend**         | Streamlit               |
| **Backend**          | Python                  |

---

## Project Structure

```text
ForgeBlog-AI/
│
├── backend.py              # LangGraph multi-agent backend
├── frontend.py             # Streamlit UI
├── images/                 # Generated FLUX diagrams
├── outputs/
│   └── *.md                # Exported blogs
│
├── .env
├── requirements.txt
└── README.md
```

---

## Frontend

![Frontend](frontend.png)

The Streamlit interface allows users to:

* Enter any technical topic
* Generate a complete blog with one click
* Preview Markdown
* Automatically save generated diagrams
* Export production-ready `.md` files

---

## Backend Pipeline

| Stage              | Purpose                             |
| ------------------ | ----------------------------------- |
| **Router**         | Detects whether research is needed  |
| **Research**       | Collects web evidence using Tavily  |
| **Orchestrator**   | Creates the structured blog plan    |
| **Writer Agents**  | Generate sections in parallel       |
| **Reducer**        | Merges all generated sections       |
| **Image Planner**  | Inserts image placeholders          |
| **FLUX Generator** | Creates technical diagrams          |
| **Export**         | Saves Markdown with embedded images |

---

## Example Output

![Generated Blog](output.png)

The exported Markdown automatically embeds generated diagrams:

```markdown
# Self Attention in Transformer Architecture

## Understanding Self Attention

Self attention allows every token to attend to every other token.

![QKV Diagram](images/qkv_flow.png)

*Query, Key and Value information flow.*

## Multi-Head Attention

...
```

---

## Environment Variables

```env
OPENROUTER_API_KEY=your_key
TAVILY_API_KEY=your_key
HF_TOKEN=hf_xxxxxxxxxxxxx
```

---

## Installation

```bash
git clone https://github.com/yourusername/ForgeBlog-AI.git
cd ForgeBlog-AI

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

streamlit run frontend.py
```

---

## Features

* Multi-Agent LangGraph architecture
* Parallel section generation
* Automatic web research
* Citation-aware technical writing
* AI-generated technical diagrams
* Markdown export with embedded images
* Modular Streamlit + Python architecture

---

## Future Improvements

* PDF & DOCX export
* Mermaid diagram generation
* Notion publishing
* GitHub Pages integration
* Vector memory for previous blogs

---

## Author

**Trishna Jaiswar**

*GenAI • Agentic AI • LangGraph • RAG Developer*

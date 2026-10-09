# ForgeBlog AI

### Autonomous Agentic AI Platform for Technical Blog Generation

**ForgeBlog AI** is an Agentic AI platform that researches, plans, writes, merges, illustrates, and exports technical blog articles through a coordinated **LangGraph multi-agent workflow**.

Instead of relying on a single LLM call, ForgeBlog AI separates content creation into specialized stages, including conditional web research, structured planning, parallel section writing, content merging, and AI-generated technical illustrations.

**Live Demo:** https://forgeblog-ai.onrender.com  
**GitHub Repository:** https://github.com/TrishnaJaiswar/ForgeBlog-AI

> Research → Plan → Write → Merge → Illustrate → Export

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Agentic Workflow](#agentic-workflow)
- [Specialized Agents](#specialized-agents)
- [Technology Stack](#technology-stack)
- [Frontend](#frontend)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Engineering Highlights](#engineering-highlights)
- [Security](#security)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## Overview

ForgeBlog AI transforms a technical topic into a structured, research-supported blog with relevant AI-generated illustrations.

For example, a user can enter:

```text
Self-Attention in Transformer Architecture
```

The application can then:

1. Determine whether external research is required.
2. Search the web using Tavily.
3. Collect relevant sources and structured evidence.
4. Generate a detailed article plan.
5. Divide the article into independent writing tasks.
6. Generate sections through parallel writer agents.
7. Merge the sections into a coherent Markdown article.
8. Identify opportunities for technical illustrations.
9. Generate images using Hugging Face FLUX.1-dev.
10. Present the article, evidence, images, and workflow progress.
11. Allow the generated content and image bundle to be downloaded.

The backend uses **FastAPI** to expose application endpoints and **LangGraph** to orchestrate the stateful agent workflow. The frontend is built with HTML, CSS, and JavaScript.

## Features

- **Agentic blog generation** — coordinated, specialized agents handle different stages of content creation.
- **Conditional research routing** — selects Closed Book, Hybrid, or Open Book processing.
- **Web research** — retrieves external evidence using Tavily.
- **Structured planning** — creates article outlines and section-level writing tasks.
- **Parallel writing** — generates independent article sections through writer workers.
- **Content reduction and merging** — combines generated sections into an ordered article.
- **AI-generated illustrations** — uses Hugging Face FLUX.1-dev for technical visuals.
- **Live workflow progress** — streams generation events using Server-Sent Events (SSE).
- **Markdown preview** — renders the generated article in the browser.
- **Evidence viewer** — displays research evidence collected during generation.
- **Execution logs** — provides visibility into workflow progress.
- **Recent blog loading** — retrieves previously generated blogs when supported by the backend.
- **Downloads** — supports Markdown, generated images, and complete blog bundles through the relevant endpoints.

## Live Demo

Try ForgeBlog AI:

**https://forgeblog-ai.onrender.com**

The application is hosted on Render. Because hosted services may sleep or take time to start, the first request can be slower.

The availability of research and image generation depends on the deployment configuration, valid API credentials, and the limits of the external providers.

---

## Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d377eebf-6c46-4c67-9039-4d32994f8c69" />


### High-Level Architecture

```text
                       USER
                         |
                         v
                  WEB FRONTEND
                  HTML / CSS / JS
                         |
                         v
                    FASTAPI API
                         |
                         v
                  LANGGRAPH GRAPH
                         |
                         v
                    ROUTER AGENT
                         |
              +----------+----------+
              |          |          |
              v          v          v
          CLOSED BOOK   HYBRID   OPEN BOOK
              |          |          |
              |          v          v
              |       RESEARCH AGENT
              |          |
              +----------+
                         |
                         v
                 ORCHESTRATOR AGENT
                         |
                         v
                PARALLEL WRITER AGENTS
                         |
                         v
                   REDUCER / MERGER
                         |
                         v
                  IMAGE PLANNER AGENT
                         |
                         v
                    FLUX IMAGE AGENT
                         |
                         v
                    FINAL OUTPUT
                         |
            +------------+-------------+
            |            |             |
            v            v             v
         Markdown     Illustrations   Downloads
```

### Main Components

| Component | Responsibility |
|---|---|
| Web frontend | Topic input, progress display, article preview, evidence, images, and downloads |
| FastAPI | API endpoints, request validation, workflow invocation, streaming, and file delivery |
| LangGraph | Workflow state, routing, agent coordination, and execution |
| Tavily | External web research |
| OpenRouter | Access to configured language models |
| Hugging Face | FLUX.1-dev image generation |
| Pydantic | Structured data validation |

---

## Agentic Workflow

```text
START
  |
  v
Router Agent
  |
  +---- Closed Book -------------------+
  |                                    |
  +---- Hybrid ---> Research Agent ----+
  |                                    |
  +---- Open Book -> Research Agent ---+
                                       |
                                       v
                              Orchestrator Agent
                                       |
                                       v
                                Writer Workers
                                       |
                                       v
                                 Reducer
                                       |
                                       v
                                Image Planner
                                       |
                                       v
                                FLUX Generator
                                       |
                                       v
                                      END
```

The graph separates the task into stages with distinct responsibilities. Conditional routing avoids unnecessary research for topics that can be handled without external sources, while research-enabled paths collect evidence before planning and writing.

### Research Modes

| Mode | Purpose |
|---|---|
| Closed Book | Uses the model's existing knowledge without external research |
| Hybrid | Combines model knowledge with web research |
| Open Book | Uses external research as a primary input |

Actual routing behavior depends on the implemented router logic and model decisions.

---

## Specialized Agents

| Agent | Responsibility |
|---|---|
| **Router Agent** | Classifies the topic and determines the research mode |
| **Research Agent** | Searches the web and builds a structured evidence collection |
| **Orchestrator Agent** | Creates the article plan, audience, tone, sections, and writing tasks |
| **Writer Agents** | Generate individual sections, potentially in parallel |
| **Reducer / Merger** | Combines generated sections in the intended order |
| **Image Planner Agent** | Identifies where illustrations would improve understanding and prepares image specifications |
| **FLUX Image Agent** | Generates technical illustrations using Hugging Face FLUX.1-dev |

### Why a Multi-Agent Architecture?

A basic generator may follow this pattern:

```text
Topic → Single LLM Call → Blog
```

ForgeBlog AI instead decomposes the task:

```text
Topic
  ↓
Conditional Routing
  ↓
Research and Evidence
  ↓
Structured Planning
  ↓
Parallel Section Writing
  ↓
Content Merging
  ↓
Image Planning
  ↓
Image Generation
  ↓
Final Article
```

This design supports modularity, conditional execution, parallel work, and easier extension of individual workflow stages. It does not automatically guarantee factual accuracy or better quality; outputs still depend on retrieval quality, model behavior, and validation.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Programming language | Python |
| Agent orchestration | LangGraph |
| LLM access | OpenRouter |
| Web research | Tavily Search API |
| Image generation | Hugging Face FLUX.1-dev |
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Markdown rendering | Marked.js |
| Data validation | Pydantic |
| Configuration | python-dotenv |
| Deployment | Render |
| Version control | Git and GitHub |

---

## Frontend

The frontend is a custom web interface built with **HTML, CSS, and JavaScript**. It is not a React application.

### Interface capabilities

- Technical topic input
- Research date and research-window settings
- Blog generation
- Live agent progress
- Router, research, planner, writer, and reducer status
- Markdown article preview
- Plan and evidence viewers
- Generated image preview
- Execution logs
- Recent blog loading
- Markdown download
- Image download
- Complete blog bundle download

The browser communicates with the FastAPI backend, and SSE events allow workflow progress to be displayed as generation proceeds.

---

## API Endpoints

The following endpoints describe the documented application interface. Confirm the actual methods and request formats in the current FastAPI implementation or at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Root or health response, if configured |
| `GET` | `/api/blogs` | List available generated blogs |
| `GET` | `/api/blogs/{filename}` | Retrieve a generated Markdown blog |
| `POST` | `/api/generate` | Start blog generation and stream progress |
| `GET` | `/api/images/{filename}` | Serve a generated image |
| `GET` | `/api/download/markdown/{filename}` | Download a Markdown article |
| `GET` | `/api/download/bundle/{filename}` | Download a blog and its images as a ZIP |
| `GET` | `/api/download/images/{filename}` | Download generated images as a ZIP |

### Example Generation Request

```json
{
  "topic": "Self-Attention in Transformer Architecture",
  "as_of": "2026-10-09",
  "recency_days": 7
}
```

The exact fields accepted by the endpoint are defined by the backend's request schema. The example illustrates the intended request shape; check the implementation before relying on it.

### API Documentation

When running locally, open:

```text
http://127.0.0.1:8000/docs
```

For the deployed service, try:

https://forgeblog-ai.onrender.com/docs

FastAPI's Swagger UI is available if the deployment exposes the documentation route.

---

## Project Structure

The following is the documented logical structure. Adjust it to match the actual repository if filenames or directories have changed.

```text
ForgeBlog-AI/
├── backend.py
├── api.py
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── images/
│   └── generated illustrations
│
└── docs/
    └── architecture.png
```

### Core Files

| File | Purpose |
|---|---|
| `backend.py` | LangGraph workflow, state, and agent logic |
| `api.py` | FastAPI application and API endpoints |
| `frontend/index.html` | Main web interface |
| `frontend/css/style.css` | Layout and visual styling |
| `frontend/js/app.js` | Frontend behavior, API requests, SSE processing, and downloads |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for required environment variables |
| `images/` | Generated illustrations, depending on storage configuration |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TrishnaJaiswar/ForgeBlog-AI.git
cd ForgeBlog-AI
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root, following the format in the next section.

### 5. Run the Application

```bash
uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

This assumes `api.py` creates the FastAPI application as `app` and serves the frontend. If the current implementation uses a different entry point, use that module instead.

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token
```

These credentials are used for the configured LLM, web-research, and image-generation services.

**Important:** Use only the variables required by the actual code. Never commit real API keys, access tokens, or secrets to GitHub.

A suitable `.gitignore` entry is:

```gitignore
.env
.env.*
!.env.example
.venv/
venv/
__pycache__/
*.py[cod]
```

If `.env` has already been committed, remove it from version control and rotate any exposed credentials.

---

## Engineering Highlights

### 1. Stateful Agent Orchestration

LangGraph coordinates the workflow and passes structured state between processing stages.

### 2. Conditional Routing

The router selects a research mode based on the topic and configured decision logic.

### 3. Research-Augmented Generation

Tavily supplies external source material that can inform the planning and writing stages.

### 4. Parallel Section Writing

Independent writing tasks can be dispatched to separate workers, reducing the need to generate every section sequentially.

### 5. Structured Outputs

Pydantic models can validate workflow data such as routing decisions, plans, writing tasks, evidence, and image specifications.

### 6. Streaming

Server-Sent Events let the browser receive generation events without waiting for the entire workflow to finish.

### 7. Multimodal Generation

The image-planning stage identifies useful visual content, and the image-generation stage can create supporting illustrations.

### 8. Modular Design

The frontend, API, orchestration graph, and external model providers are separated into distinct components.

---

## Example Workflow

**Input**

```text
Build a production-ready RAG system using LangGraph
```

**Processing**

```text
Topic
  ↓
Router Agent
  ↓
Research Decision
  ↓
Tavily Research (when required)
  ↓
Evidence Collection
  ↓
Orchestrator
  ↓
Parallel Writer Agents
  ↓
Reducer
  ↓
Image Planner
  ↓
FLUX Image Generation
  ↓
Final Markdown Article
```

**Expected output types**

```text
generated_blog.md
images/
    technical_illustration_1.png
    technical_illustration_2.png
```

The actual filenames, number of images, and output format depend on the implementation and generation results.

---

## Security and Reliability

- Keep provider API keys in environment variables or the hosting platform's secret manager.
- Do not expose private credentials in frontend JavaScript.
- Validate incoming requests and uploaded or generated filenames.
- Restrict file access to intended application directories.
- Handle external API errors, rate limits, and timeouts.
- Treat retrieved web content as untrusted input.
- Display source information where available, and verify important claims independently.
- Consider persistent storage if generated content must survive service restarts or deployments.

---

## Future Improvements

Potential next steps include:

- [ ] Persistent database storage for generated blogs
- [ ] User authentication and multi-user workspaces
- [ ] Blog editing and version history
- [ ] PDF and DOCX export
- [ ] Mermaid diagram generation
- [ ] Semantic search over previously generated blogs
- [ ] Vector memory for previous articles
- [ ] LangSmith observability
- [ ] RAGAS or equivalent evaluation
- [ ] Automated content-quality scoring
- [ ] Hallucination detection and citation verification
- [ ] Notion and GitHub publishing integrations
- [ ] Scheduled blog generation
- [ ] Docker-based deployment
- [ ] CI/CD automation

These are proposed improvements, not claims that the features are currently implemented.

---

## Author

**Trishna Jaiswar**  
2026 B.E. Graduate — Artificial Intelligence and Data Science

Interested in building practical AI engineering systems involving:

- Generative AI
- Agentic AI and multi-agent workflows
- LangGraph and LangChain
- Retrieval-Augmented Generation (RAG)
- Python and FastAPI
- LLM-powered applications

**GitHub:** https://github.com/TrishnaJaiswar  
**LinkedIn:** https://linkedin.com/in/trishna-jaiswar-a6a230322

---

## License

This project is intended for educational, research, experimentation, and portfolio purposes. Add a `LICENSE` file to the repository if you want to grant users explicit permissions under a particular open-source license.

# ForgeBlog AI

> Autonomous Agentic AI platform for researching, planning, writing, and
> illustrating technical blogs using LangGraph, OpenRouter, Tavily,
> FastAPI, and Hugging Face FLUX.1-dev.

```{=html}
<p align="center">
```
`<strong>`{=html}Research → Plan → Write → Merge → Illustrate →
Export`</strong>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Overview

ForgeBlog AI is an **Agentic AI technical blog generation platform**
designed to autonomously research, plan, write, illustrate, and export
high-quality technical articles.

Instead of using a single LLM call to generate an entire blog, ForgeBlog
AI uses a **LangGraph-based multi-agent architecture** where specialized
agents collaborate through a shared workflow state.

A user provides a technical topic such as:

``` text
Self Attention in Transformer Architecture
```

ForgeBlog AI can then:

1.  Analyze whether external research is required.
2.  Search authoritative sources using Tavily.
3.  Build a structured evidence pack.
4.  Create a detailed blog plan.
5.  Divide the blog into multiple sections.
6.  Generate sections using parallel writer agents.
7.  Merge the sections into a complete article.
8.  Determine where technical illustrations are useful.
9.  Generate illustrations using Hugging Face FLUX.1-dev.
10. Export the final technical blog as Markdown.
11. Display evidence, images, logs, and the generated article through a
    web interface.

The workflow is orchestrated using **LangGraph**, while **FastAPI**
provides the backend API and the frontend is built using **HTML, CSS,
and JavaScript**.

------------------------------------------------------------------------

# Architecture
<img width="1536" height="1024" alt="643278613-e3115c6b-8c61-40b9-bb56-d6e9ceb5db4f" src="https://github.com/user-attachments/assets/db7d1a79-1e01-4b09-8ac2-285239ab7b9f" />



## High-Level Architecture

``` text
                           ┌──────────────────────┐
                           │        USER          │
                           │                      │
                           │    Enter Topic       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────┐
                    │       WEB FRONTEND             │
                    │       HTML / CSS / JS          │
                    │                                │
                    │ • Topic Input                  │
                    │ • Generate Blog                │
                    │ • Live Progress                │
                    │ • Markdown Preview              │
                    │ • Evidence                      │
                    │ • Images                        │
                    │ • Logs                          │
                    │ • Downloads                     │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │          FASTAPI API            │
                    │                                │
                    │        /api/generate            │
                    │        /api/blogs               │
                    │        /api/images              │
                    │        /api/download             │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │      LANGGRAPH WORKFLOW         │
                    │                                │
                    │        Stateful Graph           │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    ROUTER AGENT     │
                         │                     │
                         │ Closed Book         │
                         │ Hybrid              │
                         │ Open Book           │
                         └──────────┬──────────┘
                                    │
                         Research Required?
                            ┌───────┴───────┐
                            │               │
                           YES              NO
                            │               │
                            ▼               │
                    ┌───────────────┐       │
                    │ RESEARCH      │       │
                    │ AGENT         │       │
                    │               │       │
                    │ Tavily Search │       │
                    │ Evidence      │       │
                    └───────┬───────┘       │
                            │               │
                            └───────┬───────┘
                                    ▼
                         ┌─────────────────────┐
                         │  ORCHESTRATOR AGENT │
                         │                     │
                         │ • Title             │
                         │ • Audience          │
                         │ • Tone               │
                         │ • Sections           │
                         │ • Constraints        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │      PARALLEL WRITER AGENTS    │
                    │                                │
                    │  Writer 1 → Section 1          │
                    │  Writer 2 → Section 2          │
                    │  Writer 3 → Section 3          │
                    │  Writer N → Section N          │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   REDUCER / MERGER  │
                         │                     │
                         │ Ordered Final Draft │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  IMAGE PLANNER      │
                         │                     │
                         │ Detects useful      │
                         │ illustration points │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FLUX IMAGE AGENT  │
                         │                     │
                         │ Hugging Face        │
                         │ FLUX.1-dev          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │          FINAL OUTPUT          │
                    │                                │
                    │ • Markdown Blog                │
                    │ • Generated Images             │
                    │ • Research Evidence            │
                    │ • Execution Logs               │
                    └────────────────────────────────┘
```

------------------------------------------------------------------------

# Agentic Workflow

ForgeBlog AI uses a stateful LangGraph workflow:

``` text
START
  │
  ▼
Router Agent
  │
  ├──────── Closed Book ───────────────┐
  │                                    │
  ├──────── Hybrid ──► Research ───────┤
  │                                    │
  └──────── Open Book ─► Research ─────┤
                                       │
                                       ▼
                              Orchestrator Agent
                                       │
                                       ▼
                              Parallel Writers
                                       │
                                       ▼
                                  Reducer
                                       │
                                       ▼
                              Image Planner
                                       │
                                       ▼
                              FLUX Generator
                                       │
                                       ▼
                                      END
```

The graph separates responsibilities into independent stages, allowing
the workflow to be extended with additional agents without redesigning
the entire system.

------------------------------------------------------------------------

# Agents

  -----------------------------------------------------------------------
  Agent                               Responsibility
  ----------------------------------- -----------------------------------
  **Router Agent**                    Classifies the topic as Closed
                                      Book, Hybrid, or Open Book and
                                      determines whether external
                                      research is required.

  **Research Agent**                  Uses Tavily Search to retrieve
                                      relevant web sources and build
                                      structured evidence.

  **Orchestrator Agent**              Creates the blog title, audience,
                                      tone, sections, constraints, and
                                      writing tasks.

  **Writer Agents**                   Generate individual blog sections
                                      in parallel using LangGraph
                                      workers.

  **Reducer / Merger**                Combines all generated sections
                                      into the correct final article
                                      order.

  **Image Planner Agent**             Determines where diagrams or
                                      illustrations can improve
                                      understanding and creates image
                                      specifications.

  **FLUX Image Agent**                Generates technical illustrations
                                      using Hugging Face FLUX.1-dev.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Why a Multi-Agent Architecture?

A conventional AI blog generator might use:

``` text
Topic
  ↓
Single LLM Call
  ↓
Blog
```

ForgeBlog AI instead uses:

``` text
Topic
  ↓
Router
  ↓
Research
  ↓
Planning
  ↓
Parallel Writing
  ↓
Reduction
  ↓
Image Planning
  ↓
Image Generation
  ↓
Final Blog
```

This architecture provides better separation of concerns.

Each agent focuses on a specific responsibility rather than forcing one
LLM call to perform the entire workflow.

### Benefits

-   Specialized agent responsibilities
-   Conditional execution
-   External web research
-   Structured planning
-   Parallel section generation
-   State-based workflow orchestration
-   Evidence-aware content generation
-   Automated image planning
-   AI-generated technical illustrations
-   Observable workflow progress
-   Modular architecture
-   Easier debugging and future extension

------------------------------------------------------------------------

# Router Agent

The Router Agent is the first decision-making component.

It determines whether the topic can be handled using the model's
existing knowledge or whether external research is required.

### Modes

``` text
Closed Book
    │
    └── No external research

Hybrid
    │
    └── Model knowledge + web research

Open Book
    │
    └── External research required
```

This prevents unnecessary web searches for topics where external
information may not be required.

------------------------------------------------------------------------

# Research Agent

When research is required, ForgeBlog AI uses **Tavily Search**.

The Research Agent:

1.  Receives research queries.
2.  Searches the web.
3.  Retrieves relevant sources.
4.  Extracts useful information.
5.  Builds structured evidence.
6.  Passes the evidence to the downstream planning and writing stages.

The evidence model contains structured information such as:

``` text
Title
URL
Published Date
Snippet
Source
```

This allows downstream agents to generate content using retrieved
information instead of relying exclusively on model memory.

------------------------------------------------------------------------

# Orchestrator Agent

The Orchestrator transforms the topic and available evidence into a
structured writing plan.

The plan contains information such as:

``` text
Blog Title
Audience
Tone
Blog Type
Constraints
Tasks
```

Each writing task can contain:

``` text
Task ID
Section Title
Goal
Key Points
Target Word Count
Tags
Research Requirement
Citation Requirement
Code Requirement
```

This structured planning stage allows the workflow to distribute
individual sections to separate writer agents.

------------------------------------------------------------------------

# Parallel Writer Agents

One of the major architectural features of ForgeBlog AI is **parallel
section generation**.

Instead of generating an article sequentially:

``` text
Section 1
   ↓
Section 2
   ↓
Section 3
   ↓
Section 4
```

LangGraph can distribute independent sections across multiple writer
workers:

``` text
                       ┌── Writer 1 ──► Section 1
                       │
                       ├── Writer 2 ──► Section 2
                       │
Orchestrator ──────────┼── Writer 3 ──► Section 3
                       │
                       ├── Writer 4 ──► Section 4
                       │
                       └── Writer N ──► Section N
```

After generation, the sections are sent to the Reducer.

This makes the architecture more suitable for long-form technical
content.

------------------------------------------------------------------------

# Reducer / Merger

The Reducer is responsible for assembling the independently generated
sections into a coherent article.

Its responsibilities include:

-   Collecting generated sections.
-   Maintaining section ordering.
-   Combining the sections.
-   Producing the complete Markdown draft.
-   Passing the draft to the image planning stage.

Conceptually:

``` text
Section 1 ─┐
Section 2 ─┤
Section 3 ─┼──► Reducer ──► Complete Blog
Section 4 ─┤
Section N ─┘
```

------------------------------------------------------------------------

# Image Planner Agent

After the blog content is assembled, the Image Planner determines where
visual explanations could improve the article.

It can create structured image specifications containing:

``` text
Placeholder
Filename
Alt Text
Caption
Prompt
Image Size
Quality
```

Example:

``` text
[[IMAGE_1]]
```

can be associated with:

``` text
Filename:
self_attention_flow.png

Alt:
Self-attention query key value flow diagram

Caption:
Flow of queries, keys, and values through self-attention.

Prompt:
Technical educational diagram explaining QKV attention...
```

------------------------------------------------------------------------

# FLUX Image Agent

The FLUX Image Agent generates technical illustrations using:

**Hugging Face FLUX.1-dev**

The generated images are stored in the project's image directory and
exposed through the FastAPI backend.

Example Markdown:

``` markdown
![Self Attention Architecture](images/self_attention_architecture.png)
```

This allows generated diagrams to become part of the final technical
article.

------------------------------------------------------------------------

# Frontend

The current ForgeBlog AI frontend is a custom web application built
using:

-   HTML
-   CSS
-   JavaScript
-   Marked.js

The frontend is no longer dependent on Streamlit.

## Frontend Features

-   Technical topic input
-   Research date selection
-   Research window selection
-   Blog generation
-   Live agent progress
-   Router status
-   Research status
-   Planner status
-   Writer status
-   Reducer status
-   Markdown preview
-   Plan viewer
-   Evidence viewer
-   Image preview
-   Execution logs
-   Recent generated blogs
-   Markdown download
-   Complete bundle download
-   Generated image download

------------------------------------------------------------------------

# Live Workflow Progress

ForgeBlog AI uses **Server-Sent Events (SSE)** to stream generation
progress from FastAPI to the frontend.

The frontend can receive events such as:

``` text
started
   ↓
router
   ↓
research
   ↓
orchestrator
   ↓
writer
   ↓
writer
   ↓
reducer
   ↓
image_planner
   ↓
image_generator
   ↓
complete
```

This allows the user to see the agentic workflow while the blog is being
generated.

------------------------------------------------------------------------

# FastAPI Backend

FastAPI acts as the API layer between the web frontend and the LangGraph
workflow.

The backend is responsible for:

-   Serving the frontend.
-   Starting blog generation.
-   Streaming workflow progress.
-   Returning generated blogs.
-   Serving generated images.
-   Providing downloads.
-   Handling API errors.
-   Connecting the frontend with the LangGraph application.

------------------------------------------------------------------------

# API Endpoints

  -------------------------------------------------------------------------------------
  Endpoint                              Method                  Purpose
  ------------------------------------- ----------------------- -----------------------
  `/api/blogs`                          `GET`                   Lists available
                                                                generated blogs.

  `/api/blogs/{filename}`               `GET`                   Returns a generated
                                                                Markdown blog.

  `/api/generate`                       `POST`                  Starts the LangGraph
                                                                blog generation
                                                                workflow.

  `/api/images/{filename}`              `GET`                   Serves generated
                                                                images.

  `/api/download/markdown/{filename}`   `GET`                   Downloads a Markdown
                                                                blog.

  `/api/download/bundle/{filename}`     `GET`                   Downloads the blog and
                                                                generated images as a
                                                                ZIP bundle.

  `/api/download/images/{filename}`     `GET`                   Downloads generated
                                                                images as a ZIP
                                                                archive.
  -------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Generate API

The frontend sends a request similar to:

``` json
{
  "topic": "Self Attention in Transformer Architecture",
  "as_of": "2026-10-09",
  "recency_days": 7
}
```

The FastAPI backend passes the request into the LangGraph workflow.

The generation endpoint returns progress using Server-Sent Events.

------------------------------------------------------------------------

# Outputs

ForgeBlog AI produces multiple outputs during the workflow.

## 1. Markdown Blog

``` text
*.md
```

The final technical article generated by the multi-agent workflow.

## 2. Generated Images

``` text
images/
```

Contains AI-generated technical illustrations.

## 3. Research Evidence

The Research Agent produces structured evidence containing relevant
sources and metadata.

## 4. Execution Logs

The frontend displays agent execution progress and workflow information.

------------------------------------------------------------------------

# Example Workflow

### Input

``` text
Self Attention in Transformer Architecture
```

### Processing

``` text
User Topic
    ↓
Router Agent
    ↓
Research Required?
    ↓
Tavily Research
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
Final Markdown
```

### Output

``` text
Self Attention in Transformer Architecture.md

images/
├── qkv_flow.png
├── attention_architecture.png
└── multi_head_attention.png
```

------------------------------------------------------------------------

# Example Generated Markdown

``` markdown
# Self Attention in Transformer Architecture

## Introduction

Self-attention allows each token in a sequence
to interact with other tokens in the sequence.

## Query, Key and Value

The attention mechanism transforms the input
into Query, Key, and Value representations.

![QKV Architecture](images/qkv_architecture.png)

## Multi-Head Attention

Multi-head attention allows transformer models
to capture different relationships between tokens.

...
```

------------------------------------------------------------------------

# Project Structure

``` text
ForgeBlog-AI/
│
├── backend.py
├── api.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── frontend/
│   ├── index.html
│   │
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── app.js
│
├── images/
│
└── docs/
    └── architecture.png
```

------------------------------------------------------------------------

# Core Files

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `backend.py`                        LangGraph multi-agent workflow and
                                      agent logic.

  `api.py`                            FastAPI backend, REST endpoints,
                                      SSE generation, downloads, and
                                      frontend serving.

  `frontend/index.html`               Main web application interface.

  `frontend/css/style.css`            Frontend layout and visual styling.

  `frontend/js/app.js`                Frontend state management, API
                                      communication, SSE handling,
                                      rendering, and downloads.

  `images/`                           Generated technical illustrations.

  `docs/architecture.png`             System architecture diagram.

  `requirements.txt`                  Python dependency list.

  `.gitignore`                        Files excluded from version
                                      control.

  `.env`                              Local API credentials and
                                      configuration.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Technology Stack

  Layer                           Technology
  ------------------------------- -------------------------
  **Programming Language**        Python
  **Agent Orchestration**         LangGraph
  **LLM Access**                  OpenRouter
  **Research**                    Tavily Search API
  **Image Generation**            Hugging Face FLUX.1-dev
  **Backend API**                 FastAPI
  **Frontend**                    HTML, CSS, JavaScript
  **Markdown Rendering**          Marked.js
  **Data Validation**             Pydantic
  **Environment Configuration**   python-dotenv
  **Version Control**             Git / GitHub

------------------------------------------------------------------------

# Environment Variables

Create a `.env` file in the project root.

``` env
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token
```

### Important

Never commit `.env` to GitHub.

Your API keys should only exist in your local environment or secure
deployment environment.

The `.gitignore` file should contain:

``` gitignore
.env
.env.*
!.env.example
```

------------------------------------------------------------------------

# Installation

## 1. Clone the Repository

``` bash
git clone https://github.com/TrishnaJaiswar/ForgeBlog-AI.git
```

``` bash
cd ForgeBlog-AI
```

## 2. Create a Virtual Environment

### Windows

``` powershell
python -m venv project
project\Scripts\activate
```

### Linux / macOS

``` bash
python -m venv project
source project/bin/activate
```

## 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create:

``` text
.env
```

Then add:

``` env
OPENROUTER_API_KEY=your_key
TAVILY_API_KEY=your_key
HF_TOKEN=your_key
```

------------------------------------------------------------------------

# Running the Application

Start the FastAPI server:

``` bash
uvicorn api:app --reload
```

The application will start at:

``` text
http://127.0.0.1:8000
```

Open the address in your browser.

The FastAPI server serves both the API and the frontend.

------------------------------------------------------------------------

# Development Mode

For development, use:

``` bash
uvicorn api:app --reload
```

This enables automatic server reload whenever backend Python files
change.

------------------------------------------------------------------------

# API Documentation

FastAPI automatically provides interactive API documentation.

Once the server is running, open:

``` text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to inspect and test the API endpoints.

------------------------------------------------------------------------

# Engineering Highlights

ForgeBlog AI demonstrates several important AI engineering concepts.

## 1. Agentic AI

The system uses multiple specialized agents instead of a single
monolithic LLM call.

``` text
Router
   ↓
Research
   ↓
Planner
   ↓
Workers
   ↓
Reducer
   ↓
Image Planner
   ↓
Image Generator
```

## 2. Stateful Workflow

LangGraph manages the workflow state between agents.

The state contains information such as:

``` text
Topic
Research Mode
Queries
Evidence
Plan
Sections
Merged Markdown
Image Specifications
Final Output
```

This enables agents to communicate through structured shared state.

## 3. Conditional Routing

The Router Agent determines whether research is required.

``` text
                 ┌── Closed Book
                 │
Router ──────────┼── Hybrid ──► Research
                 │
                 └── Open Book ─► Research
```

## 4. Parallel Execution

Independent writing tasks can be distributed across multiple LangGraph
workers.

``` text
             ┌── Section 1
             │
             ├── Section 2
             │
Planner ─────┼── Section 3
             │
             └── Section N
```

This is more scalable than generating every section sequentially.

## 5. Structured Outputs

Pydantic models are used for structured data such as:

-   Blog plans
-   Tasks
-   Evidence
-   Router decisions
-   Image specifications

This makes the workflow easier to validate and maintain.

## 6. Streaming

FastAPI uses Server-Sent Events to stream workflow progress from the
backend to the browser.

This creates a more transparent user experience during long-running AI
generation tasks.

## 7. Modular Architecture

The system separates the application into distinct layers:

``` text
┌─────────────────────────────┐
│          Frontend           │
│       HTML / CSS / JS       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          FastAPI            │
│          API Layer          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         LangGraph           │
│     Agentic Workflow        │
└──────────────┬──────────────┘
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
   OpenRouter Tavily   Hugging Face
```

This separation makes the application easier to maintain and extend.

------------------------------------------------------------------------

# Security Considerations

ForgeBlog AI uses API credentials for external services.

Sensitive credentials should never be placed directly inside source
code.

Do not commit:

``` text
.env
API keys
Access tokens
Private credentials
```

Sensitive files are excluded through `.gitignore`.

For production deployment, environment variables should be configured
through the deployment platform's secret management system.

------------------------------------------------------------------------

# Current Features

-   [x] Multi-Agent LangGraph architecture
-   [x] Router Agent
-   [x] Closed Book / Hybrid / Open Book routing
-   [x] Tavily web research
-   [x] Structured evidence collection
-   [x] Blog planning
-   [x] Parallel writer agents
-   [x] Reducer / content merger
-   [x] Image planning
-   [x] Hugging Face FLUX.1-dev image generation
-   [x] FastAPI backend
-   [x] HTML/CSS/JavaScript frontend
-   [x] Server-Sent Events
-   [x] Live generation progress
-   [x] Markdown preview
-   [x] Research evidence viewer
-   [x] Generated image preview
-   [x] Execution logs
-   [x] Recent blog loading
-   [x] Markdown download
-   [x] Image download
-   [x] Complete blog bundle download

------------------------------------------------------------------------

# Future Improvements

The following improvements can extend the platform further:

-   [ ] Persistent database storage
-   [ ] User authentication
-   [ ] Multi-user workspaces
-   [ ] Blog versioning
-   [ ] Blog editing
-   [ ] PDF export
-   [ ] DOCX export
-   [ ] Mermaid diagram generation
-   [ ] Vector memory for previous blogs
-   [ ] Semantic search over previously generated blogs
-   [ ] RAG over previous articles
-   [ ] LangSmith observability
-   [ ] RAGAS evaluation
-   [ ] Automated content quality scoring
-   [ ] Hallucination detection
-   [ ] Citation verification
-   [ ] Notion publishing
-   [ ] GitHub publishing
-   [ ] Scheduled blog generation
-   [ ] Cloud deployment
-   [ ] Docker containerization
-   [ ] CI/CD pipeline

------------------------------------------------------------------------

# Roadmap

## Phase 1 --- Core Agentic Workflow

``` text
Router
   ↓
Research
   ↓
Planner
   ↓
Parallel Writers
   ↓
Reducer
```

Completed.

## Phase 2 --- Multimodal Generation

``` text
Reducer
   ↓
Image Planner
   ↓
FLUX
   ↓
Technical Illustrations
```

Completed.

## Phase 3 --- Production Engineering

Planned:

``` text
Docker
   ↓
CI/CD
   ↓
Cloud Deployment
   ↓
Observability
   ↓
Evaluation
```

## Phase 4 --- Persistent AI Knowledge

Planned:

``` text
Generated Blogs
       ↓
Embeddings
       ↓
Vector Database
       ↓
Semantic Retrieval
       ↓
Blog Memory
```

This would allow ForgeBlog AI to reuse previously generated knowledge
and maintain long-term project memory.

------------------------------------------------------------------------

# Example Use Cases

ForgeBlog AI can be used for generating technical content around topics
such as:

### Artificial Intelligence

``` text
Transformer Architecture
Attention Mechanisms
Large Language Models
Generative AI
Agentic AI
Multi-Agent Systems
```

### Machine Learning

``` text
Gradient Descent
CNNs
RNNs
Transfer Learning
Model Evaluation
Feature Engineering
```

### Software Engineering

``` text
Microservices
REST APIs
Distributed Systems
Docker
Kubernetes
CI/CD
System Design
```

### Generative AI

``` text
RAG
AI Agents
LangChain
LangGraph
Vector Databases
LLM Evaluation
Prompt Engineering
```

------------------------------------------------------------------------

# Example

## Input

``` text
Build a production-ready RAG system using LangGraph
```

## Agent Workflow

``` text
Router
   ↓
Research Required
   ↓
Tavily
   ↓
Evidence Pack
   ↓
Orchestrator
   ↓
Section Tasks
   ↓
Parallel Writers
   ↓
Reducer
   ↓
Image Planner
   ↓
FLUX
   ↓
Final Markdown
```

## Final Output

``` text
Production Ready RAG with LangGraph.md
```

with generated technical diagrams stored under:

``` text
images/
```

------------------------------------------------------------------------

# GitHub

Repository:

https://github.com/TrishnaJaiswar/ForgeBlog-AI

------------------------------------------------------------------------

# Author

## Trishna Jaiswar

**2026 B.E. Graduate --- Artificial Intelligence & Data Science**

Interested in building production-oriented AI systems with:

-   Generative AI
-   Agentic AI
-   LangGraph
-   LangChain
-   RAG
-   LLM Applications
-   Multi-Agent Systems
-   AI Engineering
-   Python
-   FastAPI

------------------------------------------------------------------------

# Project Summary

ForgeBlog AI demonstrates how a complex content-generation task can be
decomposed into a coordinated **multi-agent AI workflow**.

Instead of:

``` text
Prompt → LLM → Output
```

ForgeBlog AI implements:

``` text
User
  ↓
Router Agent
  ↓
Conditional Research
  ↓
Evidence Collection
  ↓
Orchestrated Planning
  ↓
Parallel Writing
  ↓
Content Reduction
  ↓
Image Planning
  ↓
AI Image Generation
  ↓
Markdown Export
```

The project combines **Agentic AI, LangGraph orchestration, web
research, parallel execution, structured outputs, multimodal generation,
FastAPI, and a custom web frontend** into a single end-to-end
application.

------------------------------------------------------------------------

# License

This project is intended for educational, research, experimentation, and
portfolio purposes.

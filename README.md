# ForgeBlog AI

> Autonomous Agentic AI platform for researching, planning, writing, and illustrating technical blogs using LangGraph, OpenRouter, Tavily, FastAPI, and Hugging Face FLUX.1-dev.

<p align="center">
  <strong>Research → Plan → Write → Merge → Illustrate → Export</strong>
</p>

---

## Overview

ForgeBlog AI is an **Agentic AI technical blog generation platform** designed to autonomously research, plan, write, illustrate, and export high-quality technical articles.

Instead of using a single LLM call to generate an entire blog, ForgeBlog AI uses a **LangGraph-based multi-agent architecture** where specialized agents collaborate through a shared workflow state.

A user provides a technical topic such as:

```text
Self Attention in Transformer Architecture

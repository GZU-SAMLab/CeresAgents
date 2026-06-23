# CeresAgents

CeresAgents is a multi-agent crop disease diagnosis system that combines:

- an LLM-based coordinator built with LangGraph
- specialist agents for pathology, physiology, and pesticide compliance
- image-based disease classification and lesion severity estimation
- literature retrieval over a local vector store
- an optional Neo4j knowledge graph for pesticide lookup

The repository currently includes a CLI entrypoint, a Flask API service, RAG utilities, prompt definitions, and local vision service wrappers.

## Overview

The system routes each user request into one of two paths:

- `fast`: simple agricultural knowledge or lightweight image triage
- `expert`: multi-step diagnosis and recommendation with specialist agents

In the expert route, the coordinator can activate:

- `Pathologist`: symptom analysis and disease identification
- `Physiologist`: environmental and abiotic-stress auditing
- `Chemist`: pesticide lookup, registration checks, and safety/compliance reasoning

The final response is synthesized into a farmer-facing English report.
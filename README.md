# CeresAgents

CeresAgents is a multi-agent crop disease diagnosis and treatment system. It combines LLM-based specialist collaboration with vision analysis, literature retrieval, and pesticide knowledge support to produce diagnosis and prescription recommendations.

This repository contains the main inference code, API service, and a simplified evaluation pipeline for the public release.

## Structure

```text
.
|-- api/
|-- eval/
|-- rag/
|-- tools_service/
|-- ceres_agents.py
|-- cli_interface.py
|-- prompts.py
|-- requirements.txt
`-- tools.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Required:

```env
MODEL_API_KEY=your_key
MODEL_NAME=qwen-plus
MODEL_BASE_URL=https://your-endpoint/v1
```

Optional:

```env
PLANT_API_BASE_URL=http://localhost:10001
LITERATURE_DB_PATH=/path/to/vector_db
LITERATURE_COLLECTION=literature

CERES_CLASSIFIER_MODEL_PATH=/path/to/best_model.pth
CERES_SEG_LEAF_MODEL_PATH=/path/to/seg_leaf_best_model.pth
CERES_SEG_LESION_MODEL_PATH=/path/to/seg_lesion_best_model.pth

DIRECT_LLM_API_KEY=
DIRECT_LLM_BASE_URL=
DIRECT_LLM_MODEL=gpt-4o

JUDGE_API_KEY=
JUDGE_BASE_URL=
JUDGE_MODEL=deepseek-chat
```

## Run

CLI:

```bash
python cli_interface.py
```

API:

```bash
python api/server.py --host 0.0.0.0 --port 10001
```

Evaluation with CeresAgents:

```bash
python eval/run_eval.py --system ceres_full --cases /path/to/cases.jsonl
```

Evaluation with a direct LLM baseline:

```bash
python eval/run_eval.py --system direct_llm --cases /path/to/cases.jsonl
```

Generate responses only:

```bash
python eval/run_eval.py --system ceres_full --cases /path/to/cases.jsonl --generate-only
```

## Notes

- The public repository does not bundle the full benchmark, model weights, vector database, or knowledge graph data.
- Evaluation outputs are written to `eval_results/`.

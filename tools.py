import os
import re
import json
import base64
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from rag import LiteratureRetriever
from prompts import get_image_analyze_prompt

try:
    from openai import OpenAI  # DashScope 兼容
except ImportError:
    OpenAI = None  # type: ignore

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None  # type: ignore

try:
    from dotenv import load_dotenv, find_dotenv
    _loaded = load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    _loaded = False

# Project root (directory containing this module)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Upload folder (default: project uploads/)
_default_upload_folder = os.path.join(PROJECT_ROOT, "uploads")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", _default_upload_folder)

if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"[OK] Created upload folder: {UPLOAD_FOLDER}")
    except Exception as e:
        print(f"[WARN] Could not create upload folder {UPLOAD_FOLDER}: {e}")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "123456789")
API_BASE = os.getenv("PLANT_API_BASE_URL", "http://localhost:10001")


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _err(code: int, msg: str) -> str:
    """One-line JSON error payload for tool returns."""
    return json.dumps({"code": code, "error": msg}, ensure_ascii=False)


def _truncate(text: Optional[str], limit: int = 400) -> str:
    """Clip long Chinese KG text blocks so tool observations stay small."""
    if not text:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def _web_search_enabled() -> bool:
    """Web search is off by default. Set ENABLE_WEB_SEARCH=1 to enable; DISABLE_WEB_SEARCH=1 forces off."""
    if _is_truthy(os.getenv("DISABLE_WEB_SEARCH")):
        return False
    if os.getenv("ENABLE_WEB_SEARCH") is None:
        return False
    return _is_truthy(os.getenv("ENABLE_WEB_SEARCH"))

_neo4j_driver = None
_literature_retriever: Optional[LiteratureRetriever] = None


def _default_literature_db_path(collection: str) -> Optional[str]:
    bundled_root = os.path.join(PROJECT_ROOT, "knowledge_base", "vector_db")
    if collection == "literature":
        candidate = os.path.join(bundled_root, "vector_db_new_para", "literature")
    else:
        candidate = os.path.join(bundled_root, collection)
    return candidate if os.path.exists(candidate) else None


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    if GraphDatabase is None:
        return None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        _neo4j_driver = driver
        return _neo4j_driver
    except Exception:
        _neo4j_driver = None
        return None


def _get_literature_retriever() -> Optional[LiteratureRetriever]:
    global _literature_retriever
    if _literature_retriever is not None:
        return _literature_retriever

    collection = os.getenv("LITERATURE_COLLECTION", "literature")
    persist_dir = (os.getenv("LITERATURE_DB_PATH") or "").strip() or _default_literature_db_path(collection)
    if not persist_dir:
        return None

    try:
        _literature_retriever = LiteratureRetriever(
            collection_name=collection,
            persist_directory=persist_dir,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Literature retriever init failed: {exc}")
        _literature_retriever = None
    return _literature_retriever


def _require_openai_client(purpose: str = "text"):
    if OpenAI is None:
        return None, "openai package not installed; run: pip install openai"

    if purpose == "image":
        api_key = os.getenv("IMAGE_MODEL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return None, "Missing IMAGE_MODEL_API_KEY (or DASHSCOPE_API_KEY)"
        base_url = (
            os.getenv("IMAGE_MODEL_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        api_key = (
            os.getenv("MODEL_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        if not api_key:
            return None, "Missing MODEL_API_KEY (or DASHSCOPE_API_KEY / OPENAI_API_KEY / LLM_API_KEY)"
        base_url = os.getenv("MODEL_BASE_URL") or os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, None
    except Exception as e:
        return None, f"OpenAI client init failed: {e}"


def _tavily_search(query: str, max_results: int):
    if not _web_search_enabled():
        return {"error": "web search disabled"}
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "缺少 TAVILY_API_KEY"}
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic"
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}", "details": r.text}
    except Exception as e:
        return {"error": str(e)}


def _post_json(endpoint: str, data: dict, timeout: int = 30):
    url = f"{API_BASE}{endpoint}"
    try:
        r = requests.post(url, json=data, timeout=timeout)
        if r.status_code == 200:
            return {"code": 200, "msg": "ok", "data": r.json(), "endpoint": endpoint}
        return {"code": r.status_code, "error": r.text, "endpoint": endpoint}
    except Exception as e:
        return {"code": 500, "error": str(e), "endpoint": endpoint}

def _resolve_image_path(file_name: str) -> str:
    if not file_name or not str(file_name).strip():
        return ""
    file_name = str(file_name).strip()
    if os.path.isabs(file_name):
        return os.path.normpath(file_name)
    if file_name.startswith("./") or file_name.startswith("../"):
        return os.path.normpath(os.path.abspath(file_name))
    repo_rel = os.path.normpath(os.path.join(PROJECT_ROOT, file_name))
    uploads_rel = os.path.normpath(os.path.join(UPLOAD_FOLDER, file_name))
    if os.path.isfile(repo_rel):
        return repo_rel
    if os.path.isfile(uploads_rel):
        return uploads_rel
    looks_repo_relative = (os.sep in file_name) or ("/" in file_name)
    return repo_rel if looks_repo_relative else uploads_rel


def _check_plant_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


class ImageAnalyzeInput(BaseModel):
    file_name: str = Field(description="Image file name in uploads/ or a path")


@tool(args_schema=ImageAnalyzeInput)
def analyze_image(file_name: str) -> str:
    """Describe a crop/plant image in English (part, symptoms, lesions). Returns JSON string.
    Paths: repo-relative path under PROJECT_ROOT, filename under UPLOAD_FOLDER,
    or absolute / ./ / ../ paths (see ``_resolve_image_path``).
    """
    file_name = (file_name or "").strip()
    if not file_name:
        return _err(400, "file_name is required")

    full_path = _resolve_image_path(file_name)
    if not os.path.exists(full_path):
        return _err(404, f"file not found: {full_path}")

    client, err = _require_openai_client("image")
    if err:
        return _err(500, err)
    prompt = get_image_analyze_prompt("en")
    try:
        with open(full_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        completion = client.chat.completions.create(
            model=os.getenv("IMAGE_MODEL", "qwen3-vl-plus"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        result = completion.choices[0].message.content
        return json.dumps({
            "code": 200,
            "file": file_name,
            "description": result,
        }, ensure_ascii=False)
    except Exception as e:
        return _err(500, str(e))


class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=3, ge=1, le=10, description="Max results")


@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 3) -> str:
    """Tavily web search; returns JSON."""
    if not _web_search_enabled():
        return _err(403, "web search disabled; set ENABLE_WEB_SEARCH=1 to enable")
    if not query.strip():
        return _err(400, "query is empty")
    res = _tavily_search(query, max_results)
    if "error" in res:
        return _err(500, res["error"])
    return json.dumps({
        "code": 200,
        "query": query,
        "results": res.get("results", [])
    }, ensure_ascii=False)


@tool(args_schema=SearchInput)
def search_and_summarize(query: str, max_results: int = 3) -> str:
    """Search and summarize top results (JSON)."""
    if not _web_search_enabled():
        return _err(403, "web search disabled; set ENABLE_WEB_SEARCH=1 to enable")
    if not query.strip():
        return _err(400, "query is empty")
    res = _tavily_search(query, max_results)
    if "error" in res:
        return _err(500, res["error"])
    results = res.get("results", [])
    summary_parts: List[str] = []
    for i, item in enumerate(results[:3], 1):
        title = item.get("title", "(no title)")
        content = item.get("content", "")
        url = item.get("url", "")
        snippet = (content[:160] + "...") if len(content) > 160 else content
        summary_parts.append(f"{i}. {title}\\n   {snippet}\\n   source: {url}")
    summary = "\\n\\n".join(summary_parts) if summary_parts else "No relevant results"
    return json.dumps({
        "code": 200,
        "query": query,
        "summary": summary,
    }, ensure_ascii=False)


class LiteratureQueryInput(BaseModel):
    keywords: str = Field(description="Keywords or question for literature search")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to return")
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity (0–1)",
    )


@tool(args_schema=LiteratureQueryInput)
def retrieve_literature(
    keywords: str,
    top_k: int = 3,
    similarity_threshold: float = 0.3,
) -> str:
    """Retrieve formatted text from the local literature vector store."""
    retriever = _get_literature_retriever()
    if retriever is None:
        return "Literature retrieval unavailable: index not initialized."
    return retriever.retrieve_formatted(
        query=keywords,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


class DiseaseInput(BaseModel):
    disease_name: str = Field(description="Disease name or keyword as stored in the KG")


@tool(args_schema=DiseaseInput)
def get_pesticides_by_disease(disease_name: str) -> str:
    """Fuzzy match pesticides by disease. JSON output."""
    if not disease_name.strip():
        return _err(400, "disease_name is empty")
    driver = _get_driver()
    if driver is None:
        return _err(503, "Neo4j unavailable")
    cypher = """
    MATCH (p:Pesticide)-[r:TREATS]->(d:Disease)
    WHERE d.name CONTAINS $disease
    RETURN p.name AS product_name,
           p.registration_number AS registration_number,
           p.formulation AS formulation,
           p.toxicity AS toxicity_class,
           p.active_ingredients_raw AS active_ingredients,
           r.crop AS applicable_crop,
           d.name AS disease_target,
           r.dosage AS recommended_dosage
    ORDER BY p.name
    LIMIT 10
    """
    try:
        queries = [disease_name.strip()]
        seen_ids: set = set()
        data: list = []
        tried: list = []
        with driver.session() as session:
            for q in queries:
                tried.append(q)
                rows = session.run(cypher, disease=q).data()
                for row in rows:
                    key = (row.get("product_name"), row.get("disease_target"))
                    if key not in seen_ids:
                        seen_ids.add(key)
                        data.append(row)
                if data:
                    break
        return json.dumps(
            {"code": 200, "count": len(data), "queried": tried, "data": data},
            ensure_ascii=False,
        )
    except Exception as e:
        return _err(500, str(e))


class CropDiseaseInput(BaseModel):
    crop_name: str = Field(description="Crop name (English or Chinese, e.g. tomato / 番茄)")
    disease_name: str = Field(description="Disease name as stored in the KG")


@tool(args_schema=CropDiseaseInput)
def get_pesticides_by_crop_and_disease(crop_name: str, disease_name: str) -> str:
    """Pesticides for a crop+disease pair. JSON."""
    if not crop_name.strip() or not disease_name.strip():
        return _err(400, "crop_name or disease_name is empty")
    driver = _get_driver()
    if driver is None:
        return _err(503, "Neo4j unavailable")
    cypher = """
    MATCH (p:Pesticide)-[r:TREATS]->(d:Disease)
    WHERE d.name CONTAINS $disease
      AND r.crop CONTAINS $crop
    RETURN p.name AS product_name,
           p.registration_number AS registration_number,
           p.formulation AS formulation,
           p.toxicity AS toxicity_class,
           p.active_ingredients_raw AS active_ingredients,
           r.crop AS applicable_crop,
           d.name AS disease_target,
           r.dosage AS recommended_dosage
    ORDER BY p.name
    LIMIT 10
    """
    try:
        disease_candidates = [disease_name.strip()]
        crop_candidates: List[str] = [crop_name.strip()]

        seen_ids: set = set()
        data: list = []
        tried: list = []
        with driver.session() as session:
            for c in crop_candidates:
                for d in disease_candidates:
                    tried.append({"crop": c, "disease": d})
                    rows = session.run(cypher, disease=d, crop=c).data()
                    for row in rows:
                        key = (
                            row.get("product_name"),
                            row.get("disease_target"),
                            row.get("applicable_crop"),
                        )
                        if key not in seen_ids:
                            seen_ids.add(key)
                            data.append(row)
                    if data:
                        break
                if data:
                    break
        return json.dumps(
            {"code": 200, "count": len(data), "queried": tried, "data": data},
            ensure_ascii=False,
        )
    except Exception as e:
        return _err(500, str(e))


class PesticideInput(BaseModel):
    pesticide_name: str = Field(description="Pesticide product name keyword (often Chinese in KG)")


@tool(args_schema=PesticideInput)
def get_crop_and_disease_by_pesticide(pesticide_name: str) -> str:
    """Reverse lookup: pesticide → crops and diseases. JSON. Partial match."""
    if not pesticide_name.strip():
        return _err(400, "pesticide_name is empty")
    driver = _get_driver()
    if driver is None:
        return _err(503, "Neo4j unavailable")
    try:
        cypher = """
        MATCH (p:Pesticide)-[r:TREATS]->(d:Disease)
        WHERE p.name CONTAINS $pesticide
        RETURN p.name AS product_name,
               p.registration_number AS registration_number,
               p.formulation AS formulation,
               p.toxicity AS toxicity_class,
               p.active_ingredients_raw AS active_ingredients,
               r.crop AS applicable_crop,
               d.name AS disease_target,
               r.dosage AS recommended_dosage
        ORDER BY r.crop, d.name
        LIMIT 10
        """
        with driver.session() as session:
            data = session.run(cypher, pesticide=pesticide_name.strip()).data()
        return json.dumps({"code": 200, "count": len(data), "data": data}, ensure_ascii=False)
    except Exception as e:
        return _err(500, str(e))


@tool()
def check_kg_status() -> str:
    """Neo4j driver health and coarse graph counts (JSON)."""
    driver = _get_driver()
    connected = driver is not None
    stats = {}
    if connected:
        try:
            with driver.session() as session:
                stats["crop_count"] = session.run("MATCH (c:Crop) RETURN count(c) AS n").single()["n"]
                stats["disease_count"] = session.run("MATCH (d:Disease) RETURN count(d) AS n").single()["n"]
                stats["pesticide_count"] = session.run("MATCH (p:Pesticide) RETURN count(p) AS n").single()["n"]
                stats["treats_count"] = session.run(
                    "MATCH (:Pesticide)-[r:TREATS]->(:Disease) RETURN count(r) AS n"
                ).single()["n"]
        except Exception as e:
            return _err(500, str(e))
    return json.dumps({"code": 200, "connected": connected, "stats": stats}, ensure_ascii=False)


class CropInput(BaseModel):
    crop_name: str = Field(description="Crop name (English or Chinese, e.g. tomato / 番茄)")


@tool(args_schema=CropInput)
def list_diseases_by_crop(crop_name: str) -> str:
    """List diseases in KG for a crop with pesticide counts. Use when crop+disease query returns empty."""
    if not crop_name.strip():
        return _err(400, "crop_name is empty")
    driver = _get_driver()
    if driver is None:
        return _err(503, "Neo4j unavailable")
    crop_query = crop_name.strip()
    cypher = """
    MATCH (p:Pesticide)-[r:TREATS]->(d:Disease)
    WHERE r.crop CONTAINS $crop
    RETURN d.name AS disease_name,
           count(DISTINCT p) AS pesticide_count
    ORDER BY pesticide_count DESC, disease_name
    LIMIT 30
    """
    try:
        with driver.session() as session:
            data = session.run(cypher, crop=crop_query).data()
        return json.dumps({"code": 200, "count": len(data), "data": data}, ensure_ascii=False)
    except Exception as e:
        return _err(500, str(e))


class PesticideDetailInput(BaseModel):
    pesticide_name: str = Field(description="Pesticide name (often Chinese; partial match OK)")


@tool(args_schema=PesticideDetailInput)
def get_pesticide_detail(pesticide_name: str) -> str:
    """Dosage / PHI / toxicity / precautions for a pesticide product (JSON). Chinese text blocks are truncated to 400 chars."""
    if not pesticide_name.strip():
        return _err(400, "pesticide_name is empty")
    driver = _get_driver()
    if driver is None:
        return _err(503, "Neo4j unavailable")
    cypher = """
    MATCH (p:Pesticide)
    WHERE p.name CONTAINS $pesticide
    RETURN p.name AS product_name,
           p.registration_number AS registration_number,
           p.formulation AS formulation,
           p.toxicity AS toxicity_class,
           p.active_ingredients_raw AS active_ingredients,
           p.instructions AS usage_instructions,
           p.precautions AS precautions,
           p.emergency_measures AS emergency_measures
    LIMIT 5
    """
    try:
        with driver.session() as session:
            rows = session.run(cypher, pesticide=pesticide_name.strip()).data()
        for row in rows:
            for k in ("usage_instructions", "precautions", "emergency_measures"):
                row[k] = _truncate(row.get(k), 400)
        return json.dumps({"code": 200, "count": len(rows), "data": rows}, ensure_ascii=False)
    except Exception as e:
        return _err(500, str(e))


class DiseaseClassifyInput(BaseModel):
    file_name: str = Field(description="Image file name or path")
    topk: int = Field(default=5, ge=1, le=10, description="Top-K predictions")


@tool(args_schema=DiseaseClassifyInput)
def classify_disease(file_name: str, topk: int = 5) -> str:
    """Disease classification from leaf image; returns JSON."""
    if not _check_plant_api_health():
        return _err(503, "plant API is not running")
    image_path = _resolve_image_path(file_name)
    if not os.path.exists(image_path):
        return _err(404, f"file not found: {image_path}")
    res = _post_json("/disease/classify", {"image_path": image_path, "topk": topk})
    if res.get("code") != 200:
        return _err(res.get("code", 500), str(res.get("error", "plant API error")))
    return json.dumps(res.get("data", {}), ensure_ascii=False)


class SegmentationInput(BaseModel):
    file_name: str = Field(description="Image file name or path")
    save_masks: bool = Field(default=True, description="Whether to save segmentation masks")


@tool(args_schema=SegmentationInput)
def segment_disease(file_name: str, save_masks: bool = True) -> str:
    """Lesion segmentation and severity; returns JSON."""
    if not _check_plant_api_health():
        return _err(503, "plant API is not running")
    image_path = _resolve_image_path(file_name)
    if not os.path.exists(image_path):
        return _err(404, f"file not found: {image_path}")
    res = _post_json("/segmentation/severity", {"image_path": image_path, "save_masks": save_masks})
    if res.get("code") != 200:
        return _err(res.get("code", 500), str(res.get("error", "plant API error")))
    return json.dumps(res.get("data", {}), ensure_ascii=False)


@tool()
def check_plant_api_status() -> str:
    """Health check for the local plant-disease Flask API (JSON)."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200:
            return json.dumps({"code": 200, "data": r.json()}, ensure_ascii=False)
        return _err(r.status_code, r.text)
    except Exception as e:
        return _err(500, str(e))


ALL_TOOLS = [
    analyze_image,
    retrieve_literature,
    get_pesticides_by_disease,
    get_pesticides_by_crop_and_disease,
    get_crop_and_disease_by_pesticide,
    list_diseases_by_crop,
    get_pesticide_detail,
    check_kg_status,
    classify_disease,
    segment_disease,
    check_plant_api_status,
]

__all__ = ["ALL_TOOLS"]

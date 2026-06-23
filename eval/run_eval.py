from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from eval.diagnosis_eval import aggregate_diagnosis_results, evaluate_diagnosis
from eval.prescription_eval import aggregate_prescription_results, evaluate_prescription
from eval.route_eval import aggregate_route_results, evaluate_route
from eval.tool_eval import aggregate_tool_results, evaluate_tools

RESULTS_DIR = PROJECT_ROOT / "eval_results"

SYSTEMS = {
    "ceres_full": "CeresAgents (full system: RAG + KG + CV)",
    "ceres_no_rag": "CeresAgents (ablation: no RAG)",
    "ceres_no_kg": "CeresAgents (ablation: no KG)",
    "ceres_no_cv": "CeresAgents (ablation: no CV)",
    "ceres_single_llm": "CeresAgents (ablation: single LLM, no multi-agent)",
    "direct_llm": "Direct LLM baseline (single-call, OpenAI-compatible API)",
}

CERES_SYSTEMS = {
    "ceres_full",
    "ceres_no_rag",
    "ceres_no_kg",
    "ceres_no_cv",
    "ceres_single_llm",
}
SUPPORTS_ROUTING_EVAL = CERES_SYSTEMS

_ABLATION_DISABLED_TOOLS: Dict[str, List[str]] = {
    "ceres_no_rag": ["retrieve_literature"],
    "ceres_no_kg": [
        "get_pesticides_by_disease",
        "get_pesticides_by_crop_and_disease",
        "get_crop_and_disease_by_pesticide",
        "list_diseases_by_crop",
        "get_pesticide_detail",
    ],
    "ceres_no_cv": ["classify_disease", "segment_disease", "analyze_image"],
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {i}: {exc}")
    return records


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_judge_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("MODEL_API_KEY")
    if not api_key:
        raise ValueError("Set JUDGE_API_KEY (or MODEL_API_KEY) for judge LLM calls")

    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("MODEL_BASE_URL")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    model = os.getenv("JUDGE_MODEL", "deepseek-chat")
    return OpenAI(**kwargs), model


def _build_direct_llm_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("DIRECT_LLM_API_KEY") or os.getenv("MODEL_API_KEY")
    if not api_key:
        raise ValueError("Set DIRECT_LLM_API_KEY (or MODEL_API_KEY) for direct_llm baseline calls")

    base_url = os.getenv("DIRECT_LLM_BASE_URL") or os.getenv("MODEL_BASE_URL")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    model = os.getenv("DIRECT_LLM_MODEL", "gpt-5")
    return OpenAI(**kwargs), model


def _resolve_context(provided_context: Any) -> str:
    if not provided_context:
        return ""
    if isinstance(provided_context, str):
        return provided_context.strip()
    if isinstance(provided_context, dict):
        parts = [
            f"{k}: {v}"
            for k, v in provided_context.items()
            if v and str(v).strip() and str(v).strip().lower() != "null"
        ]
        return "; ".join(parts)
    return str(provided_context).strip()


def _load_image_b64(source_image: str) -> Optional[str]:
    if not source_image:
        return None

    path_text = str(source_image).strip()
    if not path_text:
        return None

    if os.path.isabs(path_text) or path_text.startswith("./") or path_text.startswith("../"):
        image_path = Path(path_text)
    else:
        repo_rel = PROJECT_ROOT / path_text
        uploads_rel = PROJECT_ROOT / "uploads" / path_text
        if repo_rel.is_file():
            image_path = repo_rel
        elif uploads_rel.is_file():
            image_path = uploads_rel
        else:
            image_path = repo_rel

    if not image_path.exists():
        return None

    try:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def _init_ceres_agent(system_name: str) -> Any:
    from ceres_agents import CeresAgentsGraph

    disabled = set(_ABLATION_DISABLED_TOOLS.get(system_name, []))
    return CeresAgentsGraph(verbose=False, disabled_tools=disabled)


def _parse_response_from_state(agent: Any, final_response: str) -> Dict[str, Any]:
    structured: Dict[str, Any] = getattr(agent, "last_structured_log", None) or {}
    last_state: Dict[str, Any] = agent.get_last_state() if hasattr(agent, "get_last_state") else {}

    route = structured.get("route") or last_state.get("route_decision", "") or "unknown"
    route_lower = str(route).lower()
    if route_lower in {"fast", "simple"}:
        route = "fast"
    elif route_lower in {"expert", "standard", "slow", "complex"}:
        route = "expert"

    activated_agents: List[str] = structured.get("experts_used") or last_state.get("active_experts") or []

    tool_calls: List[Dict[str, Any]] = []
    run_id = last_state.get("run_id")
    for entry in last_state.get("reasoning_log", []):
        if not isinstance(entry, dict):
            continue
        if run_id and entry.get("run_id") and entry.get("run_id") != run_id:
            continue
        agent_name = entry.get("agent", "")
        for tool_result in entry.get("tool_results") or []:
            if not isinstance(tool_result, dict):
                continue
            tool_calls.append(
                {
                    "agent": agent_name,
                    "tool": tool_result.get("tool_name", ""),
                    "args": tool_result.get("args", {}),
                }
            )

    return {
        "final_response": final_response,
        "route": route,
        "activated_agents": activated_agents,
        "tool_calls": tool_calls,
        "structured_log": structured,
    }


def _call_ceres_direct(
    case: Dict[str, Any],
    agent: Any,
    system_name: str,
    response_sink: Optional[Any] = None,
) -> Dict[str, Any]:
    user_input = case.get("user_input", {})
    query = user_input.get("query", "")
    context = user_input.get("provided_context")
    ctx_str = _resolve_context(context)
    if ctx_str:
        query = f"{query}\n\nContext: {ctx_str}"

    if system_name == "ceres_single_llm" and not query.startswith("[EVAL:force_fast]"):
        query = f"[EVAL:force_fast] {query}"

    try:
        final_response: str = agent.diagnose(
            query=query,
            image_path=case.get("source_image"),
            history=[],
            debug=False,
        )
        response = _parse_response_from_state(agent, final_response)
    except Exception as exc:
        response = {
            "error": str(exc),
            "final_response": "",
            "route": "unknown",
            "activated_agents": [],
            "tool_calls": [],
            "structured_log": {},
        }

    if response_sink is not None:
        record = {"case_id": case.get("id", ""), "system": system_name, **response}
        response_sink.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        response_sink.flush()

    return response


DIRECT_SYSTEM_PROMPT = """You are an expert agricultural advisor.

Answer the user's crop-disease question directly and concisely.
If an image or symptom description is provided, infer the most likely disease and explain the evidence.
If treatment is requested, provide practical and safety-aware recommendations.
Do not invent unavailable evidence or claim to have used external tools.
"""


def _call_direct_llm(
    case: Dict[str, Any],
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    user_input = case.get("user_input", {})
    query = user_input.get("query", "")
    context = user_input.get("provided_context")
    user_text = query
    ctx_str = _resolve_context(context)
    if ctx_str:
        user_text = f"{query}\n\nContext: {ctx_str}"

    source_image = case.get("source_image") or ""
    img_b64 = _load_image_b64(source_image) if source_image else None
    if img_b64:
        ext = Path(source_image).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=120,
        )
        content = (resp.choices[0].message.content or "").strip()
        return {
            "final_response": content,
            "route": "n/a",
            "activated_agents": [],
            "tool_calls": [],
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "final_response": "",
            "route": "n/a",
            "activated_agents": [],
            "tool_calls": [],
        }


def evaluate_case(
    case: Dict[str, Any],
    response: Dict[str, Any],
    system_name: str,
    judge_client: OpenAI,
    judge_model: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "case_id": case.get("id", ""),
        "track": case.get("metadata", {}).get("track", ""),
        "system_name": system_name,
        "has_error": bool(response.get("error")),
    }

    if system_name in SUPPORTS_ROUTING_EVAL:
        result["route_eval"] = evaluate_route(case, response)
        result["tool_eval"] = evaluate_tools(case, response)

    diag = evaluate_diagnosis(case, response, judge_client, judge_model)
    if diag is not None:
        result["diagnosis_eval"] = diag

    rx = evaluate_prescription(case, response, judge_client, judge_model)
    if rx is not None:
        result["prescription_eval"] = rx

    return result


def compute_aggregate_metrics(results: List[Dict[str, Any]], system_name: str) -> Dict[str, Any]:
    route_results = [r["route_eval"] for r in results if "route_eval" in r]
    tool_results = [r["tool_eval"] for r in results if "tool_eval" in r]
    diag_results = [r["diagnosis_eval"] for r in results if "diagnosis_eval" in r]
    rx_results = [r["prescription_eval"] for r in results if "prescription_eval" in r]

    metrics: Dict[str, Any] = {
        "system": system_name,
        "description": SYSTEMS.get(system_name, system_name),
        "total_cases_evaluated": len(results),
        "error_cases": sum(1 for r in results if r.get("has_error")),
    }

    if route_results:
        metrics["route_agent"] = aggregate_route_results(route_results)
    if tool_results:
        metrics["tool"] = aggregate_tool_results(tool_results)
    if diag_results:
        metrics["diagnosis"] = aggregate_diagnosis_results(diag_results)
    if rx_results:
        metrics["prescription"] = aggregate_prescription_results(rx_results)

    return metrics


def run_system_eval(
    system_name: str,
    cases: List[Dict[str, Any]],
    judge_client: Optional[OpenAI],
    judge_model: str,
    responses_path: Optional[Path],
    sleep_s: float,
    resume: bool,
    max_cases: int = 0,
    generate_only: bool = False,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {system_name}")
    print(f"{'=' * 60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{system_name}_eval.jsonl"
    metrics_path = RESULTS_DIR / f"{system_name}_metrics.json"
    responses_out_path = RESULTS_DIR / f"{system_name}_responses.jsonl"

    pre_responses: Dict[str, Dict[str, Any]] = {}
    if responses_path and responses_path.exists():
        for rec in load_jsonl(responses_path):
            case_id = rec.get("case_id", rec.get("id", ""))
            if case_id:
                pre_responses[case_id] = rec
        print(f"  [pre-loaded] {len(pre_responses)} responses from {responses_path}")

    done_ids: set[str] = set()
    if resume and results_path.exists():
        for rec in load_jsonl(results_path):
            case_id = rec.get("case_id", "")
            if case_id:
                done_ids.add(case_id)
        print(f"  [resume] Skipping {len(done_ids)} already-evaluated cases")

    needs_live_calls = any(
        case.get("id", "") not in pre_responses
        for case in cases
        if case.get("id", "") not in done_ids
    )
    ceres_agent: Optional[Any] = None
    direct_llm_client: Optional[OpenAI] = None
    direct_llm_model: str = ""
    if needs_live_calls:
        if system_name in CERES_SYSTEMS:
            print(f"  [init] Loading CeresAgentsGraph for {system_name}...")
            ceres_agent = _init_ceres_agent(system_name)
            print("  [init] CeresAgentsGraph ready")
        elif system_name == "direct_llm":
            direct_llm_client, direct_llm_model = _build_direct_llm_client()
            print(f"  [init] Direct LLM client ready (model: {direct_llm_model})")
    else:
        print("  [init] All cases covered by pre-loaded responses; skipping agent init.")

    all_results: List[Dict[str, Any]] = []
    scores: Dict[str, List[float]] = {"diag": [], "route_ok": [], "rx": []}

    eval_write_mode = "a" if resume else "w"
    resp_write_mode = "a" if resume else "w"

    if responses_path and responses_path.resolve() == responses_out_path.resolve():
        responses_out_path = responses_out_path.with_name(
            responses_out_path.stem + "_new" + responses_out_path.suffix
        )
        print(f"  [WARN] Redirecting new responses to: {responses_out_path}")

    pending = [case for case in cases if case.get("id", "") not in done_ids]
    if max_cases > 0:
        pending = pending[:max_cases]
        print(f"  [--max-cases] limiting to {max_cases} case(s)")

    bar = tqdm(total=len(pending), desc=f"[{system_name}]", unit="case", dynamic_ncols=True)

    with (
        results_path.open(eval_write_mode, encoding="utf-8") as out,
        responses_out_path.open(resp_write_mode, encoding="utf-8") as resp_out,
    ):
        for case in pending:
            case_id = case.get("id", "")
            track = case.get("metadata", {}).get("track", "?")

            bar.set_postfix_str(f"running [{track}] {case_id[:40]}")
            if case_id in pre_responses:
                response = pre_responses[case_id]
            elif system_name in CERES_SYSTEMS:
                response = _call_ceres_direct(case, ceres_agent, system_name, response_sink=resp_out)
            else:
                response = _call_direct_llm(case, direct_llm_client, direct_llm_model)
                record = {"case_id": case_id, "system": system_name, **response}
                resp_out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                resp_out.flush()

            if generate_only:
                bar.set_postfix_str(f"saved [{track}] {case_id[:40]}")
            else:
                try:
                    result = evaluate_case(case, response, system_name, judge_client, judge_model)
                    all_results.append(result)
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()

                    diag = result.get("diagnosis_eval", {}).get("total_score")
                    if diag is not None:
                        scores["diag"].append(diag)
                    rx_score = result.get("prescription_eval", {}).get("total_score")
                    if rx_score is not None:
                        scores["rx"].append(rx_score)
                    route_ok = result.get("route_eval", {}).get("route_correct")
                    if route_ok is not None:
                        scores["route_ok"].append(float(route_ok))

                    parts: List[str] = []
                    if scores["diag"]:
                        parts.append(f"Diag={sum(scores['diag']) / len(scores['diag']):.1f}")
                    if scores["rx"]:
                        parts.append(f"RX={sum(scores['rx']) / len(scores['rx']):.1f}")
                    if scores["route_ok"]:
                        parts.append(
                            f"Route={sum(scores['route_ok']) / len(scores['route_ok']) * 100:.0f}%"
                        )
                    bar.set_postfix_str("  ".join(parts))
                except Exception as exc:
                    bar.write(f"  [ERROR] {case_id}: {exc}")
                    error_rec = {"case_id": case_id, "system_name": system_name, "error": str(exc)}
                    all_results.append(error_rec)
                    out.write(json.dumps(error_rec, ensure_ascii=False) + "\n")
                    out.flush()

            bar.update(1)
            time.sleep(sleep_s)

    bar.close()

    if generate_only:
        print(f"\n  [generate-only] Responses saved: {responses_out_path}")
        return {
            "system": system_name,
            "generate_only": True,
            "responses_saved": str(responses_out_path),
        }

    metrics = compute_aggregate_metrics(all_results, system_name)
    save_json(metrics, metrics_path)

    print(f"\n  Results:   {results_path}")
    print(f"  Responses: {responses_out_path}")
    print(f"  Metrics:   {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public CeresAgents evaluation")
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="Path to the evaluation case JSONL file.",
    )
    parser.add_argument(
        "--system",
        choices=list(SYSTEMS.keys()),
        default="ceres_full",
        help="Which CeresAgents system variant to evaluate.",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=None,
        help="Pre-generated responses JSONL to reuse instead of live system calls.",
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate responses and skip judge-based evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cases.exists():
        print(f"[ERROR] Cases file not found: {args.cases}")
        print("  The public repository does not bundle the full paper benchmark by default.")
        print("  Provide a case file with --cases <path-to-jsonl> if you want to run evaluation.")
        sys.exit(1)

    cases = load_jsonl(args.cases)
    print(f"[run_eval] Loaded {len(cases)} cases from {args.cases}")

    if args.generate_only:
        judge_client, judge_model = None, ""
        print("[run_eval] --generate-only: skipping judge LLM init.")
    else:
        judge_client, judge_model = _build_judge_client()

    metrics = run_system_eval(
        system_name=args.system,
        cases=cases,
        judge_client=judge_client,
        judge_model=judge_model,
        responses_path=args.responses,
        sleep_s=args.sleep,
        resume=args.resume,
        max_cases=args.max_cases,
        generate_only=args.generate_only,
    )

    if args.generate_only:
        print("\n[run_eval] Generate-only run complete. No metrics computed.")
    else:
        combined_path = RESULTS_DIR / f"{args.system}_metrics.json"
        print(f"\n[run_eval] Metrics saved: {combined_path}")


if __name__ == "__main__":
    main()

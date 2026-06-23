from __future__ import annotations

from typing import Any, Dict, List, Tuple

# 系统只有 fast 和 expert 两条路径，直接比较即可
_VALID_ROUTES = {"fast", "expert"}

_EXPERT_AGENTS = {"pathologist", "physiologist", "chemist"}


def _normalize_route(route: str) -> str:
    r = (route or "").strip().lower()
    if r == "fast":
        return "fast"
    return "expert"  # expert / slow / standard / conflict / unknown 都视为 expert


def _extract_route(response: Dict[str, Any]) -> str:
    if "route" in response:
        return str(response["route"]).lower()
    meta = response.get("metadata", {}) or {}
    if "route" in meta:
        return str(meta["route"]).lower()
    # 推断降级：有专家 agent 激活 → expert
    for a in response.get("activated_agents", []) or []:
        if str(a).lower() in _EXPERT_AGENTS:
            return "expert"
    return "fast"


def _extract_agents(response: Dict[str, Any]) -> List[str]:
    agents = list(response.get("activated_agents", []) or [])
    if not agents and _extract_route(response) == "fast":
        return ["Generalist"]
    return [str(a) for a in agents]


def _set_f1(predicted: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
    pred_set = {a.lower() for a in predicted}
    gt_set   = {a.lower() for a in ground_truth}

    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_route(case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    gt = case.get("ground_truth_evaluation", {})
    expected_raw    = gt.get("expected_route", "")
    expected_route  = _normalize_route(expected_raw)
    must_invoke     = list(gt.get("must_invoke_agents", []) or [])

    predicted_route  = _normalize_route(_extract_route(response))
    activated_agents = _extract_agents(response)

    route_correct = (predicted_route == expected_route)
    precision, recall, f1 = _set_f1(activated_agents, must_invoke)

    expected_fast = (expected_route == "fast") or (not must_invoke)
    fired_expert  = any(a.lower() in _EXPERT_AGENTS for a in activated_agents)
    over_expert_activation = expected_fast and fired_expert

    return {
        "case_id":                  case.get("id", ""),
        "track":                    case.get("metadata", {}).get("track", ""),
        "expected_route_raw":       expected_raw,
        "expected_route":           expected_route,
        "predicted_route":          predicted_route,
        "route_correct":            route_correct,
        "must_invoke_agents":       must_invoke,
        "activated_agents":         activated_agents,
        "agent_precision":          round(precision, 4),
        "agent_recall":             round(recall, 4),
        "agent_f1":                 round(f1, 4),
        "over_expert_activation":   over_expert_activation,
    }


def aggregate_route_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    total = len(results)
    route_correct       = sum(1 for r in results if r.get("route_correct"))
    avg_agent_f1        = sum(r.get("agent_f1", 0.0) for r in results) / total
    avg_agent_precision = sum(r.get("agent_precision", 0.0) for r in results) / total
    avg_agent_recall    = sum(r.get("agent_recall", 0.0) for r in results) / total

    fast_cases = [r for r in results if r.get("expected_route_project") == "fast"
                  or not r.get("must_invoke_agents")]
    over_expert_rate = (
        sum(1 for r in fast_cases if r.get("over_expert_activation")) / len(fast_cases)
        if fast_cases else 0.0
    )

    track_stats: Dict[str, Dict[str, float]] = {}
    for r in results:
        t = r.get("track", "unknown")
        s = track_stats.setdefault(t, {"total": 0, "route_correct": 0, "agent_f1_sum": 0.0})
        s["total"] += 1
        if r.get("route_correct"):
            s["route_correct"] += 1
        s["agent_f1_sum"] += r.get("agent_f1", 0.0)
    per_track = {
        t: {
            "route_accuracy": round(v["route_correct"] / v["total"], 4),
            "avg_agent_f1":   round(v["agent_f1_sum"] / v["total"], 4),
            "n":              int(v["total"]),
        }
        for t, v in track_stats.items()
    }

    return {
        "total_cases":                total,
        "route_accuracy":             round(route_correct / total, 4),
        "avg_agent_precision":        round(avg_agent_precision, 4),
        "avg_agent_recall":           round(avg_agent_recall, 4),
        "avg_agent_f1":               round(avg_agent_f1, 4),
        "over_expert_activation_rate": round(over_expert_rate, 4),
        "per_track":                  per_track,
    }
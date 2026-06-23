"""
Evaluator 3 — Tool Usage Accuracy.

Computes Precision / Recall / F1 between the tools actually called by the
system and the `must_use_tools` ground truth in each eval case.  Applies only
to systems that expose a tool-call log.

Alongside the F1 numbers this evaluator reports `unnecessary_tool_rate`,
the fraction of cases where the ground truth expected no specific tool but
the system chose to call one anyway.  This is treated as a *process*
observation, not a correctness error — the primary F1 stays lenient for
such cases (so precautionary tool use isn't double-punished).

F1 edge-case semantics (kept consistent with route_eval.py):
  - empty GT + empty pred                   -> F1 = 1.0
  - empty GT + non-empty pred               -> F1 = 1.0 (captured by unnecessary_tool_rate)
  - non-empty GT + empty pred               -> F1 = 0.0
  - otherwise                               -> standard set-based F1
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# Canonical name normalization for tool aliases.  Keeps the legacy map
# intact so previously-written cases still match.
_TOOL_ALIASES: Dict[str, str] = {
    "disease classification":   "classify_disease",
    "disease_classification":   "classify_disease",
    "disease classify":         "classify_disease",
    "segment disease":          "segment_disease",
    "image analysis":           "analyze_image",
    "analyze image":            "analyze_image",
    "literature retrieval":     "retrieve_literature",
    "retrieve literature":      "retrieve_literature",
    "knowledge graph query":    "get_pesticides_by_crop_and_disease",
    "kg query":                 "get_pesticides_by_crop_and_disease",
    "get pesticides":           "get_pesticides_by_crop_and_disease",
    "weather query":            "retrieve_literature",
}


def _normalize(name: str) -> str:
    cleaned = (name or "").strip().lower()
    return _TOOL_ALIASES.get(cleaned, cleaned)


def _extract_tools(response: Dict[str, Any]) -> List[str]:
    raw: List[str] = []
    for tc in response.get("tool_calls", []) or []:
        if isinstance(tc, dict):
            raw.append(tc.get("tool") or tc.get("name") or "")
        elif isinstance(tc, str):
            raw.append(tc)
    for t in response.get("tools_used", []) or []:
        if isinstance(t, str):
            raw.append(t)
        elif isinstance(t, dict):
            raw.append(t.get("tool") or t.get("name") or "")

    seen: set[str] = set()
    result: List[str] = []
    for t in raw:
        n = _normalize(t)
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _set_f1(predicted: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
    pred_set = set(predicted)
    gt_set   = set(ground_truth)

    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gt_set:
        # Over-calls tracked separately via unnecessary_tool_rate.
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_tools(case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    gt = case.get("ground_truth_evaluation", {})
    must_use     = [_normalize(t) for t in (gt.get("must_use_tools", []) or [])]
    tools_called = _extract_tools(response)

    precision, recall, f1 = _set_f1(tools_called, must_use)

    extra_tools = sorted(set(tools_called) - set(must_use))
    missing_tools = sorted(set(must_use) - set(tools_called))

    return {
        "case_id":          case.get("id", ""),
        "track":            case.get("metadata", {}).get("track", ""),
        "must_use_tools":   must_use,
        "tools_called":     tools_called,
        "tool_precision":   round(precision, 4),
        "tool_recall":      round(recall, 4),
        "tool_f1":          round(f1, 4),
        "missing_tools":    missing_tools,
        "extra_tools":      extra_tools,
        # Process observation: the ground truth expected no specific tools
        # but the system called at least one anyway.
        "unnecessary_tool_call": (not must_use) and bool(tools_called),
    }


def aggregate_tool_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    total = len(results)
    avg_f1        = sum(r.get("tool_f1", 0.0) for r in results) / total
    avg_precision = sum(r.get("tool_precision", 0.0) for r in results) / total
    avg_recall    = sum(r.get("tool_recall", 0.0) for r in results) / total

    # Unnecessary-tool rate on the subset where GT expected no tools.
    no_gt_cases = [r for r in results if not r.get("must_use_tools")]
    unnecessary_rate = (
        sum(1 for r in no_gt_cases if r.get("unnecessary_tool_call")) / len(no_gt_cases)
        if no_gt_cases else 0.0
    )

    # Most frequently missed tools (helps spot CeresAgents configuration issues).
    missing_counts: Dict[str, int] = {}
    for r in results:
        for t in r.get("missing_tools", []):
            missing_counts[t] = missing_counts.get(t, 0) + 1

    # Per-track breakdown.
    track_stats: Dict[str, Dict[str, float]] = {}
    for r in results:
        t = r.get("track", "unknown")
        s = track_stats.setdefault(t, {"total": 0, "f1_sum": 0.0})
        s["total"] += 1
        s["f1_sum"] += r.get("tool_f1", 0.0)
    per_track = {
        t: {"avg_tool_f1": round(v["f1_sum"] / v["total"], 4), "n": int(v["total"])}
        for t, v in track_stats.items()
    }

    return {
        "total_cases":            total,
        "avg_tool_precision":     round(avg_precision, 4),
        "avg_tool_recall":        round(avg_recall, 4),
        "avg_tool_f1":            round(avg_f1, 4),
        "unnecessary_tool_rate":  round(unnecessary_rate, 4),
        "most_missed_tools":      sorted(missing_counts.items(), key=lambda x: -x[1])[:10],
        "per_track":              per_track,
    }

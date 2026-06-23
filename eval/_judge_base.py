"""
Shared infrastructure for all LLM-as-judge evaluators.

This module exists because the four judge evaluators (diagnosis / prescription /
safety / output_quality) used to each carry their own copy of:
  - JSON parsing
  - retry/timeout loop
  - response field extraction
  - tool-call summarisation

They also disagreed on WHAT counts as "tool evidence" and WHETHER the judge
could see tool calls at all.  The resulting divergence made it easy for
evaluators to silently reward text-only LLM baselines for evidence they never
actually produced.

All evaluators now use the helpers below so that:
  1. There is one canonical definition of tool-evidence categories (cv / kg / rag).
  2. Every judge prompt is fed the same structured evidence block with the
     same "No evidence available — direct LLM baseline" marker when absent.
  3. Hard caps ("no KG call -> accuracy <= 32") are applied by evaluator code
     against `has_evidence()`, never left to the judge's discretion.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tool-evidence taxonomy
# ---------------------------------------------------------------------------
#
# Each category lists substrings that MUST appear in the recorded tool name
# for that tool-call to count as evidence for that category.
#
# Matching is substring-based (lower-cased) so that tool aliases and ablation
# variants all collapse to the same category.
TOOL_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "cv": (
        "analyze_image",
        "classify_disease",
        "segment_disease",
        "detect_",
        "vision_",
    ),
    "kg": (
        "get_pesticides",
        "get_pesticide_detail",
        "list_diseases_by_crop",
        "get_crop_and_disease_by_pesticide",
        "knowledge_graph",
        "pesticide_kg",
    ),
    "rag": (
        "retrieve_literature",
        "search_literature",
        "literature_",
    ),
}

# Human-readable labels for the prompt block.
_CATEGORY_LABELS = {
    "cv":  "Computer-vision tools (image analysis / disease classification / segmentation)",
    "kg":  "Pesticide knowledge-graph tools",
    "rag": "Literature retrieval tools",
}


# ---------------------------------------------------------------------------
# Tool evidence extraction
# ---------------------------------------------------------------------------

def _iter_tool_calls(response: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield each tool-call dict from the response log."""
    for tc in response.get("tool_calls", []) or []:
        if isinstance(tc, dict):
            yield tc
        elif isinstance(tc, str):
            yield {"tool": tc, "args": {}, "output": None}


def _classify(tool_name: str) -> Optional[str]:
    """Return the category key for a tool name, or None if uncategorised."""
    lname = (tool_name or "").strip().lower()
    if not lname:
        return None
    for cat, patterns in TOOL_CATEGORIES.items():
        for p in patterns:
            if p in lname:
                return cat
    return None


def collect_tool_evidence(response: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Bucket every tool-call in the response into {cv, kg, rag, other}.

    Returns a dict with keys "cv" | "kg" | "rag" | "other".  Each value is a
    list of `{"tool": name, "args": {...}, "output_preview": str}` entries.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {"cv": [], "kg": [], "rag": [], "other": []}
    for tc in _iter_tool_calls(response):
        name = str(tc.get("tool") or tc.get("name") or "")
        cat = _classify(name) or "other"
        out = tc.get("output")
        if out is None:
            preview = ""
        else:
            preview = json.dumps(out, ensure_ascii=False) if not isinstance(out, str) else out
            if len(preview) > 240:
                preview = preview[:240] + "..."
        buckets[cat].append({
            "tool": name,
            "args": tc.get("args") or {},
            "output_preview": preview,
        })
    return buckets


def has_evidence(evidence: Dict[str, List[Dict[str, Any]]], *categories: str) -> bool:
    """Return True if the response contains at least one tool call in ANY of the given categories."""
    return any(evidence.get(c) for c in categories)


def format_tool_evidence(
    evidence: Dict[str, List[Dict[str, Any]]],
    scopes: Iterable[str] = ("cv", "kg", "rag"),
) -> str:
    """
    Render a compact, judge-friendly evidence block.

    For each requested category, either list the recorded calls or explicitly
    state "No evidence" so the judge cannot infer missing evidence as merely
    absent from the prompt.
    """
    lines: List[str] = []
    for cat in scopes:
        label = _CATEGORY_LABELS.get(cat, cat)
        calls = evidence.get(cat, [])
        if not calls:
            lines.append(f"- {label}: No evidence recorded (this response did NOT call any tool in this category).")
            continue
        lines.append(f"- {label}: {len(calls)} call(s) recorded.")
        for c in calls[:4]:
            args_str = json.dumps(c["args"], ensure_ascii=False) if c["args"] else "{}"
            out = c["output_preview"]
            if out:
                lines.append(f"    * {c['tool']}  args={args_str}  output={out}")
            else:
                lines.append(f"    * {c['tool']}  args={args_str}")
        if len(calls) > 4:
            lines.append(f"    * ... plus {len(calls) - 4} more call(s)")
    return "\n".join(lines) if lines else "No tool-call log available."


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------

def extract_final_response(response: Dict[str, Any]) -> str:
    """Return the system's final user-facing text, regardless of key name."""
    if not response:
        return ""
    for key in ("final_response", "response", "answer", "output", "synthesis"):
        v = response.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


# ---------------------------------------------------------------------------
# Case metadata helpers
# ---------------------------------------------------------------------------

def case_has_image(case: Dict[str, Any]) -> bool:
    """Return True if the eval case was associated with an input image."""
    if case.get("source_image"):
        return True
    if case.get("metadata", {}).get("has_image"):
        return True
    ui = case.get("user_input", {}) or {}
    if ui.get("image_path"):
        return True
    return False


# ---------------------------------------------------------------------------
# Judge call + JSON parsing
# ---------------------------------------------------------------------------

def parse_judge_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM reply, tolerant of markdown fences."""
    if not text:
        return {"__parse_error__": "empty text"}
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"__parse_error__": f"parse failed: {text[:200]}"}


def call_judge(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 1200,
    timeout: int = 60,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Call the judge LLM and return a parsed dict.

    On any failure (network / empty content / malformed JSON) the result
    carries a "__error__" key; callers decide how to degrade.
    """
    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                content = (getattr(msg, "reasoning_content", None) or "").strip()
            if not content:
                raise ValueError("empty judge content")
            parsed = parse_judge_json(content)
            if "__parse_error__" in parsed:
                raise ValueError(parsed["__parse_error__"])
            return parsed
        except Exception as e:  # noqa: BLE001 — we want to retry any failure
            last_err = str(e)
            if attempt < retries:
                time.sleep(2 * attempt)
    return {"__error__": last_err or "unknown judge failure"}


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def clamp(value: Any, lo: float, hi: float) -> float:
    """Clamp any value (tolerant to bad types) into [lo, hi]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = lo
    return max(lo, min(hi, v))

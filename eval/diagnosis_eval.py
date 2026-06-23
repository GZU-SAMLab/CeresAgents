"""
Evaluator 4 — Diagnostic Accuracy (LLM-as-Judge).

Only applicable to tracks where the system is expected to produce a disease
diagnosis: T2_diagnosis, T4_full, T5_multimodal, T6_environment.
Scoring philosophy
------------------
The judge evaluates **textual credibility signals in the response itself**:
specific observations linked to the diagnosis, symptom-to-disease reasoning
chains, calibrated confidence language, named references.  It does NOT know
which system has tool access, and it does not receive a tool-use log.  Any
system — Ceres, direct LLM — earns grounding credit only if its text shows
those signals on its own merits.

The `has_cv_evidence` / `has_rag_evidence` flags are still recorded in the
per-case output for downstream process analysis, but they do not affect the
score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._judge_base import (
    call_judge,
    case_has_image,
    clamp,
    collect_tool_evidence,
    extract_final_response,
    format_tool_evidence,
    has_evidence,
)

DIAGNOSIS_TRACKS = {"T2_diagnosis", "T4_full", "T5_multimodal", "T6_environment"}

_JUDGE_SYSTEM = (
    "You are an expert agricultural pathologist scoring an AI system's crop "
    "disease diagnosis against ground truth. "
    "Score from the response text against the case context. "
    "You do not know which model produced the response, but you DO have access "
    "to tool evidence — use it to verify specific claims: a citation, symptom "
    "observation, or quantitative claim that can be traced to tool evidence is "
    "more credible than one that cannot. "
    "Output ONLY a valid JSON object."
)

_JUDGE_PROMPT = """Evaluate the diagnosis below against the ground-truth disease.

## Case
Crop:                  {crop}
Ground-truth disease:  {disease_label}
User query:            {query}
Image provided:        {has_image}

## System final response
{system_output}

## Recorded tool evidence
{tool_evidence}

Use tool evidence to verify claims:
- A paper title or finding that appears in BOTH the response text and the
  RAG evidence is a verified citation → score D2 higher.
- A symptom percentage, lesion area, or quantitative measure that traces to
  CV tool output is verified → score D2 higher.
- If the response cites "studies show" or "literature confirms" without any
  matching RAG evidence, treat as unverified → do not award top-band D2.
- If tool evidence is empty or "(none)", the response has no external
  grounding → cap D2 at 18 for responses relying solely on parametric knowledge.

## Scoring rubric (total 100)

### D1. Disease Identification (0-40)
- 36-40: Correct disease named as the primary diagnosis at species level.
- 25-35: Correct disease mentioned but not ranked first, OR correct at
         family level only.
- 10-24: Broad category correct (e.g. "a fungal disease") but pathogen wrong.
- 0-9:   Wrong or irrelevant diagnosis.

### D2. Evidence Grounding (0-25)
Score the response's textual credibility signals. Look for:
  - specific, verifiable observations tied to the diagnosis (named symptoms,
    lesion patterns, affected area percentages);
  - explicit symptom-to-diagnosis reasoning chains;
  - citations of specific paper titles or extension sources that also appear
    in the RAG tool evidence (verified citations);
  - concrete quantitative claims traceable to CV or RAG tool output.

- 22-25: Rich, specific evidence with at least one VERIFIED signal (citation
         matching RAG evidence, or quantitative claim matching CV output).
         Clear reasoning chain with specific observations.
- 15-21: Specific evidence but no verifiable tool-trace; OR verified tool
         evidence but reasoning chain is implicit.
- 6-14:  Generic "based on the image" / "the literature supports this" style
         without specifics; no matching tool evidence.
- 1-5:   Mostly unsupported confident claims.
- 0:     No evidence-style reasoning at all.

### D3. Differential Diagnosis (0-20)
- 18-20: Appropriate differentials with clear distinguishing criteria.
- 12-17: Some differentials; distinctions partial.
- 5-11:  Minimal or weak differentials.
- 0-4:   No differential or clearly wrong.

### D4. Confidence Appropriateness (0-15)
- 13-15: Confidence matches evidence quality; hedges appropriately under
         uncertainty or partial evidence.
- 9-12:  Mostly calibrated, minor over/under-confidence.
- 4-8:   Notably over- or under-confident given evidence.
- 0-3:   Wildly miscalibrated.

## Output format (JSON only, no prose, no markdown)
{{
  "disease_identified": "string — disease the system named as primary",
  "disease_correct": true/false,
  "differential_reasonable": true/false,
  "confidence_appropriate": true/false,
  "disease_score": 0-40,
  "evidence_grounding_score": 0-25,
  "differential_score": 0-20,
  "confidence_score": 0-15,
  "reasoning": "2-3 sentence justification referencing specific textual signals"
}}"""


def _build_prompt(case: Dict[str, Any], response: Dict[str, Any]) -> str:
    meta = case.get("metadata", {})
    tool_evidence = format_tool_evidence(collect_tool_evidence(response), scopes=("cv", "rag", "kg"))
    return _JUDGE_PROMPT.format(
        crop=meta.get("crop", "unknown"),
        disease_label=meta.get("disease_label", "unknown"),
        query=(case.get("user_input", {}) or {}).get("query", ""),
        has_image="yes" if case_has_image(case) else "no",
        system_output=extract_final_response(response),
        tool_evidence=tool_evidence,
    )


def _clamp_scores(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp each sub-score to its allowed range; compute total."""
    disease = int(clamp(raw.get("disease_score", 0), 0, 40))
    grounding = int(clamp(raw.get("evidence_grounding_score", 0), 0, 25))
    diff = int(clamp(raw.get("differential_score", 0), 0, 20))
    conf = int(clamp(raw.get("confidence_score", 0), 0, 15))

    total = disease + grounding + diff + conf
    return {
        "disease_score": disease,
        "evidence_grounding_score": grounding,
        "differential_score": diff,
        "confidence_score": conf,
        "total_score": total,
    }


def evaluate_diagnosis(
    case: Dict[str, Any],
    response: Dict[str, Any],
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Score a single diagnosis case.  Returns None for non-diagnosis tracks."""
    track = case.get("metadata", {}).get("track", "")
    if track not in DIAGNOSIS_TRACKS:
        return None

    system_output = extract_final_response(response)
    evidence = collect_tool_evidence(response)

    base_record = {
        "case_id":               case.get("id", ""),
        "track":                 track,
        "disease_label":         case.get("metadata", {}).get("disease_label", ""),
        "system_output":         system_output[:500],
        "has_cv_evidence":       has_evidence(evidence, "cv"),
        "has_rag_evidence":      has_evidence(evidence, "rag"),
    }

    if not system_output:
        return {
            **base_record,
            "disease_identified":     "",
            "disease_correct":        False,
            "differential_reasonable": False,
            "confidence_appropriate": False,
            "disease_score":          0,
            "evidence_grounding_score": 0,
            "differential_score":     0,
            "confidence_score":       0,
            "llm_judge_score":        0,
            "llm_judge_reasoning":    "No system output",
        }

    raw = call_judge(
        client, model,
        system_prompt=_JUDGE_SYSTEM,
        user_prompt=_build_prompt(case, response),
        max_tokens=1200,
    )

    if "__error__" in raw:
        return {
            **base_record,
            "disease_identified":      "",
            "disease_correct":         False,
            "differential_reasonable": False,
            "confidence_appropriate":  False,
            "disease_score":           0,
            "evidence_grounding_score": 0,
            "differential_score":      0,
            "confidence_score":        0,
            "llm_judge_score":         0,
            "llm_judge_reasoning":     f"Judge error: {raw['__error__']}",
        }

    scores = _clamp_scores(raw)

    return {
        **base_record,
        "disease_identified":       str(raw.get("disease_identified", "")),
        "disease_correct":          bool(raw.get("disease_correct", False)),
        "differential_reasonable":  bool(raw.get("differential_reasonable", False)),
        "confidence_appropriate":   bool(raw.get("confidence_appropriate", False)),
        **scores,
        "llm_judge_score":          scores["total_score"],
        "llm_judge_reasoning":      raw.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Aggregation (shape preserved for downstream consumers)
# ---------------------------------------------------------------------------

def aggregate_diagnosis_results(results: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    valid = [r for r in results if r]
    if not valid:
        return {}

    total = len(valid)
    correct = sum(1 for r in valid if r.get("disease_correct"))
    avg_total = sum(r.get("llm_judge_score", 0) for r in valid) / total

    avg_disease  = sum(r.get("disease_score", 0)  for r in valid) / total
    avg_ground   = sum(r.get("evidence_grounding_score", 0) for r in valid) / total
    avg_diff     = sum(r.get("differential_score", 0) for r in valid) / total
    avg_conf     = sum(r.get("confidence_score", 0) for r in valid) / total
    grounded_rate = sum(1 for r in valid
                        if r.get("has_cv_evidence") or r.get("has_rag_evidence")) / total

    # Per-track breakdown (same shape analyze_results.py expects)
    track_stats: Dict[str, Dict[str, float]] = {}
    for r in valid:
        t = r.get("track", "unknown")
        s = track_stats.setdefault(t, {"total": 0, "score_sum": 0.0})
        s["total"] += 1
        s["score_sum"] += r.get("llm_judge_score", 0)
    per_track = {
        t: {"avg_score": round(v["score_sum"] / v["total"], 2), "n": int(v["total"])}
        for t, v in track_stats.items()
    }

    return {
        "total_cases":             total,
        "disease_correct_rate":    round(correct / total, 4),
        "avg_llm_judge_score":     round(avg_total, 2),
        "avg_disease_score":       round(avg_disease, 2),
        "avg_evidence_grounding_score": round(avg_ground, 2),
        "avg_differential_score":  round(avg_diff, 2),
        "avg_confidence_score":    round(avg_conf, 2),
        "tool_grounded_rate":      round(grounded_rate, 4),
        "per_track":               per_track,
    }
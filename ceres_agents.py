import os
import re
import json
import builtins
from contextlib import contextmanager
from typing import (
    TypedDict,
    Annotated,
    Sequence,
    Literal,
    Dict,
    Any,
    List,
    Optional,
    Callable,
)
from operator import add
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from prompts import (
    get_synthesize_prompt,
    get_unified_preprocessing_prompt,
    GENERALIST_SYSTEM,
    PATHOLOGIST_SYSTEM,
    PHYSIOLOGIST_SYSTEM,
    CHEMIST_SYSTEM,
    REBUTTAL_SYSTEM,
)
from tools import ALL_TOOLS


_AGENT_TAG_PREFIX = "agent:"


class TokenUsageTracker(BaseCallbackHandler):

    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        self._buckets = {}

    def _bucket(self, name: str) -> Dict[str, int]:
        return self._buckets.setdefault(
            name,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        usage: Dict[str, Any] = {}
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            raw = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if isinstance(raw, dict):
                usage = dict(raw)

        if not usage:
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                msg = getattr(generations[0][0], "message", None)
                md = getattr(msg, "usage_metadata", None) if msg is not None else None
                if isinstance(md, dict) and md:
                    usage = {
                        "prompt_tokens": md.get("input_tokens", 0),
                        "completion_tokens": md.get("output_tokens", 0),
                        "total_tokens": md.get("total_tokens", 0),
                    }

        def _as_int(v: Any) -> int:
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                return 0

        return {
            "prompt_tokens": _as_int(usage.get("prompt_tokens")),
            "completion_tokens": _as_int(usage.get("completion_tokens")),
            "total_tokens": _as_int(usage.get("total_tokens")),
        }

    def on_llm_end(self, response, *, tags=None, **kwargs) -> None:
        agent_name = "unknown"
        for t in tags or []:
            if isinstance(t, str) and t.startswith(_AGENT_TAG_PREFIX):
                agent_name = t[len(_AGENT_TAG_PREFIX) :] or "unknown"
                break

        usage = self._extract_usage(response)
        bucket = self._bucket(agent_name)
        bucket["calls"] += 1
        bucket["prompt_tokens"] += usage["prompt_tokens"]
        bucket["completion_tokens"] += usage["completion_tokens"]
        bucket["total_tokens"] += usage["total_tokens"]

    def snapshot(self) -> Dict[str, Any]:
        total = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        by_agent: Dict[str, Dict[str, int]] = {}
        for name, bucket in self._buckets.items():
            by_agent[name] = dict(bucket)
            for k in total:
                total[k] += bucket.get(k, 0)
        return {"total": total, "by_agent": by_agent}


class EvidenceItem(BaseModel):
    source: str
    content: str


class PathologistExpertOutput(BaseModel):
    reasoning: str = ""
    disease: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    differential: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    parsed_ok: bool = True
    parse_error: Optional[str] = None


class AbioticStressItem(BaseModel):
    type: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AlternativeHypothesisItem(BaseModel):
    diagnosis: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_action: Optional[str] = None


class PhysiologistExpertOutput(BaseModel):
    reasoning: str = ""
    environment_fit: Optional[bool] = None
    limiting_factors: List[str] = Field(default_factory=list)
    abiotic_stress: Optional[AbioticStressItem] = None
    veto: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    conflict_with_pathologist: bool = False
    conflict_reason: Optional[str] = None
    alternative_hypothesis: Optional[AlternativeHypothesisItem] = None
    raw_text: Optional[str] = None
    parsed_ok: bool = True
    parse_error: Optional[str] = None


class RebuttalOutput(BaseModel):
    rebuttal_mode: str = "A"
    rebuttal_reasoning: str = ""
    key_discriminating_factor: Optional[str] = None
    revised_disease: Optional[str] = None
    revised_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    revised_differential: List[str] = Field(default_factory=list)
    accepted_challenges: List[str] = Field(default_factory=list)
    rejected_challenges: List[str] = Field(default_factory=list)
    veto_considered: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    parsed_ok: bool = True
    parse_error: Optional[str] = None


class PrescriptionItem(BaseModel):
    name: Optional[str] = None
    registration_number: Optional[str] = None
    active_substance: Optional[str] = None
    formulation: Optional[str] = None
    dosage: Optional[str] = None
    max_applications: Optional[str] = None
    phi_days: Optional[int] = None
    phi_note: Optional[str] = None
    interval: Optional[str] = None
    toxicity: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    registration: Optional[str] = None


class RejectedItem(BaseModel):
    name: Optional[str] = None
    reason: Optional[str] = None


class UnregisteredAdvisoryItem(BaseModel):
    name: Optional[str] = None
    registration_number: Optional[str] = "unregistered"
    active_substance: Optional[str] = None
    formulation: Optional[str] = None
    reference_dosage: Optional[str] = None
    phi_days: Optional[int] = None
    phi_note: Optional[str] = None
    toxicity: Optional[str] = None
    mechanistic_rationale: Optional[str] = None
    disclaimer: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ChemistExpertOutput(BaseModel):
    reasoning: str = ""
    prescriptions: List[PrescriptionItem] = Field(default_factory=list)
    unregistered_advisory: List[UnregisteredAdvisoryItem] = Field(default_factory=list)
    rotation_plan: List[str] = Field(default_factory=list)
    rejected: List[RejectedItem] = Field(default_factory=list)
    compliance_verified: bool = False
    safety_violations: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    parsed_ok: bool = True
    parse_error: Optional[str] = None


def _pydantic_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_validate(model_cls, data: Any):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def parse_or_fallback(model_cls, text: str):
    import re as _re

    _first_exc: Optional[Exception] = None
    try:
        payload = json.loads(clean_json_output(text))
        model = _model_validate(model_cls, payload)
        return _pydantic_dump(model)
    except Exception as exc:
        _first_exc = exc

    try:
        json_match = _re.search(r"\{[\s\S]*\}", text)
        if json_match:
            payload = json.loads(json_match.group(0))
            model = _model_validate(model_cls, payload)
            result = _pydantic_dump(model)
            result["parsed_ok"] = True
            return result
    except Exception:
        pass

    model = _model_validate(
        model_cls,
        {
            "raw_text": text,
            "parsed_ok": False,
            "parse_error": str(_first_exc),
        },
    )
    return _pydantic_dump(model)


_LITERATURE_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+?)\]")


def _normalize_reference_title(raw_title: str) -> str:
    title = (raw_title or "").strip()
    if not title:
        return ""

    title = re.sub(r"\.(pdf|docx?|txt)\s*$", "", title, flags=re.IGNORECASE).strip()
    title = title.strip(" -_+|")

    if "+" in title:
        parts = [p.strip() for p in title.split("+") if p.strip()]
        if parts:
            for part in reversed(parts):
                if len(part) >= 8 and re.search(r"[A-Za-z]{3,}", part):
                    title = part
                    break
            else:
                title = parts[-1]

    return title.strip() or raw_title.strip()


def _extract_literature_sources(tool_results: Any) -> List[str]:
    if not isinstance(tool_results, list):
        return []
    seen: set = set()
    sources: List[str] = []
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name") != "retrieve_literature":
            continue
        text = item.get("result")
        if not isinstance(text, str):
            continue
        for match in _LITERATURE_SOURCE_RE.findall(text):
            src = _normalize_reference_title(match)
            if not src or src.lower() == "unknown" or src in seen:
                continue
            seen.add(src)
            sources.append(src)
    return sources


def _collect_references_from_slots(expert_slots: Dict[str, Any]) -> List[str]:
    seen: set = set()
    merged: List[str] = []
    for name in ("pathologist", "physiologist", "chemist", "generalist"):
        slot = (expert_slots or {}).get(name) or {}
        for src in _extract_literature_sources(slot.get("tool_results")):
            if src not in seen:
                seen.add(src)
                merged.append(src)
    return merged


def _derive_validated_hypotheses(
    pathologist_slot: Dict[str, Any],
    physiologist_slot: Dict[str, Any],
    fallback: Optional[List[str]] = None,
) -> List[str]:
    candidates: List[str] = []
    primary = (pathologist_slot or {}).get("disease")
    if primary:
        candidates.append(str(primary))
    for alt in (pathologist_slot or {}).get("differential", []) or []:
        if alt and str(alt) not in candidates:
            candidates.append(str(alt))

    veto_list = [
        str(v).lower() for v in (physiologist_slot or {}).get("veto", []) or [] if v
    ]
    if veto_list and candidates:

        def vetoed(name: str) -> bool:
            low = name.lower()
            return any(v in low or low in v for v in veto_list)

        filtered = [c for c in candidates if not vetoed(c)]
        if filtered:
            return filtered

    if candidates:
        return candidates
    if fallback:
        return [str(x) for x in fallback if x]
    return []


def _extract_tool_results(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    tool_calls_map: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    tool_call_id = tool_call.get("id")
                    if not tool_call_id:
                        continue
                    tool_calls_map[tool_call_id] = {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args", {}) or {},
                    }
                except Exception:
                    continue

        if isinstance(msg, ToolMessage):
            tool_id = getattr(msg, "tool_call_id", None)
            tool_info = tool_calls_map.get(tool_id, {})
            results.append(
                {
                    "tool_name": getattr(msg, "name", tool_info.get("name", "unknown")),
                    "args": tool_info.get("args", {}) or {},
                    "result": msg.content,
                }
            )

    return results


def clean_json_output(text: str) -> str:
    text = text.strip()

    import re

    json_block_match = re.search(r"```json\s*\n(.+?)\n```", text, re.DOTALL)
    if json_block_match:
        text = json_block_match.group(1).strip()
    else:
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0] if "\n" in text else text[:-3]
        text = text.strip()

    text = text.replace("\\'", "'")
    return text


def _get_expert_info(
    state: "BlackboardState", expert_name: str
) -> Optional[Dict[str, Any]]:
    expert_slots = state.get("expert_slots", {})
    if expert_name in expert_slots and expert_slots[expert_name]:
        return expert_slots[expert_name]

    for entry in state.get("reasoning_log", []):
        if entry.get("agent") == expert_name:
            return entry
    return None


def _extract_json_from_text(text: str) -> Optional[Dict]:
    if not text:
        return None

    import re

    json_match = re.search(r"```json\s*\n(.+?)\n```", text, re.DOTALL)
    if json_match:
        try:
            import json as json_lib

            return json_lib.loads(json_match.group(1))
        except:
            pass
    return None


def _join_lines(lines: List[Optional[str]]) -> str:
    return "\n".join([line for line in lines if line is not None])


def _format_blackboard_context(
    sections: List[Dict[str, Any]], trailing_lines: Optional[List[str]] = None
) -> str:
    if not sections and not trailing_lines:
        return ""

    block_lines: List[str] = ["📋 Other experts on the blackboard:", "═" * 50]

    for idx, section in enumerate(sections):
        title = section.get("title", "")
        content_lines = [line for line in section.get("lines", []) if line]

        if title:
            block_lines.append(f"【{title }】")
        block_lines.extend(content_lines)

        if idx < len(sections) - 1:
            block_lines.append("")

    if trailing_lines:
        if sections:
            block_lines.append("")
        block_lines.extend([line for line in trailing_lines if line])

    block_lines.append("═" * 50)
    return "\n".join(block_lines)


def _short_text(value: Any, max_len: int = 240) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if not (text.startswith("{") or text.startswith("[")):
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _summarize_tool_result_payload(payload: Any) -> Any:
    payload = _parse_json_if_possible(payload)
    if isinstance(payload, dict):
        summary: Dict[str, Any] = {}
        for key in [
            "code",
            "msg",
            "error",
            "summary",
            "count",
            "result_count",
            "endpoint",
        ]:
            if key in payload:
                summary[key] = payload.get(key)
        if "data" in payload and isinstance(payload.get("data"), list):
            summary["data_count"] = len(payload.get("data", []))
        if "forecast" in payload and isinstance(payload.get("forecast"), list):
            summary["forecast_days"] = len(payload.get("forecast", []))
        if "results" in payload and isinstance(payload.get("results"), list):
            summary["results_count"] = len(payload.get("results", []))
        if summary:
            return summary
    if isinstance(payload, list):
        return {"list_count": len(payload)}
    return _short_text(payload)


def _expert_structured_summary(agent: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    if agent == "pathologist":
        summary: Dict[str, Any] = {
            "disease": entry.get("disease"),
            "confidence": entry.get("confidence"),
            "differential": entry.get("differential", []),
            "conflicts": entry.get("conflicts", []),
            "parsed_ok": entry.get("parsed_ok", True),
        }
        if entry.get("original_disease") is not None:
            summary["original_disease"] = entry.get("original_disease")
        if entry.get("original_confidence") is not None:
            summary["original_confidence"] = entry.get("original_confidence")
        if entry.get("rebuttal"):
            summary["rebuttal"] = entry["rebuttal"]
        return summary
    if agent == "physiologist":
        summary = {
            "environment_fit": entry.get("environment_fit"),
            "limiting_factors": entry.get("limiting_factors", []),
            "abiotic_stress": entry.get("abiotic_stress"),
            "challenges": entry.get("challenges", []),
            "veto": entry.get("veto", []),
            "conflict_with_pathologist": entry.get("conflict_with_pathologist", False),
            "conflict_reason": entry.get("conflict_reason"),
            "alternative_hypothesis": entry.get("alternative_hypothesis"),
            "parsed_ok": entry.get("parsed_ok", True),
        }
        return summary
    if agent == "chemist":
        prescriptions = entry.get("prescriptions", []) or []
        unregistered = entry.get("unregistered_advisory", []) or []
        return {
            "prescription_count": len(prescriptions),
            "prescriptions": [
                item.get("name")
                for item in prescriptions
                if isinstance(item, dict) and item.get("name")
            ],
            "unregistered_advisory_count": len(unregistered),
            "unregistered_advisory": [
                item.get("name")
                for item in unregistered
                if isinstance(item, dict) and item.get("name")
            ],
            "rejected_count": len(entry.get("rejected", []) or []),
            "safety_violations": entry.get("safety_violations", []),
            "compliance_verified": entry.get("compliance_verified", False),
            "parsed_ok": entry.get("parsed_ok", True),
        }
    if agent == "generalist":
        return {
            "content_preview": _short_text(entry.get("content", ""), max_len=280),
        }
    return {"keys": list(entry.keys())}


def _compact_tool_evidence(
    tool_results: Any, max_items: int = 6
) -> List[Dict[str, Any]]:
    if not isinstance(tool_results, list):
        return []

    compact_items: List[Dict[str, Any]] = []
    for tool_item in tool_results[:max_items]:
        if not isinstance(tool_item, dict):
            continue
        compact_items.append(
            {
                "tool": tool_item.get("tool_name"),
                "args": tool_item.get("args", {}),
                "result_summary": _summarize_tool_result_payload(
                    tool_item.get("result")
                ),
            }
        )
    return compact_items


def _compact_expert_slot_for_synthesis(
    agent: str, slot: Dict[str, Any]
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "agent": agent,
        "parsed_ok": slot.get("parsed_ok", True),
    }

    references = _extract_literature_sources(slot.get("tool_results", []))
    if references:
        compact["references"] = references

    if agent == "pathologist":
        compact.update(
            {
                "disease": slot.get("disease"),
                "confidence": slot.get("confidence"),
                "differential": slot.get("differential", []),
                "conflicts": slot.get("conflicts", []),
                "reasoning": _short_text(slot.get("reasoning", ""), max_len=700),
                "tool_evidence": _compact_tool_evidence(
                    slot.get("tool_results", []), max_items=5
                ),
            }
        )
        if slot.get("original_disease") is not None:
            compact["original_disease"] = slot.get("original_disease")
        if slot.get("original_confidence") is not None:
            compact["original_confidence"] = slot.get("original_confidence")
        if slot.get("rebuttal"):
            compact["rebuttal"] = slot.get("rebuttal")
        return compact

    if agent == "physiologist":
        compact.update(
            {
                "environment_fit": slot.get("environment_fit"),
                "limiting_factors": slot.get("limiting_factors", []),
                "abiotic_stress": slot.get("abiotic_stress"),
                "challenges": slot.get("challenges", []),
                "veto": slot.get("veto", []),
                "conflict_with_pathologist": slot.get(
                    "conflict_with_pathologist", False
                ),
                "conflict_reason": slot.get("conflict_reason"),
                "alternative_hypothesis": slot.get("alternative_hypothesis"),
                "reasoning": _short_text(slot.get("reasoning", ""), max_len=500),
                "tool_evidence": _compact_tool_evidence(
                    slot.get("tool_results", []), max_items=3
                ),
            }
        )
        return compact

    if agent == "chemist":
        prescriptions = slot.get("prescriptions", []) or []
        compact_prescriptions: List[Dict[str, Any]] = []
        for item in prescriptions[:5]:
            if not isinstance(item, dict):
                continue
            compact_prescriptions.append(
                {
                    "name": item.get("name"),
                    "active_substance": item.get("active_substance"),
                    "dosage": item.get("dosage"),
                    "interval": item.get("interval"),
                    "registration": item.get("registration"),
                    "warnings": item.get("warnings", [])[:3],
                }
            )

        compact.update(
            {
                "prescriptions": compact_prescriptions,
                "rotation_plan": (slot.get("rotation_plan", []) or [])[:4],
                "rejected": slot.get("rejected", []) or [],
                "compliance_verified": slot.get("compliance_verified", False),
                "safety_violations": slot.get("safety_violations", []) or [],
                "reasoning": _short_text(slot.get("reasoning", ""), max_len=700),
                "tool_evidence": _compact_tool_evidence(
                    slot.get("tool_results", []), max_items=5
                ),
            }
        )
        return compact

    compact["summary"] = _short_text(slot, max_len=500)
    return compact


@contextmanager
def _suppress_print(enabled: bool):
    if enabled:
        yield
        return
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = original_print


class DiagnosticSlots(TypedDict):
    crop: Optional[str]
    symptoms: List[str]
    environment: Dict[str, Any]
    history: List[str]
    validated_hypotheses: List[str]
    recommendations: List[str]


class BlackboardState(TypedDict):
    query: str
    image_path: Optional[str]
    location: Optional[str]
    history_context: List[str]
    run_id: str

    messages: Annotated[Sequence[BaseMessage], add]
    reasoning_log: Annotated[List[Dict[str, Any]], add]

    slots: DiagnosticSlots

    expert_slots: Dict[str, Dict[str, Any]]

    complexity_score: float
    complexity_level: str
    route_decision: Literal["fast", "expert"]
    intent: Literal["diagnosis", "prescription", "both", "qa"]
    intent_confidence: float
    intent_reasoning: str
    active_experts: List[str]
    has_conflicts: bool

    conflict_details: List[str]

    visual_results: Dict[str, Any]

    kg_results: List[Dict[str, Any]]
    rag_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]

    final_response: Optional[str]
    compliance_verified: bool


class CoordinatorAgent:

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List,
        agent_name: str = "coordinator",
        callbacks: Optional[List] = None,
    ):
        self.base_llm = llm
        self.tools = tools
        self.agent_name = agent_name
        self.callbacks = list(callbacks or [])

    def _run_config(
        self, override_agent: Optional[str] = None, **extra
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        cfg["tags"] = [f"{_AGENT_TAG_PREFIX }{override_agent or self .agent_name }"]
        if self.callbacks:
            cfg["callbacks"] = list(self.callbacks)
        cfg.update(extra)
        return cfg

    def synthesize_response(self, state: BlackboardState) -> str:
        prompt = get_synthesize_prompt()
        chain = prompt | self.base_llm

        compact_mode = _env_flag("CERES_SYNTHESIS_COMPACT", default=True)

        if compact_mode:
            visual_results = state.get("visual_results", {}) or {}
            visual_summary: Dict[str, Any] = {
                "count": len(visual_results),
                "keys": list(visual_results.keys()),
            }
            pathologist_visual = visual_results.get("pathologist_analysis")
            if isinstance(pathologist_visual, dict):
                visual_summary["pathologist"] = {
                    "disease": pathologist_visual.get("disease"),
                    "confidence": pathologist_visual.get("confidence"),
                }

            raw_kg_results = state.get("kg_results", []) or []
            kg_payload = [
                _summarize_tool_result_payload(item) for item in raw_kg_results[:10]
            ]

            visual_payload = visual_summary
            reasoning_payload = state.get("reasoning_log", []) or []
            slots_payload = state.get("slots", {})
        else:
            visual_payload = state.get("visual_results", {})
            reasoning_payload = state.get("reasoning_log", [])
            kg_payload = state.get("kg_results", [])
            slots_payload = state.get("slots", {})

        intent_str = str(state.get("intent") or "diagnosis")
        active_experts: List[str] = state.get("active_experts") or []
        references_pool: List[str] = state.get("references_pool") or []
        references_payload = (
            "\n".join(f"- {src }" for src in references_pool)
            if references_pool
            else "(none)"
        )

        result = chain.invoke(
            {
                "query": state["query"],
                "intent": intent_str,
                "active_experts": (
                    ", ".join(active_experts) if active_experts else "none"
                ),
                "slots": json.dumps(slots_payload, ensure_ascii=False),
                "visual_results": json.dumps(visual_payload, ensure_ascii=False),
                "reasoning_log": json.dumps(reasoning_payload, ensure_ascii=False),
                "kg_results": json.dumps(kg_payload, ensure_ascii=False),
                "references_pool": references_payload,
            },
            config=self._run_config(),
        )

        return result.content


class GeneralistAgent:

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List,
        callbacks=None,
        agent_name: str = "generalist",
    ):
        allowed_tools = [
            t
            for t in tools
            if t.name
            in [
                "classify_disease",
                "retrieve_literature",
                "analyze_image",
            ]
        ]
        self.tools = allowed_tools
        self.base_llm = llm
        self.agent_name = agent_name
        self.callbacks = list(callbacks or [])

        from prompts import GENERALIST_SYSTEM

        self.agent = create_react_agent(llm, allowed_tools, prompt=GENERALIST_SYSTEM)

    def _run_config(self, **extra) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"tags": [f"{_AGENT_TAG_PREFIX }{self .agent_name }"]}
        if self.callbacks:
            cfg["callbacks"] = list(self.callbacks)
        cfg.update(extra)
        return cfg

    def respond(self, state: BlackboardState) -> Dict[str, Any]:
        messages = [HumanMessage(content=state["query"])]

        result = self.agent.invoke(
            {"messages": messages}, config=self._run_config(recursion_limit=12)
        )

        final_message = result["messages"][-1]

        response_data = {
            "agent": "generalist",
            "content": final_message.content,
            "tool_results": _extract_tool_results(result["messages"]),
            "run_id": state.get("run_id"),
            "timestamp": datetime.now().isoformat(),
        }

        return response_data


class PathologistAgent:

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List,
        callbacks=None,
        agent_name: str = "pathologist",
    ):
        allowed_tools = [
            t
            for t in tools
            if t.name
            in [
                "classify_disease",
                "segment_disease",
                "analyze_image",
                "retrieve_literature",
            ]
        ]
        self.tools = allowed_tools
        self.base_llm = llm
        self.agent_name = agent_name
        self.callbacks = list(callbacks or [])

        from prompts import PATHOLOGIST_SYSTEM

        self.agent = create_react_agent(llm, allowed_tools, prompt=PATHOLOGIST_SYSTEM)

    def _run_config(self, **extra) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"tags": [f"{_AGENT_TAG_PREFIX }{self .agent_name }"]}
        if self.callbacks:
            cfg["callbacks"] = list(self.callbacks)
        cfg.update(extra)
        return cfg

    def analyze(self, state: BlackboardState) -> Dict[str, Any]:
        slots = state.get("slots", {})

        query_lines = ["[Diagnosis task]"]
        if slots.get("crop"):
            query_lines.append(f"Crop: {slots ['crop']}")
        if slots.get("symptoms"):
            query_lines.append(f"Symptoms: {', '.join (slots ['symptoms'])}")
        else:
            query_lines.append(f"Original query: {state ['query']}")

        if state.get("image_path"):
            query_lines.append(f"Image: {state ['image_path']}")
        else:
            query_lines.append(
                "No image is attached — you MUST NOT call analyze_image, classify_disease, or segment_disease; "
                "follow the text-only workflow (retrieve_literature only for differentials and validation)."
            )

        validated = slots.get("validated_hypotheses") or []
        if validated:
            query_lines.append(
                f"User-asserted / lab-confirmed diagnoses (trust these — confidence ≥ 0.85): {', '.join (validated )}"
            )

        query_text = _join_lines(query_lines)

        messages = [HumanMessage(content=query_text)]

        result = self.agent.invoke(
            {"messages": messages}, config=self._run_config(recursion_limit=10)
        )

        final_message = result["messages"][-1]

        parsed = parse_or_fallback(PathologistExpertOutput, final_message.content)

        analysis_data = {
            "agent": "pathologist",
            "final_output": final_message.content,
            **parsed,
            "tool_results": _extract_tool_results(result["messages"]),
            "run_id": state.get("run_id"),
            "timestamp": datetime.now().isoformat(),
        }

        return analysis_data


class PhysiologistAgent:

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List,
        callbacks=None,
        agent_name: str = "physiologist",
    ):
        allowed_tools = [
            t
            for t in tools
            if t.name
            in [
                "retrieve_literature",
            ]
        ]
        self.tools = allowed_tools
        self.base_llm = llm
        self.agent_name = agent_name
        self.callbacks = list(callbacks or [])

        from prompts import PHYSIOLOGIST_SYSTEM

        self.agent = create_react_agent(llm, allowed_tools, prompt=PHYSIOLOGIST_SYSTEM)

    def _run_config(self, **extra) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"tags": [f"{_AGENT_TAG_PREFIX }{self .agent_name }"]}
        if self.callbacks:
            cfg["callbacks"] = list(self.callbacks)
        cfg.update(extra)
        return cfg

    def audit(self, state: BlackboardState) -> Dict[str, Any]:
        query_text = _join_lines(
            [
                f"Query: {state ['query']}",
                f"Environment: {json .dumps (state ['slots'].get ('environment',{}),ensure_ascii =False )}",
                f"Hypotheses: {', '.join (state ['slots'].get ('validated_hypotheses',[]))}",
            ]
        )

        pathologist_diagnosis = _get_expert_info(state, "pathologist")

        if pathologist_diagnosis and pathologist_diagnosis.get("disease") in [None, ""]:
            parsed = _extract_json_from_text(pathologist_diagnosis.get("raw_text", ""))
            if parsed:
                pathologist_diagnosis = {**pathologist_diagnosis, **parsed}

        if pathologist_diagnosis:
            disease = pathologist_diagnosis.get("disease", "unknown")
            confidence = pathologist_diagnosis.get("confidence", 0)
            differential = pathologist_diagnosis.get("differential", [])
            path_reasoning = pathologist_diagnosis.get("reasoning", "")

            section_lines = [
                f"- Primary diagnosis: {disease }",
                f"- Confidence: {confidence *100 :.1f}%",
            ]
            if differential:
                section_lines.append(f"- Differential: {', '.join (differential )}")
            if path_reasoning:
                section_lines.append(
                    f"- Pathologist reasoning (abridged): {_short_text (path_reasoning ,max_len =400 )}"
                )

            trailing = [
                "Your task: independently audit whether the observed environment supports the pathologist's diagnosis.",
                "Based on pathogen biology (temperature, humidity, season, host phenology, soil), identify SPECIFIC conflicts.",
                "Write each conflict as a full sentence in the `challenges` list — the pathologist will respond to each in a rebuttal step.",
                "Use `veto` ONLY for pathogens that are environmentally IMPOSSIBLE (>30% outside viable range, incompatible season, etc.).",
                "If the environment fully supports the diagnosis, leave both `challenges` and `veto` empty — that is a valid outcome.",
            ]

            query_text = f"{query_text }\n\n" + _format_blackboard_context(
                sections=[
                    {
                        "title": "Pathologist — diagnosis",
                        "lines": section_lines,
                    }
                ],
                trailing_lines=trailing,
            )

        messages = [HumanMessage(content=query_text)]

        result = self.agent.invoke(
            {"messages": messages}, config=self._run_config(recursion_limit=10)
        )

        final_message = result["messages"][-1]

        parsed = parse_or_fallback(PhysiologistExpertOutput, final_message.content)

        audit_data = {
            "agent": "physiologist",
            "final_output": final_message.content,
            **parsed,
            "tool_results": _extract_tool_results(result["messages"]),
            "run_id": state.get("run_id"),
            "timestamp": datetime.now().isoformat(),
        }

        return audit_data


class ChemistAgent:

    def __init__(
        self, llm: ChatOpenAI, tools: List, callbacks=None, agent_name: str = "chemist"
    ):
        allowed_tools = [
            t
            for t in tools
            if t.name
            in [
                "get_pesticides_by_disease",
                "get_pesticides_by_crop_and_disease",
                "get_crop_and_disease_by_pesticide",
                "list_diseases_by_crop",
                "get_pesticide_detail",
                "retrieve_literature",
            ]
        ]
        self.tools = allowed_tools
        self.base_llm = llm
        self.agent_name = agent_name
        self.callbacks = list(callbacks or [])

        from prompts import CHEMIST_SYSTEM

        self.agent = create_react_agent(llm, allowed_tools, prompt=CHEMIST_SYSTEM)

    def _run_config(self, **extra) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"tags": [f"{_AGENT_TAG_PREFIX }{self .agent_name }"]}
        if self.callbacks:
            cfg["callbacks"] = list(self.callbacks)
        cfg.update(extra)
        return cfg

    def prescribe(self, state: BlackboardState) -> Dict[str, Any]:
        pathologist_diagnosis = _get_expert_info(state, "pathologist")
        physiologist_audit = _get_expert_info(state, "physiologist")

        if pathologist_diagnosis and pathologist_diagnosis.get("disease") in [None, ""]:
            parsed = _extract_json_from_text(pathologist_diagnosis.get("raw_text", ""))
            if parsed:
                pathologist_diagnosis = {**pathologist_diagnosis, **parsed}
                print(
                    f"   💡 Parsed pathologist diagnosis from raw_text: {parsed .get ('disease','N/A')}"
                )

        crop = state["slots"].get("crop", "unknown")
        if (crop in ("unknown", "未知", None, "")) and pathologist_diagnosis:
            disease = pathologist_diagnosis.get("disease", "")
            if disease and "_" in str(disease):
                crop_candidate = str(disease).split("_")[0].lower()
                crop_map = {
                    "apple": "apple",
                    "potato": "potato",
                    "tomato": "tomato",
                    "grape": "grape",
                    "corn": "corn",
                    "wheat": "wheat",
                    "rice": "rice",
                    "soybean": "soybean",
                    "cotton": "cotton",
                }
                if crop_candidate in crop_map:
                    crop = crop_map[crop_candidate]
                    print(f"   💡 Inferred crop from diagnosis label: {crop }")

        query_text = _join_lines(
            [
                f"Crop: {crop }",
                f"Disease hypotheses: {', '.join (state ['slots'].get ('validated_hypotheses',[]))}",
            ]
        )

        if pathologist_diagnosis or physiologist_audit:
            sections = []

            if pathologist_diagnosis:
                disease = pathologist_diagnosis.get("disease", "unknown")
                confidence = pathologist_diagnosis.get("confidence", 0)
                sections.append(
                    {
                        "title": "Pathologist — diagnosis",
                        "lines": [
                            f"- Disease: {disease }",
                            f"- Confidence: {confidence *100 :.1f}%",
                        ],
                    }
                )

            if physiologist_audit:
                env_fit = physiologist_audit.get("environment_fit", True)
                veto = physiologist_audit.get("veto", [])
                physio_lines = [f"- Environment fit: {'yes'if env_fit else 'no'}"]
                if veto:
                    physio_lines.append(f"- ⚠️ Vetoed pathogens: {', '.join (veto )}")
                    physio_lines.append(
                        "- If the diagnosis matches a vetoed pathogen, pause or revise prescriptions."
                    )
                sections.append(
                    {
                        "title": "Physiologist — environmental audit",
                        "lines": physio_lines,
                    }
                )

            query_text = f"{query_text }\n\n" + _format_blackboard_context(
                sections=sections,
                trailing_lines=[
                    "Your task: query the knowledge graph for registered pesticide options for the confirmed disease.",
                    "Verify crop–disease–product registration via KG results.",
                ],
            )

        messages = [HumanMessage(content=query_text)]

        result = self.agent.invoke(
            {"messages": messages}, config=self._run_config(recursion_limit=16)
        )

        final_message = result["messages"][-1]

        tool_results = _extract_tool_results(result["messages"])
        parsed = parse_or_fallback(ChemistExpertOutput, final_message.content)

        prescription_data = {
            "agent": "chemist",
            "final_output": final_message.content,
            **parsed,
            "tool_results": tool_results,
            "run_id": state.get("run_id"),
            "timestamp": datetime.now().isoformat(),
        }

        if not prescription_data.get("compliance_verified"):
            kg_results = [
                tr.get("result")
                for tr in tool_results
                if isinstance(tr.get("tool_name"), str)
                and str(tr.get("tool_name")).startswith("get_pesticides")
            ]
            if kg_results:
                prescription_data["compliance_verified"] = True
                for kg_result in kg_results:
                    low = str(kg_result).lower()
                    if (
                        "error" in low
                        or "未找到" in str(kg_result)
                        or "not found" in low
                        or '"count": 0' in low
                    ):
                        prescription_data["safety_violations"].append(
                            "No registered pesticide data found in KG"
                        )

        return prescription_data


class CeresAgentsGraph:

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = True,
        disabled_tools: Optional[set[str]] = None,
    ):
        self.verbose = verbose
        self.disabled_tools = set(disabled_tools or set())

        resolved_model = (
            model or os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL", "qwen-plus")
        )
        resolved_api_key = (
            api_key
            or os.getenv("MODEL_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("MODEL_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.llm = ChatOpenAI(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=0.1,
        )
        self.model_name = resolved_model
        self.base_url = resolved_base_url

        self.tools = [t for t in ALL_TOOLS if t.name not in self.disabled_tools]

        self.token_tracker = TokenUsageTracker()
        tracker_callbacks = [self.token_tracker]

        self.coordinator = CoordinatorAgent(
            self.llm,
            self.tools,
            agent_name="coordinator",
            callbacks=tracker_callbacks,
        )
        self.generalist = GeneralistAgent(
            self.llm,
            self.tools,
            agent_name="generalist",
            callbacks=tracker_callbacks,
        )
        self.pathologist = PathologistAgent(
            self.llm,
            self.tools,
            agent_name="pathologist",
            callbacks=tracker_callbacks,
        )
        self.physiologist = PhysiologistAgent(
            self.llm,
            self.tools,
            agent_name="physiologist",
            callbacks=tracker_callbacks,
        )
        self.chemist = ChemistAgent(
            self.llm,
            self.tools,
            agent_name="chemist",
            callbacks=tracker_callbacks,
        )

        self.graph = self._build_graph()

        self._current_session_id: Optional[str] = None
        self.last_structured_log: Optional[Dict[str, Any]] = None
        self.last_state: Optional[Dict[str, Any]] = None

    def get_llm_config(self) -> Dict[str, Any]:
        return {
            "model": getattr(self, "model_name", None)
            or getattr(self.llm, "model_name", None),
            "base_url": getattr(self, "base_url", None),
        }

    def _polish_fast_track_response(self, query: str, draft: str) -> str:
        if not draft or len(draft.strip()) < 40:
            return draft

        polish_prompt = f"""
You are an agricultural plant-health assistant. Polish the draft below into a coherent, concise, trustworthy answer in English.

Rules:
1) Keep facts from the draft only; do not add new conclusions.
2) Tight structure: definition / key traits / conditions / impact / practical tips (merge as needed).
3) Remove repetition and citation clutter.
4) Use block-level references if needed; at most 3–5 references total.
5) Natural prose, not a stitched collage.
6) Prefer one key source per point (two only if necessary).
7) If information is thin, end with one clear next step (e.g. upload a clear leaf close-up).

User question: {query }

Draft:
{draft }
""".strip()

        try:
            result = self.coordinator.base_llm.invoke(
                polish_prompt,
                config=self.coordinator._run_config(),
            )
            polished = (result.content or "").strip()
            return polished if polished else draft
        except Exception:
            return draft

    def _build_blackboard_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        expert_slots = state.get("expert_slots", {}) or {}
        expert_slot_summary: Dict[str, Any] = {}
        for expert_name, slot in expert_slots.items():
            if isinstance(slot, dict) and slot:
                expert_slot_summary[expert_name] = _expert_structured_summary(
                    expert_name, slot
                )
            else:
                expert_slot_summary[expert_name] = {}

        return {
            "slots": state.get("slots", {}),
            "expert_slots": expert_slot_summary,
            "route": {
                "complexity_score": state.get("complexity_score"),
                "complexity_level": state.get("complexity_level"),
                "intent": state.get("intent"),
                "intent_confidence": state.get("intent_confidence"),
                "route_decision": state.get("route_decision"),
                "active_experts": state.get("active_experts", []),
            },
            "discussion": {
                "has_conflicts": state.get("has_conflicts"),
                "conflict_details": state.get("conflict_details", []),
            },
            "knowledge_pool": {
                "visual_count": len(state.get("visual_results", {}) or {}),
                "kg_count": len(state.get("kg_results", []) or []),
                "rag_count": len(state.get("rag_results", []) or []),
                "web_count": len(state.get("web_results", []) or []),
                "compliance_verified": state.get("compliance_verified"),
            },
        }

    def _build_expert_dialogue(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        reasoning_log = state.get("reasoning_log", []) or []
        current_run_id = state.get("run_id")
        dialogue: List[Dict[str, Any]] = []

        for idx, entry in enumerate(reasoning_log, start=1):
            if not isinstance(entry, dict):
                continue
            entry_run_id = entry.get("run_id")
            if current_run_id and entry_run_id and entry_run_id != current_run_id:
                continue

            agent = entry.get("agent", "unknown")
            if agent not in ["generalist", "pathologist", "physiologist", "chemist"]:
                continue

            raw_message = (
                entry.get("reasoning")
                or entry.get("content")
                or entry.get("final_output")
                or entry.get("raw_text")
                or ""
            )

            tool_calls = []
            for tool_item in entry.get("tool_results", []) or []:
                if not isinstance(tool_item, dict):
                    continue
                tool_calls.append(
                    {
                        "tool": tool_item.get("tool_name"),
                        "args": tool_item.get("args", {}),
                        "result_summary": _summarize_tool_result_payload(
                            tool_item.get("result")
                        ),
                    }
                )

            dialogue.append(
                {
                    "turn": idx,
                    "agent": agent,
                    "timestamp": entry.get("timestamp"),
                    "message": _short_text(raw_message, max_len=1000),
                    "structured_summary": _expert_structured_summary(agent, entry),
                    "tool_calls": tool_calls,
                }
            )

        return dialogue

    def build_structured_log(
        self, state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        state_obj = state or getattr(self, "last_state", None)
        if not state_obj:
            return None

        return {
            "run_id": state_obj.get("run_id"),
            "timestamp": datetime.now().isoformat(),
            "query": state_obj.get("query"),
            "image_path": state_obj.get("image_path"),
            "location": state_obj.get("location"),
            "dialogue": self._build_expert_dialogue(state_obj),
            "blackboard_snapshot": self._build_blackboard_snapshot(state_obj),
            "token_usage": self.token_tracker.snapshot(),
            "final_response": state_obj.get("final_response"),
        }

    def _route_after_pathologist(self, state: BlackboardState) -> str:
        active = state.get("active_experts", [])
        if "physiologist" in active:
            return "physiologist"
        if "chemist" in active:
            return "chemist"
        return "finalize"

    def _route_after_rebuttal(self, state: BlackboardState) -> str:
        active = state.get("active_experts", [])
        return "chemist" if "chemist" in active else "finalize"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(BlackboardState)

        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("unified_preprocessing", self._unified_preprocessing_node)
        workflow.add_node("route_decision", self._route_decision_node)
        workflow.add_node("early_return", self._early_return_node)

        workflow.add_node("generalist_agent", self._generalist_node)

        workflow.add_node("expert_selection", self._expert_selection_node)
        workflow.add_node("pathologist_agent", self._pathologist_node)
        workflow.add_node("physiologist_agent", self._physiologist_node)
        workflow.add_node("rebuttal", self._rebuttal_node)
        workflow.add_node("chemist_agent", self._chemist_node)
        workflow.add_node("synthesize", self._synthesize_node)

        workflow.set_entry_point("initialize")

        workflow.add_edge("initialize", "unified_preprocessing")
        workflow.add_conditional_edges(
            "unified_preprocessing",
            self._post_intent_condition,
            {"early": "early_return", "route": "route_decision"},
        )

        workflow.add_conditional_edges(
            "route_decision",
            self._route_condition,
            {"fast": "generalist_agent", "expert": "expert_selection"},
        )

        workflow.add_edge("generalist_agent", END)

        workflow.add_edge("expert_selection", "pathologist_agent")

        workflow.add_conditional_edges(
            "pathologist_agent",
            self._route_after_pathologist,
            {
                "physiologist": "physiologist_agent",
                "chemist": "chemist_agent",
                "finalize": "synthesize",
            },
        )

        workflow.add_edge("physiologist_agent", "rebuttal")

        workflow.add_conditional_edges(
            "rebuttal",
            self._route_after_rebuttal,
            {
                "chemist": "chemist_agent",
                "finalize": "synthesize",
            },
        )

        workflow.add_edge("chemist_agent", "synthesize")

        workflow.add_edge("early_return", END)
        workflow.add_edge("synthesize", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _initialize_node(self, state: BlackboardState) -> BlackboardState:
        print(f"🔄 [初始化] 清空历史推理日志，开始新的诊断会话")
        return {
            "messages": [HumanMessage(content=state["query"])],
            "reasoning_log": [],
            "has_conflicts": False,
            "visual_results": {},
            "kg_results": [],
            "rag_results": [],
            "web_results": [],
            "expert_slots": {"pathologist": {}, "physiologist": {}, "chemist": {}},
            "conflict_details": [],
            "active_experts": [],
        }

    def _unified_preprocessing_node(self, state: BlackboardState) -> BlackboardState:
        print(f"\n[preprocess] intent + routing + experts + slots")

        prompt = get_unified_preprocessing_prompt()
        chain = prompt | self.coordinator.base_llm

        forced_fast = str(state.get("query", "")).startswith("[EVAL:force_fast]")
        if forced_fast:
            print("   ✓ force_fast marker detected — bypassing LLM routing")
            return {
                "complexity_score": 2.0,
                "complexity_level": "low",
                "route_decision": "fast",
                "active_experts": [],
                "slots": DiagnosticSlots(
                    crop=None,
                    symptoms=[],
                    environment={},
                    history=[],
                    validated_hypotheses=[],
                    recommendations=[],
                ),
                "intent": "diagnosis",
                "intent_confidence": 1.0,
                "intent_reasoning": "forced fast-route by evaluation marker",
            }

        result = chain.invoke(
            {
                "query": state["query"],
                "has_image": "yes" if state.get("image_path") else "no",
                "history": "\n".join(state.get("history_context", [])),
            },
            config=self.coordinator._run_config(),
        )

        def _to_list(value):
            if value is None:
                return []
            if isinstance(value, list):
                return [str(x) for x in value if x is not None]
            if isinstance(value, str):
                return [value] if value.strip() else []
            return [str(value)]

        def _to_dict(value):
            return value if isinstance(value, dict) else {}

        try:
            parsed = json.loads(clean_json_output(result.content))

            intent_raw = parsed.get("intent", {})
            intent = intent_raw.get("intent", "diagnosis")
            confidence = float(intent_raw.get("confidence", 0.5))
            reasoning = intent_raw.get("reasoning", "")

            routing_raw = parsed.get("routing", {})
            route = routing_raw.get("route", "expert")
            active_experts = [
                e
                for e in routing_raw.get("active_experts", [])
                if e in ("pathologist", "physiologist", "chemist")
            ]
            routing_reason = routing_raw.get("reason", "")

            if route == "expert" and not active_experts:
                active_experts = ["pathologist"]

            level = parsed.get("complexity_level", "medium")
            if level not in ("low", "medium", "high"):
                level = "medium"
            level_score_map = {"low": 2.0, "medium": 5.0, "high": 8.0}
            s_complexity = level_score_map.get(level, 5.0)

            slots_raw = parsed.get("slots", {})
            slots = DiagnosticSlots(
                crop=slots_raw.get("crop"),
                symptoms=_to_list(slots_raw.get("symptoms")),
                environment=_to_dict(slots_raw.get("environment")),
                history=_to_list(slots_raw.get("history")),
                validated_hypotheses=_to_list(slots_raw.get("validated_hypotheses")),
                recommendations=_to_list(slots_raw.get("recommendations")),
            )

        except Exception as e:
            print(f"   ⚠️ Preprocess parse failed, using fallback rules: {e }")
            s_complexity = 5.0
            level = "medium"
            slots = DiagnosticSlots(
                crop=None,
                symptoms=[],
                environment={},
                history=[],
                validated_hypotheses=[],
                recommendations=[],
            )
            query_lower = state["query"].lower()
            if any(kw in query_lower for kw in ["你好", "hi", "hello", "在吗", "hey"]):
                intent, route, active_experts = "greeting", "fast", []
            elif any(
                kw in query_lower
                for kw in ["什么是", "介绍", "了解", "科普", "what is", "tell me about"]
            ):
                intent, route, active_experts = "knowledge_qa", "fast", []
            elif (
                "用药" in query_lower
                or "怎么治" in query_lower
                or "农药" in query_lower
                or "fungicide" in query_lower
                or "pesticide" in query_lower
                or "what to spray" in query_lower
            ):
                intent, route, active_experts = (
                    "prescription",
                    "expert",
                    ["pathologist", "chemist"],
                )
            else:
                intent, route, active_experts = "diagnosis", "expert", ["pathologist"]
            confidence = 0.5
            reasoning = "fallback heuristic"
            routing_reason = "fallback"

        intent_display = {
            "greeting": "greeting",
            "out_of_scope": "out_of_scope",
            "knowledge_qa": "knowledge_qa",
            "diagnosis": "diagnosis",
            "prescription": "prescription",
            "diagnosis_prescription": "diagnosis+prescription",
        }
        print(
            f"   • intent: {intent_display .get (intent ,intent )} (confidence: {confidence :.2f})"
        )
        print(f"   • route: {route } | experts: {active_experts or '— (Fast Track)'}")
        print(f"   • complexity: {level } | reason: {routing_reason [:80 ]}")

        intent_map = {
            "knowledge_qa": "qa",
            "diagnosis": "diagnosis",
            "prescription": "prescription",
            "diagnosis_prescription": "both",
        }
        internal_intent = intent_map.get(intent, intent)

        if intent == "greeting":
            print(f"   ✓ greeting — short-circuit response")
            return {
                "complexity_score": s_complexity,
                "complexity_level": level,
                "route_decision": "fast",
                "active_experts": [],
                "slots": slots,
                "intent": "greeting",
                "intent_confidence": confidence,
                "intent_reasoning": reasoning,
                "final_response": (
                    "Hello! I am CeresAgents, an intelligent crop disease diagnosis assistant.\n\n"
                    "I can help you with:\n"
                    "- Diagnosing crop diseases from symptoms or images\n"
                    "- Science-based, compliance-aware pesticide suggestions (via the knowledge graph when available)\n"
                    "- How environment may affect disease risk\n"
                    "- General plant pathology Q&A\n\n"
                    "Tell me what is happening to your crop, or upload a clear photo of affected tissue."
                ),
            }

        if intent == "out_of_scope":
            print(f"   ✓ out-of-scope — short-circuit response")
            return {
                "complexity_score": s_complexity,
                "complexity_level": level,
                "route_decision": "fast",
                "active_experts": [],
                "slots": slots,
                "intent": "out_of_scope",
                "intent_confidence": confidence,
                "intent_reasoning": reasoning,
                "final_response": (
                    "Sorry — I only answer questions about crop health, plant diseases, and pesticide use "
                    "aligned with this assistant’s scope.\n\n"
                    "If you have a field diagnosis, IPM, or pesticide-registration question, ask in English "
                    "and I will do my best."
                ),
            }

        print(
            f"   • reasoning: {reasoning [:100 ]}{'...'if len (reasoning )>100 else ''}"
        )
        return {
            "complexity_score": s_complexity,
            "complexity_level": level,
            "route_decision": route,
            "active_experts": active_experts,
            "slots": slots,
            "intent": internal_intent,
            "intent_confidence": confidence,
            "intent_reasoning": reasoning,
        }

    def _route_decision_node(self, state: BlackboardState) -> BlackboardState:
        route = state.get("route_decision", "expert")
        active = state.get("active_experts", [])
        route_map = {
            "fast": "快速通道 - 通才智能体",
            "expert": "专家通道 - 动态激活专家",
        }
        print(f"\n[路由] {route_map .get (route ,route )}")
        print(
            f"   复杂度: {state .get ('complexity_level','unknown')} | 意图: {state .get ('intent','unknown')}"
        )
        if route == "expert":
            print(f"   激活专家: {active }")
        return {}

    def _route_condition(self, state: BlackboardState) -> str:
        route = state["route_decision"]
        print(f"[调试] 路由条件判断: {route }")
        return route

    def _post_intent_condition(self, state: BlackboardState) -> str:
        if state.get("final_response"):
            return "early"
        if state.get("intent") == "out_of_scope":
            return "early"
        return "route"

    def _early_return_node(self, state: BlackboardState) -> BlackboardState:
        return {
            "final_response": state.get("final_response"),
        }

    def _generalist_node(self, state: BlackboardState) -> BlackboardState:
        print(f"\n[通才智能体] 快速通道独立处理")
        print(
            f"   • 查询: {state ['query'][:60 ]}..."
            if len(state["query"]) > 60
            else f"   • 查询: {state ['query']}"
        )

        response_data = self.generalist.respond(state)

        if response_data.get("tool_results"):
            tool_names = [tr["tool_name"] for tr in response_data["tool_results"]]
            print(f"   • 调用工具: {', '.join (tool_names )}")

        final_response = response_data.get("content", "")

        print(f"   ✓ 快速通道完成")

        return {
            "reasoning_log": [response_data],
            "final_response": final_response,
        }

    def _expert_selection_node(self, state: BlackboardState) -> BlackboardState:
        experts = state.get("active_experts", [])

        expert_map = {
            "pathologist": "病理学家 - 症状分析与疾病诊断",
            "physiologist": "生理学家 - 环境因子审计",
            "chemist": "化学家 - 农药处方与合规验证",
        }

        print(f"[选定专家团队]:")
        for expert in experts:
            print(f"   • {expert_map .get (expert ,expert )}")
        print(f"[协作策略]: {len (experts )}个专家参与")

        return {"active_experts": experts}

    def _pathologist_node(self, state: BlackboardState) -> BlackboardState:
        print(f"\n[病理学家] 症状分析与病原体识别")
        print(f"   • 可用工具: 图像分类、病害分割、文献检索")

        analysis = self.pathologist.analyze(state)

        if analysis.get("tool_results"):
            tool_names = [tr["tool_name"] for tr in analysis["tool_results"]]
            print(f"   • 调用工具: {', '.join (tool_names )}")

        disease = analysis.get("disease", "未确定")
        confidence = analysis.get("confidence", 0.0)
        print(f"   ✓ 病理诊断完成 → 诊断: {disease } (置信度 {confidence *100 :.0f}%)")

        current_slots = state.get("expert_slots", {}) or {}
        return {
            "reasoning_log": [analysis],
            "visual_results": {
                **state.get("visual_results", {}),
                "pathologist_analysis": analysis,
            },
            "expert_slots": {**current_slots, "pathologist": analysis},
        }

    def _physiologist_node(self, state: BlackboardState) -> BlackboardState:
        print(f"\n[生理学家] 环境因子审计")
        print(f"   • 可用工具: 文献检索")

        expert_slots = state.get("expert_slots", {}) or {}
        pathologist_slot = expert_slots.get("pathologist") or {}
        if not pathologist_slot:
            for entry in state.get("reasoning_log", []):
                if entry.get("agent") == "pathologist":
                    pathologist_slot = entry
                    break

        if pathologist_slot.get("disease"):
            print(
                f"   📋 读取黑板 pathologist 插槽 → 目标验证: {pathologist_slot .get ('disease')}"
            )

        audit = self.physiologist.audit(state)

        if audit.get("tool_results"):
            tool_names = [tr["tool_name"] for tr in audit["tool_results"]]
            print(f"   • 调用工具: {', '.join (tool_names )}")

        env_fit = audit.get("environment_fit")
        veto = audit.get("veto", [])
        conflict = audit.get("conflict_with_pathologist", False)
        conflict_reason = audit.get("conflict_reason", "")
        print(f"   ✓ 环境审计完成 → 环境支持: {env_fit }, 否决列表: {veto or '无'}")
        if conflict:
            alt = (audit.get("alternative_hypothesis") or {}).get(
                "diagnosis", "unknown"
            )
            print(f"   ⚡ 冲突检测: conflict_with_pathologist=True → 替代假设: {alt }")
            print(f"      原因: {conflict_reason or '(未说明)'}")

        return {
            "reasoning_log": [audit],
            "expert_slots": {**expert_slots, "physiologist": audit},
        }

    def _chemist_node(self, state: BlackboardState) -> BlackboardState:
        print(f"\n[化学家] 农药处方与合规验证")
        print(f"   • 可用工具: 知识图谱查询、文献检索")

        expert_slots = state.get("expert_slots", {}) or {}
        pathologist_slot = expert_slots.get("pathologist", {})
        physiologist_slot = expert_slots.get("physiologist", {})

        if pathologist_slot.get("disease"):
            confidence = pathologist_slot.get("confidence", 0)
            rebuttal_info = pathologist_slot.get("rebuttal") or {}
            revised = pathologist_slot.get("original_disease") and (
                pathologist_slot.get("original_disease")
                != pathologist_slot.get("disease")
            )
            suffix = ""
            if revised:
                suffix = f", ⚠️讨论后修正自 {pathologist_slot ['original_disease']}"
            elif rebuttal_info.get("accepted_challenges"):
                suffix = (
                    f", 经讨论 accepted={len (rebuttal_info ['accepted_challenges'])}"
                )
            print(
                f"   📋 读取黑板 → 诊断: {pathologist_slot ['disease']} "
                f"(置信度 {confidence *100 :.0f}%{suffix })"
            )
        if physiologist_slot.get("veto"):
            print(f"   📋 读取黑板 → 生理学家否决列表: {physiologist_slot ['veto']}")
        if physiologist_slot.get("challenges"):
            print(
                f"   📋 读取黑板 → 生理学家质疑数: {len (physiologist_slot ['challenges'])}"
            )

        slots_ref = state.get("slots", {}) or {}
        preproc_hypotheses = slots_ref.get("validated_hypotheses") or []
        hypotheses = _derive_validated_hypotheses(
            pathologist_slot, physiologist_slot, fallback=preproc_hypotheses
        )
        if hypotheses and hypotheses != preproc_hypotheses:
            state = {
                **state,
                "slots": {**slots_ref, "validated_hypotheses": hypotheses},
            }
            print(f"   🎯 Validated hypotheses for KG probe: {hypotheses }")

        prescription = self.chemist.prescribe(state)

        if prescription.get("tool_results"):
            tool_names = [tr["tool_name"] for tr in prescription["tool_results"]]
            print(f"   • 调用工具: {', '.join (tool_names )}")

        rx_count = len(prescription.get("prescriptions", []) or [])
        violations = prescription.get("safety_violations", [])
        if violations:
            print(f"   ⚠ 安全问题: {len (violations )} 个")
        else:
            print(
                f"   ✓ 农药处方完成 → {rx_count } 个方案，合规: {'是'if prescription .get ('compliance_verified')else '待验证'}"
            )

        return {
            "reasoning_log": [prescription],
            "kg_results": state.get("kg_results", [])
            + [r["result"] for r in prescription.get("tool_results", [])],
            "expert_slots": {**expert_slots, "chemist": prescription},
            "compliance_verified": bool(prescription.get("compliance_verified", False)),
        }

    def _rebuttal_node(self, state: BlackboardState) -> BlackboardState:
        expert_slots = state.get("expert_slots", {}) or {}
        pathologist_slot = dict(expert_slots.get("pathologist", {}) or {})
        physiologist_slot = dict(expert_slots.get("physiologist", {}) or {})

        challenges = [
            c
            for c in (physiologist_slot.get("challenges") or [])
            if isinstance(c, str) and c.strip()
        ]
        veto_list = [
            v
            for v in (physiologist_slot.get("veto") or [])
            if isinstance(v, str) and v.strip()
        ]
        conflict_flag = bool(physiologist_slot.get("conflict_with_pathologist", False))
        conflict_reason = physiologist_slot.get("conflict_reason") or ""
        alt_hypothesis = physiologist_slot.get("alternative_hypothesis") or {}

        if not challenges and not veto_list and not conflict_flag:
            print(
                f"\n[rebuttal] no challenges, no veto, no conflict — short-circuit, diagnosis stands"
            )
            return {"has_conflicts": False}

        rebuttal_mode = "B" if conflict_flag else "A"
        print(
            f"\n[rebuttal] MODE {rebuttal_mode } | {len (challenges )} challenge(s), "
            f"{len (veto_list )} veto, conflict={conflict_flag } — invoking rebuttal LLM"
        )

        original_disease = pathologist_slot.get("disease") or "unknown"
        original_confidence = float(pathologist_slot.get("confidence", 0.0) or 0.0)
        original_differential = pathologist_slot.get("differential", []) or []
        original_reasoning = _short_text(
            pathologist_slot.get("reasoning", ""), max_len=600
        )
        physio_reasoning = _short_text(
            physiologist_slot.get("reasoning", ""), max_len=500
        )

        user_parts = [
            "Your previous diagnosis (on the blackboard):",
            f"- disease: {original_disease }",
            f"- confidence: {original_confidence :.2f}",
            f"- differential: {', '.join (original_differential )if original_differential else '(none)'}",
            f"- reasoning (abridged): {original_reasoning or '(none)'}",
            "",
            "Physiologist's environmental audit:",
            f"- reasoning (abridged): {physio_reasoning or '(none)'}",
        ]

        if challenges:
            user_parts.append(
                "- challenges (respond to EACH one — accept or reject with a pathology-grounded reason):"
            )
            for idx, ch in enumerate(challenges, start=1):
                user_parts.append(f"  {idx }. {ch }")

        if veto_list:
            user_parts.append(
                f"- veto (pathogens deemed environmentally impossible): {', '.join (veto_list )}"
            )

        if rebuttal_mode == "B":
            user_parts += [
                "",
                f"⚡ CONFLICT DETECTED (rebuttal_mode = B):",
                f"The physiologist has proposed a competing primary diagnosis that directly challenges yours.",
                f"Conflict reason: {conflict_reason or '(see physiologist reasoning above)'}",
            ]
            if alt_hypothesis.get("diagnosis"):
                user_parts += [
                    f"Alternative hypothesis: {alt_hypothesis ['diagnosis']}",
                    f"  Evidence: {'; '.join (alt_hypothesis .get ('evidence',[]))or '(see reasoning)'}",
                    f"  Confidence: {alt_hypothesis .get ('confidence',0 ):.2f}",
                    f"  Recommended action: {alt_hypothesis .get ('recommended_action','(not specified)')}",
                    "",
                    "You MUST address this alternative hypothesis directly:",
                    "1. Acknowledge it by name.",
                    "2. Argue why your pathological evidence (lesion morphology, distribution, tool data) supports YOUR diagnosis over theirs.",
                    "3. State the KEY DISCRIMINATING FACTOR — the one specific data point or test that would definitively resolve this conflict.",
                    "4. If their evidence is stronger, concede and revise your diagnosis accordingly.",
                ]
            user_parts.append("")
            user_parts.append(f"Set rebuttal_mode = 'B' in your output.")

        user_parts.append("")
        user_parts.append("Produce the RebuttalOutput JSON now.")

        messages = [
            SystemMessage(content=REBUTTAL_SYSTEM),
            HumanMessage(content="\n".join(user_parts)),
        ]

        try:
            response = self.coordinator.base_llm.invoke(
                messages,
                config=self.coordinator._run_config(override_agent="pathologist"),
            )
            raw_text = getattr(response, "content", "") or ""
        except Exception as exc:
            print(f"   ⚠ rebuttal LLM call failed: {exc }")
            return {"has_conflicts": False}

        parsed = parse_or_fallback(RebuttalOutput, raw_text)

        if not parsed.get("parsed_ok", True):
            print(
                f"   ⚠ rebuttal output could not be parsed — keeping original diagnosis"
            )
            failed_entry = {
                "agent": "pathologist",
                "stage": "rebuttal",
                "parsed_ok": False,
                "raw_text": raw_text,
                "run_id": state.get("run_id"),
                "timestamp": datetime.now().isoformat(),
            }
            return {"reasoning_log": [failed_entry], "has_conflicts": False}

        accepted = [
            c
            for c in (parsed.get("accepted_challenges") or [])
            if isinstance(c, str) and c.strip()
        ]
        rejected = [
            c
            for c in (parsed.get("rejected_challenges") or [])
            if isinstance(c, str) and c.strip()
        ]
        key_factor = parsed.get("key_discriminating_factor")
        mode_out = parsed.get("rebuttal_mode", rebuttal_mode)

        revised_disease = parsed.get("revised_disease") or original_disease
        revised_confidence = parsed.get("revised_confidence")
        if revised_confidence is None:
            revised_confidence = original_confidence
        revised_differential = (
            parsed.get("revised_differential") or original_differential
        )

        pathologist_slot["original_disease"] = original_disease
        pathologist_slot["original_confidence"] = original_confidence
        pathologist_slot["disease"] = revised_disease
        pathologist_slot["confidence"] = round(float(revised_confidence), 3)
        pathologist_slot["differential"] = revised_differential
        pathologist_slot["rebuttal"] = {
            "rebuttal_mode": mode_out,
            "rebuttal_reasoning": parsed.get("rebuttal_reasoning", ""),
            "key_discriminating_factor": key_factor,
            "accepted_challenges": accepted,
            "rejected_challenges": rejected,
            "veto_considered": veto_list,
        }

        has_conflicts = (
            bool(accepted) or (revised_disease != original_disease) or conflict_flag
        )

        new_conflicts: List[str] = []
        if conflict_flag:
            alt_name = alt_hypothesis.get("diagnosis", "unknown alternative")
            new_conflicts.append(
                f"MODE B conflict: physiologist proposed '{alt_name }' vs pathologist '{original_disease }'"
            )
            if key_factor:
                new_conflicts.append(f"key discriminating factor: {key_factor }")
        if revised_disease != original_disease:
            new_conflicts.append(
                f"rebuttal revised disease: {original_disease } → {revised_disease }"
            )
        for ch in accepted:
            new_conflicts.append(f"accepted challenge: {ch }")

        if new_conflicts:
            pathologist_slot["conflicts"] = (
                list(pathologist_slot.get("conflicts", [])) + new_conflicts
            )
        updated_conflict_details = (
            list(state.get("conflict_details", [])) + new_conflicts
        )

        print(
            f"   -> rebuttal MODE {mode_out }: accepted {len (accepted )}, rejected {len (rejected )}; "
            f"confidence {original_confidence :.2f} → {pathologist_slot ['confidence']:.2f}"
            + (
                f"; disease revised → {revised_disease }"
                if revised_disease != original_disease
                else ""
            )
        )
        if key_factor:
            print(f"   -> key discriminating factor: {key_factor [:120 ]}")

        rebuttal_log_entry = {
            "agent": "pathologist",
            "stage": "rebuttal",
            "rebuttal_mode": mode_out,
            "final_output": raw_text,
            "reasoning": parsed.get("rebuttal_reasoning", ""),
            "key_discriminating_factor": key_factor,
            "accepted_challenges": accepted,
            "rejected_challenges": rejected,
            "revised_disease": revised_disease,
            "revised_confidence": pathologist_slot["confidence"],
            "original_disease": original_disease,
            "original_confidence": original_confidence,
            "parsed_ok": True,
            "run_id": state.get("run_id"),
            "timestamp": datetime.now().isoformat(),
        }

        return {
            "expert_slots": {**expert_slots, "pathologist": pathologist_slot},
            "has_conflicts": has_conflicts,
            "conflict_details": updated_conflict_details,
            "reasoning_log": [rebuttal_log_entry],
        }

    def _synthesize_node(self, state: BlackboardState) -> BlackboardState:
        expert_slots = state.get("expert_slots", {}) or {}
        active_experts = state.get("active_experts", [])
        has_veto = bool(state.get("has_conflicts", False))
        conflicts = state.get("conflict_details", [])

        expert_display = {
            "pathologist": "pathologist",
            "physiologist": "physiologist",
            "chemist": "chemist",
        }

        print(f"\n[synthesize] building final report from blackboard...")
        print(f"   experts: {[expert_display .get (e ,e )for e in active_experts ]}")
        for name in active_experts:
            slot = expert_slots.get(name, {})
            if slot:
                print(f"   ✓ {expert_display .get (name ,name )} slot ready")
        if has_veto:
            print(f"   ⚠️ veto / conflict notes: {conflicts }")

        compact_mode = _env_flag("CERES_SYNTHESIS_COMPACT", default=True)
        synthesized_log = []
        for name in ["pathologist", "physiologist", "chemist"]:
            slot = expert_slots.get(name)
            if not slot or not slot.get("parsed_ok", True):
                continue
            if compact_mode:
                synthesized_log.append(_compact_expert_slot_for_synthesis(name, slot))
            else:
                synthesized_log.append({"agent": name, **slot})

        references_pool = _collect_references_from_slots(expert_slots)
        if references_pool:
            print(f"   📖 references pool: {len (references_pool )} source(s)")

        state_for_synthesis = {
            **state,
            "reasoning_log": synthesized_log,
            "references_pool": references_pool,
        }
        response = self.coordinator.synthesize_response(state_for_synthesis)

        print(f"   final report length: {len (response )} chars")
        return {"final_response": response}

    def diagnose(
        self,
        query: str,
        image_path: Optional[str] = None,
        location: Optional[str] = None,
        history: Optional[list] = None,
        debug: bool = True,
    ) -> str:
        effective_debug = debug and self.verbose
        if effective_debug:
            print(f"\n[CeresAgents] diagnosis start")
            print(f"query: {query }")
            print(f"image: {image_path or 'none'}")
            print(f"location: {location or 'none'}")
            print(f"\n" + "=" * 60)
        run_id = str(uuid4())

        initial_state = BlackboardState(
            query=query,
            image_path=image_path,
            location=location,
            history_context=history or [],
            run_id=run_id,
            messages=[],
            reasoning_log=[],
            slots=DiagnosticSlots(
                crop=None,
                symptoms=[],
                environment={},
                history=[],
                validated_hypotheses=[],
                recommendations=[],
            ),
            expert_slots={"pathologist": {}, "physiologist": {}, "chemist": {}},
            complexity_score=0.0,
            complexity_level="medium",
            route_decision="fast",
            intent="diagnosis",
            intent_confidence=0.0,
            intent_reasoning="",
            active_experts=[],
            has_conflicts=False,
            conflict_details=[],
            visual_results={},
            kg_results=[],
            rag_results=[],
            web_results=[],
            final_response=None,
            compliance_verified=False,
        )

        thread_id = f"fresh_{run_id }"
        config = {"configurable": {"thread_id": thread_id}}

        if effective_debug:
            print(f"[会话管理] 使用新线程ID: {thread_id }")
            print(f"[执行ID] run_id: {run_id }")

        self.token_tracker.reset()

        with _suppress_print(effective_debug):
            result = self.graph.invoke(initial_state, config)

        self.last_state = result
        self.last_structured_log = self.build_structured_log(result)

        final_response = result.get("final_response", "无法生成诊断报告")

        if effective_debug:
            print(f"\n" + "=" * 60)
            print(f"[CeresAgents] 诊断完成")
            print(f"路由决策: {result .get ('route_decision','unknown')}")
            print(f"用户意图: {result .get ('intent','unknown')}")
            print(f"活跃专家: {', '.join (result .get ('active_experts',[]))}")
            print(f"推理日志条目: {len (result .get ('reasoning_log',[]))}")
            print(f"\n最终报告:\n{final_response }")

        return final_response

    def get_last_state(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "last_state", None)

    def get_last_structured_log(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "last_structured_log", None)

    def run(self, input_data: dict) -> str:
        query = input_data.get("query", "")
        image_path = input_data.get("image_path")
        location = input_data.get("location")
        session_id = input_data.get("session_id")

        if session_id:
            self.set_session_id(session_id)

        result = self.diagnose(
            query=query, image_path=image_path, location=location, debug=True
        )

        return result

    def set_session_id(self, session_id: str):
        self._current_session_id = session_id

    def get_conversation_history(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not session_id:
            session_id = getattr(self, "_current_session_id", None)
        if not session_id:
            print("[WARN] 当前无会话记录可用")
            return []

        try:
            config = {"configurable": {"thread_id": session_id}}
            history = []

            return history
        except Exception as e:
            print(f"[WARN] 获取会话历史失败: {e }")
            return []

    def clear_conversation_history(self, session_id: Optional[str] = None):
        if not session_id:
            session_id = getattr(self, "_current_session_id", None)
        if not session_id:
            print("[WARN] 当前无会话记录可清理")
            return

        try:
            print(f"会话 {session_id } 的历史已清空")
        except Exception as e:
            print(f"[WARN] 清空会话历史失败: {e }")

    def stream_diagnose(
        self,
        query: str,
        image_path: Optional[str] = None,
        location: Optional[str] = None,
        history: Optional[list] = None,
        debug: bool = True,
    ):
        effective_debug = debug and self.verbose
        if effective_debug:
            print(f"\n[CeresAgents] 开始流式诊断")
            print(f"查询: {query }")
        run_id = str(uuid4())

        initial_state = BlackboardState(
            query=query,
            image_path=image_path,
            location=location,
            history_context=history or [],
            run_id=run_id,
            messages=[],
            reasoning_log=[],
            slots=DiagnosticSlots(
                crop=None,
                symptoms=[],
                environment={},
                history=[],
                validated_hypotheses=[],
                recommendations=[],
            ),
            complexity_score=0.0,
            complexity_level="medium",
            route_decision="fast",
            intent="diagnosis",
            intent_confidence=0.0,
            intent_reasoning="",
            active_experts=[],
            has_conflicts=False,
            visual_results={},
            kg_results=[],
            rag_results=[],
            web_results=[],
            final_response=None,
            compliance_verified=False,
            conflict_details=[],
            expert_slots={"pathologist": {}, "physiologist": {}, "chemist": {}},
        )
        thread_id = self._current_session_id or run_id
        config = {"configurable": {"thread_id": thread_id}}

        with _suppress_print(effective_debug):
            for event in self.graph.stream(initial_state, config):
                if effective_debug:
                    node_name = list(event.keys())[0] if event else "unknown"
                    print(f"\n[Stream Event] 节点: {node_name }")
                yield event


__all__ = [
    "BlackboardState",
    "DiagnosticSlots",
    "CoordinatorAgent",
    "GeneralistAgent",
    "PathologistAgent",
    "PhysiologistAgent",
    "ChemistAgent",
    "CeresAgentsGraph",
]

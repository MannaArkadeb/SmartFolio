from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from tools import explainability_llm_attention as base_attn

from .config import XAIRequest
from .registry import register_mcp_tool


def run_attention_narrative(cfg: XAIRequest, summary_path: Optional[Path] = None) -> Dict[str, object]:
    summary = summary_path or cfg.output_dir / f"attention_summary_{cfg.market}.json"
    if not summary.exists():
        raise FileNotFoundError(f"Attention summary not found at {summary}")

    summary_payload = base_attn.load_attention_summary(summary)
    prompt = base_attn.assemble_prompt(summary_payload)
    prompt_path = cfg.output_dir / "hgat_attention_prompt_mcp.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    if cfg.llm:
        text = base_attn.llm_generate(prompt, model=cfg.llm_model)
        used_llm = True
    else:
        text = f"Loaded HGAT attention summary for model: {summary_payload.get('model_path', 'N/A')}"
        used_llm = False

    out_path = cfg.output_dir / f"hgat_attention_narrative_{cfg.market}_mcp.md"
    out_path.write_text(text, encoding="utf-8")

    return {
        "summary_path": str(summary),
        "prompt_path": str(prompt_path),
        "markdown_path": str(out_path),
        "used_llm": used_llm,
        "preview": text.splitlines()[:10],
    }


ATTENTION_NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "monthly_log_csv": {"type": "string"},
        "model_path": {"type": "string"},
        "market": {"type": "string"},
        "data_root": {"type": "string"},
        "top_k": {"type": "integer"},
        "lookback_days": {"type": "integer"},
        "llm": {"type": "boolean"},
        "llm_model": {"type": "string"},
        "output_dir": {"type": "string"},
        "monthly_run_id": {"type": "string"},
        "summary_path": {"type": "string"},
    },
    "required": ["date", "monthly_log_csv", "model_path", "lookback_days", "top_k", "market", "data_root", "output_dir", "monthly_run_id", "llm", "llm_model"],
}


@register_mcp_tool(
    name="generate_attention_narrative",
    description="Create an LLM-based explanation from the HGAT attention summary.",
    schema=ATTENTION_NARRATIVE_SCHEMA,
)
def mcp_generate_attention_narrative(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    summary_override = payload.get("summary_path")
    summary = Path(summary_override).expanduser() if summary_override else None
    return run_attention_narrative(cfg, summary_path=summary)

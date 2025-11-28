from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from tools import explainability_llm_agent as base_agent

from .config import XAIRequest
from .registry import register_mcp_tool


def run_tree_narrative(
    cfg: XAIRequest,
    snapshot_path: Optional[Path] = None,
) -> Dict[str, object]:
    snapshot = snapshot_path or cfg.output_dir / f"explain_tree_{cfg.market}.joblib"
    if not snapshot.exists():
        raise FileNotFoundError(f"Tree snapshot not found at {snapshot}")

    context = base_agent.load_snapshot(snapshot)
    prompt = base_agent.assemble_prompt(context)
    prompt_path = cfg.output_dir / "input_tree_llm_prompt_mcp.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    if cfg.llm:
        narrative = base_agent.llm_narrative(prompt, model=cfg.llm_model)
        used_llm = True
    else:
        narrative = base_agent.fallback_narrative(context)
        used_llm = False

    out_path = cfg.output_dir / f"explainability_narrative_tree_{cfg.market}_mcp.md"
    out_path.write_text(narrative, encoding="utf-8")

    return {
        "snapshot_path": str(snapshot),
        "prompt_path": str(prompt_path),
        "markdown_path": str(out_path),
        "used_llm": used_llm,
        "preview": narrative.splitlines()[:10],
    }


TREE_NARRATIVE_SCHEMA = {
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
        "snapshot_path": {"type": "string"},
    },
    "required": ["date", "monthly_log_csv", "model_path", "lookback_days", "top_k", "market", "data_root", "output_dir", "monthly_run_id", "llm", "llm_model"],
}


@register_mcp_tool(
    name="generate_tree_narrative",
    description="Create a natural-language summary from the decision-tree surrogate payload.",
    schema=TREE_NARRATIVE_SCHEMA,
)
def mcp_generate_tree_narrative(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    snapshot_override = payload.get("snapshot_path")
    snapshot = Path(snapshot_override).expanduser() if snapshot_override else None
    return run_tree_narrative(cfg, snapshot_path=snapshot)

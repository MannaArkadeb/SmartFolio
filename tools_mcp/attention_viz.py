from __future__ import annotations

import json
from typing import Dict, List

from tools import attention_viz as base_attention

from .config import XAIRequest
from .registry import register_mcp_tool


def _build_args(cfg: XAIRequest, start_date: str, end_date: str) -> List[str]:
    return [
        "--model-path",
        cfg.model_path,
        "--market",
        cfg.market,
        "--test-start-date",
        start_date,
        "--test-end-date",
        end_date,
        "--data-root",
        cfg.data_root,
        "--device",
        "cpu",
        "--save-summary",
        "--save-raw",
        "--output-dir",
        str(cfg.output_dir),
    ]


def run_attention_job(cfg: XAIRequest) -> Dict[str, object]:
    start_date, end_date = cfg.extra.get("window", cfg.rolling_window())
    argv = _build_args(cfg, start_date, end_date)
    base_attention.main(argv)

    summary_path = cfg.output_dir / f"attention_summary_{cfg.market}.json"
    raw_path = cfg.output_dir / f"attention_tensors_{cfg.market}.npz"

    summary_payload = None
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "start_date": start_date,
        "end_date": end_date,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "raw_path": str(raw_path) if raw_path.exists() else None,
        "summary": summary_payload,
    }


@register_mcp_tool(
    name="generate_attention_summary",
    description="Run HGAT attention visualization over the lookback window and capture artifacts.",
)
def mcp_generate_attention_summary(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    return run_attention_job(cfg)

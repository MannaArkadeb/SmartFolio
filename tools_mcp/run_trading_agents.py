from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tools.orchestrator_xai import top_k_for_date_from_log
from tradingagents.combined_weight_agent import WeightSynthesisAgent

from .config import XAIRequest
from .registry import register_mcp_tool


def run_trading_job(cfg: XAIRequest) -> Dict[str, object]:
    holdings = top_k_for_date_from_log(
        Path(cfg.monthly_log_csv),
        cfg.date,
        top_k=cfg.top_k,
        run_id=cfg.monthly_run_id,
    )
    agent = WeightSynthesisAgent()
    reports: List[Dict[str, object]] = []

    for row in holdings:
        ticker = str(row.get("ticker", "?")).strip().upper()
        weight = float(row.get("weight", 0.0))
        as_of = row.get("as_of")
        try:
            report = agent.generate_report(
                ticker,
                weight,
                as_of=as_of,
                lookback_days=30,
                max_articles=8,
                use_llm=cfg.llm,
                llm_model=cfg.llm_model,
            )
            md_path = cfg.output_dir / f"{ticker}_summary_mcp.md"
            md_path.write_text(
                report.to_markdown(include_components=True, include_metrics=True, include_articles=True),
                encoding="utf-8",
            )
            reports.append(
                {
                    "ticker": ticker,
                    "success": True,
                    "weight": weight,
                    "as_of": as_of,
                    "output_path": str(md_path),
                    "summary_points": list(report.summary_points),
                    "llm_used": bool(report.generated_via_llm),
                }
            )
        except Exception as exc:  # noqa: BLE001
            reports.append(
                {
                    "ticker": ticker,
                    "success": False,
                    "weight": weight,
                    "as_of": as_of,
                    "error": str(exc),
                }
            )

    return {"holdings": holdings, "reports": reports}


@register_mcp_tool(
    name="run_trading_agent",
    description="Run the WeightSynthesisAgent for each focus holding and persist markdown reports.",
)
def mcp_run_trading_agent(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    return run_trading_job(cfg)

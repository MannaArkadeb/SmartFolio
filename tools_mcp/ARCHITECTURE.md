# MCP Server Plan for SmartFolio XAI

## Goals
1. Expose the explainability pipeline through a Model Context Protocol (MCP) server built on Pathway, so agentic clients can request analyses for specific snapshot dates.
2. Keep the original tooling logic (attention viz, explain_tree, explainability LLM agents, trading agent orchestration) but run it inside isolated MCP tools.
3. Provide granular MCP tools for individual artefacts (attention, tree, trading agent, final synthesis) as well as a high-level orchestrator tool that coordinates all of them.

## High-Level Components
| Component | Description | MCP Tool |
|-----------|-------------|----------|
| `pathway_server.py` | Bootstraps a Pathway-based MCP server, registers tool handlers, and manages request routing. | N/A (server entry point) |
| `xai_orchestrator_tool.py` | Wraps the orchestrator logic to run the full end-to-end explainability pass for a single date and top-K holdings. | `run_xai_orchestrator` |
| `attention_tool.py` | Runs `attention_viz` for a given date window and returns serialized summary data plus file paths. | `generate_attention_summary` |
| `tree_tool.py` | Executes `explain_tree` with focus tickers and returns surrogate payload locations. | `generate_tree_surrogate` |
| `narrative_tool.py` | Invokes both LLM narrators (tree + attention) and returns markdown paths. | `generate_xai_narratives` |
| `trading_agent_tool.py` | Calls `WeightSynthesisAgent` per ticker to fetch markdown + bullets. | `run_trading_agent` |
| `final_report_tool.py` | Consumes outputs from other tools and synthesizes final per-ticker summaries (LLM optional). | `synthesize_final_reports` |

All tools share a common configuration dataclass mirroring `OrchestratorConfig`, so arguments stay consistent.

## Request Flow
1. **Client** invokes `run_xai_orchestrator` with payload `{date, market, model_path, monthly_log_csv, ...}`.
2. **Server** validates the payload, then internally calls sub-tools in sequence:
   - `generate_attention_summary`
   - `generate_tree_surrogate`
   - `generate_xai_narratives`
   - `run_trading_agent`
   - `synthesize_final_reports`
3. Each step streams Pathway logs/metrics back via MCP progress events.
4. Final response returns artifacts list (markdown, JSON index, per-ticker narratives) and summary metadata.

## Pathway Integration
- Use `pathway`’s async event loop utilities to run blocking Python functions on worker threads.
- Leverage Pathway’s `@pw.io.mcp.tool` decorator to register synchronous tools that wrap the existing Python functions.
- Provide structured JSON schemas for inputs/outputs (Pydantic or dataclasses converted to dicts) to stay MCP-compliant.

## Deployment Notes
- `tools_mcp` will mirror the required files from `tools/` but may include MCP-specific wrappers.
- Environment variables (e.g., `GOOGLE_API_KEY`) continue to be read at runtime; MCP server just surfaces missing-key errors through tool errors.
- Future extension: add streaming tokens from LLM calls as MCP progress messages if needed.

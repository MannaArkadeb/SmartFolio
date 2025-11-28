from __future__ import annotations

from .config import XAIRequest
from .mcp_server import SmartFolioMCPServer, list_registered_tools

# Importing these modules registers their MCP handlers with the shared registry.
from . import attention_viz as _attention_viz  # noqa: F401
from . import explain_tree as _explain_tree  # noqa: F401
from . import explainability_llm_agent as _tree_llm  # noqa: F401
from . import explainability_llm_attention as _attn_llm  # noqa: F401
from . import run_trading_agents as _trading  # noqa: F401
from . import orchestrator_xai as _orchestrator  # noqa: F401

__all__ = [
	"SmartFolioMCPServer",
	"XAIRequest",
	"list_registered_tools",
]

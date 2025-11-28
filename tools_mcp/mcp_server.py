from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional

try:
    import pathway as pw  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    pw = None

from .registry import ToolDefinition, iter_tools


class SmartFolioMCPServer:
    """Thin wrapper around Pathway's MCP server facilities."""

    def __init__(self, app_id: str = "smartfolio-xai", host: str = "127.0.0.1", port: int = 9123):
        self.app_id = app_id
        self.host = host
        self.port = port

    def _load_pathway_mcp_primitives(self):
        if pw is None:
            raise ImportError(
                "Pathway is not installed. Install the official Pathway build to expose MCP tooling."
            )

        candidates = ("pathway.io.mcp", "pathway.mcp")
        last_err: Optional[Exception] = None
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
                return module
            except Exception as exc:  # pragma: no cover - runtime discovery
                last_err = exc
        raise ImportError(
            "Unable to import pathway MCP module. Make sure your Pathway distribution includes pathway.io.mcp"
        ) from last_err

    def serve(self) -> Any:
        module = self._load_pathway_mcp_primitives()
        server_cls = getattr(module, "Server", None) or getattr(module, "ServerApp", None)
        tool_cls = getattr(module, "Tool", None) or getattr(module, "ToolDefinition", None)
        if server_cls is None or tool_cls is None:
            raise RuntimeError("Pathway MCP module does not expose Server/Tool primitives")

        tool_specs = []
        for definition in iter_tools():
            try:
                tool_spec = tool_cls(
                    name=definition.name,
                    description=definition.description,
                    handler=definition.handler,
                    schema=definition.schema,
                )
            except TypeError:
                # Some builds expect callable supplied separately
                tool_spec = tool_cls(
                    name=definition.name,
                    description=definition.description,
                    schema=definition.schema,
                )
                tool_spec.handler = definition.handler  # type: ignore[attr-defined]
            tool_specs.append(tool_spec)

        server = server_cls(app_id=self.app_id, host=self.host, port=self.port, tools=tool_specs)
        return server.run()  # type: ignore[no-any-return]


def list_registered_tools() -> Iterable[Dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.schema,
        }
        for tool in iter_tools()
    ]

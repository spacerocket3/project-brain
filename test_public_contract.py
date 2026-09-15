import importlib
import os
import unittest
from unittest.mock import patch

import mcp_server as mcp


class PublicContractTests(unittest.TestCase):
    def test_default_mcp_surface_is_cartography_only(self):
        names = {tool["name"] for tool in mcp.TOOLS}
        self.assertEqual(names, {
            "project_brain_status",
            "search_repository",
            "inspect_symbol",
            "investigate_flow",
            "diagnose_retrieval",
            "load_deep_context",
            "refresh_repository",
            "refresh_embeddings",
        })
        self.assertFalse(any(name.startswith("consult_") for name in names))

    def test_generation_tools_can_be_selected_only_by_explicit_opt_in(self):
        with patch.dict(
            os.environ, {"PROJECT_BRAIN_ENABLE_GENERATION_TOOLS": "1"}, clear=False,
        ):
            opted_in = importlib.reload(mcp)
            self.assertIn("consult_nemotron", {tool["name"] for tool in opted_in.TOOLS})
        importlib.reload(mcp)

    def test_status_does_not_start_or_query_a_generation_model(self):
        with patch.object(mcp.brain, "status", return_value={"projects": []}) as status:
            self.assertEqual(
                mcp.call_tool("project_brain_status", {}),
                {"projects": [], "scope_notes": {}},
            )
        status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

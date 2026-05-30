"""Regression tests for root-level AI context file policy."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_hermes_marketing_doc_does_not_match_agent_context_filename():
    """Hermes Agent auto-loads root HERMES.md/.hermes.md into prompts; keep the long human doc outside the root context names."""
    assert not (ROOT / "HERMES.md").exists()
    assert (ROOT / "docs" / "why-hermes.md").is_file()

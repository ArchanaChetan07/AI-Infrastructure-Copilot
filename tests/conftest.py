"""
pytest conftest.py — shared fixtures and hooks for all test suites.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: requires ANTHROPIC_API_KEY")
    config.addinivalue_line("markers", "slow: runs all 3 scenarios end-to-end")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a helpful summary after integration tests."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    if skipped > 0:
        terminalreporter.write_sep(
            "-",
            f"{skipped} test(s) skipped — set ANTHROPIC_API_KEY in .env to run integration tests",
        )

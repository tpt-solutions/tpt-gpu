"""Tests for Rust dispatch module (tptr-core dispatch tests).

These tests verify the Rust dispatch functionality.
They run via cargo test on the tptr-core crate.
"""
import subprocess
import os
import shutil
import pytest


def run_cargo_test(package=None, test_name=None):
    """Run a cargo test command and return the result."""
    cmd = ["cargo", "test", "-p", package or "tptr-core"]
    if test_name:
        cmd.append(test_name)
    cmd.append("--")
    cmd.append("--nocapture")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "..", "layer4_tptr"),
    )
    return result


class TestRustDispatch:
    """Tests for Rust dispatch module - requires cargo."""

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_dispatch_batch_compiles(self):
        """Verify dispatch batch module compiles."""
        result = run_cargo_test("tptr-core", "dispatch::batch::tests")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_dispatch_pool_compiles(self):
        """Verify dispatch pool module compiles."""
        result = run_cargo_test("tptr-core", "dispatch::pool::tests")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_dispatch_ops_compiles(self):
        """Verify dispatch ops module compiles."""
        result = run_cargo_test("tptr-core", "dispatch::ops::tests")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_all_core_tests_pass(self):
        """Run all tptr-core tests."""
        result = run_cargo_test("tptr-core")
        assert result.returncode == 0, f"stderr: {result.stderr}"


import shutil


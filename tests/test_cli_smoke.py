#!/usr/bin/env python3
"""
test_cli_smoke.py
================
Smoke tests for CLI subcommands to ensure they can run --help without errors.

These tests verify that the CLI is properly structured and can display help
without crashing, which is essential for user experience. Also includes
visualization tests to ensure --save-visualizations creates output files.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class TestCLISmoke:
    """Smoke tests for CLI scripts."""
    
    def test_cli_main_help(self):
        """Test that the main CLI shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"CLI help failed: {result.stderr}"
        assert "train" in result.stdout
        assert "eval" in result.stdout
        assert "benchmark-attn" in result.stdout
    
    def test_cli_train_help(self):
        """Test that the train subcommand shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "train", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Train help failed: {result.stderr}"
        assert "--data-root" in result.stdout
        assert "--epochs" in result.stdout
        assert "--arch" in result.stdout
    
    def test_cli_eval_help(self):
        """Test that the eval subcommand shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "eval", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Eval help failed: {result.stderr}"
        assert "--mode" in result.stdout
        assert "single" in result.stdout
        assert "ensemble" in result.stdout
        assert "--data-root" in result.stdout
    
    def test_cli_benchmark_help(self):
        """Test that the benchmark-attn subcommand shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "benchmark-attn", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Benchmark help failed: {result.stderr}"
        assert "--attention-types" in result.stdout
        assert "--save-visualizations" in result.stdout
    
    def test_unified_eval_help(self):
        """Test that the unified eval script shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "eval.py"), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Unified eval help failed: {result.stderr}"
        assert "--mode" in result.stdout
        assert "single" in result.stdout
        assert "ensemble" in result.stdout
    
    def test_benchmark_script_help(self):
        """Test that the benchmark script shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "benchmark_p2_attention.py"), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Benchmark script help failed: {result.stderr}"
        assert "--attention-types" in result.stdout
        assert "--save-visualizations" in result.stdout
    
    def test_generate_dataset_help(self):
        """Test that the dataset generator shows help without error."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "generate_dataset.py"), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Dataset generator help failed: {result.stderr}"
        assert "--config" in result.stdout
    
    def test_eval_mode_options(self):
        """Test that eval help shows mode options correctly."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "eval", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Eval help failed: {result.stderr}"
        assert "--mode" in result.stdout
        assert "single" in result.stdout
        assert "ensemble" in result.stdout
    
    def test_benchmark_save_visualizations_help(self):
        """Test that benchmark help shows --save-visualizations option."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "benchmark_p2_attention.py"), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Benchmark help failed: {result.stderr}"
        assert "--save-visualizations" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI error handling."""
    
    def test_cli_invalid_command(self):
        """Test that invalid commands are handled gracefully."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "invalid-command"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_eval_mode_validation(self):
        """Test that eval mode validation works."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "eval.py"), 
            "--mode", "invalid-mode", "--data-root", "dummy", "--weights", "dummy"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        # Should fail at argument parsing level
    
    def test_missing_required_args(self):
        """Test that missing required arguments are caught."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "eval"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


class TestVisualizationOutput:
    """Test that visualization features create output files."""
    
    def test_benchmark_visualization_creates_files(self):
        """Test that running benchmark with --save-visualizations creates output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            viz_dir = Path(temp_dir) / "test_viz"
            
            # Run a minimal benchmark with visualization
            result = subprocess.run([
                sys.executable, str(SCRIPTS_DIR / "benchmark_p2_attention.py"),
                "--attention-types", "arc_aware",
                "--num-samples", "10",  # Very small for speed
                "--batch-size", "2",
                "--save-visualizations", str(viz_dir),
                "--output-dir", temp_dir,
                "--data-root", "dummy_data"  # Will create synthetic data
            ], capture_output=True, text=True, timeout=60)
            
            # The benchmark should run (may fail due to missing data, but should create viz dir)
            # Check if visualization directory was created
            assert viz_dir.exists(), "Visualization directory should be created"
            
            # Check if any files were created in the visualization directory
            viz_files = list(viz_dir.glob("*"))
            assert len(viz_files) > 0, f"At least one visualization file should be created, found: {viz_files}"
            
            # Check for expected file patterns
            expected_patterns = ["*arc_aware*", "*.png", "*.txt"]
            found_pattern = False
            for pattern in expected_patterns:
                if list(viz_dir.glob(pattern)):
                    found_pattern = True
                    break
            
            assert found_pattern, f"Should find files matching expected patterns in {viz_files}"
    
    def test_cli_benchmark_dry_run_with_visualizations(self):
        """Test that CLI benchmark dry-run works with visualization flag."""
        result = subprocess.run([
            sys.executable, str(SCRIPTS_DIR / "cli.py"), "benchmark-attn",
            "--attention-types", "arc_aware",
            "--save-visualizations", "test_output",
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Benchmark dry-run failed: {result.stderr}"
        assert "DRY RUN MODE" in result.stdout or "save_visualizations" in result.stdout


if __name__ == "__main__":
    # Run the tests directly if script is executed
    pytest.main([__file__, "-v"])

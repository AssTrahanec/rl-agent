import subprocess
import sys


def test_run_improved_smoke():
    """Smoke test: run_improved with dummy data."""
    result = subprocess.run(
        [sys.executable, "experiments/run_improved.py",
         "--dummy", "--total-timesteps", "1000",
         "--seeds", "42"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    assert "DONE" in result.stdout

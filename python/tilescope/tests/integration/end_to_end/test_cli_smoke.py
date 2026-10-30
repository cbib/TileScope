"""One CLI execution through the installed entry point, in a subprocess.

The API smoke test calls `run_tilescope_from_csv` directly and so never
exercises `tilescope.tilescope:main`. This test runs the CLI the way a
user does — with no verbosity flag, so the config-driven logging branch
is taken — and requires a zero exit status after a complete run.
"""
import subprocess
import sys

from tilescope.tests.helpers.paths import FIXTURES_DIR


def test_cli_default_verbosity_completes_with_exit_zero(tmp_path):
    out = tmp_path / "cli-smoke"
    csv = FIXTURES_DIR / "grid_square_hole_and_circle.csv"
    proc = subprocess.run(
        [
            sys.executable, "-m", "tilescope.tilescope", str(csv),
            "--profile", "debugging",
            "--output", str(out),
            "--feature-a-id", "Neuron",
            "--feature-b-id", "AC",
        ],
        capture_output=True, text=True, timeout=600,
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"CLI exited {proc.returncode}:\n{log[-3000:]}"
    assert (out / "roi_enrichment.csv").is_file()
    assert (out / "effective_config.json").is_file()
    assert f"Results directory: {out.resolve()}" in log

"""Run the evaluation gate end to end on a synthetic population.

CI has no access to the frozen data populations (they are local-only,
verified by checksum manifests), so this script builds the synthetic
gate population from the test factories, trains and registers a model in
a throwaway workspace, and runs the real gate CLI against it. The gate's
exit code propagates: a failing verdict fails the CI job, and the
report lands at the path given by --report for artifact upload.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from factories import write_gate_population  # noqa: E402 (needs the tests path above)
from risk_scoring import gate, train  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_gate_ci.py",
        description="Train on the synthetic gate population and run the gate CLI.",
    )
    parser.add_argument("--report", type=Path, default=Path("gate_report.md"))
    args = parser.parse_args(argv)
    report_path = args.report.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        csv_dir = workspace / "data" / "baseline" / "csv"
        write_gate_population(csv_dir)
        train.train(csv_dir, workspace)
        os.chdir(workspace)
        gate.main(["run", "--report", str(report_path)])


if __name__ == "__main__":
    main()

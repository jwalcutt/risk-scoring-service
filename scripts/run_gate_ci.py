"""Run the chained retrain on a synthetic population, for CI.

CI has no access to the frozen data populations (they are local-only,
verified by checksum manifests), so this script builds the synthetic
gate population from the test factories and runs the same retrain
entrypoint an operator runs on real data, in a throwaway workspace. The
gate's verdict propagates to the exit code: a failing verdict fails the
CI job, and the report lands at the path given by --report for artifact
upload.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from factories import write_gate_population  # noqa: E402 (needs the tests path above)
from risk_scoring import pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_gate_ci.py",
        description="Retrain and gate on the synthetic gate population.",
    )
    parser.add_argument("--report", type=Path, default=Path("gate_report.md"))
    args = parser.parse_args(argv)
    report_path = args.report.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        csv_dir = workspace / "data" / "baseline" / "csv"
        write_gate_population(csv_dir)
        outcome = pipeline.retrain(csv_dir, workspace, report_path=report_path)
        print()
        print(outcome.gate.report)
        if outcome.gate.result.verdict != "pass":
            sys.exit(1)


if __name__ == "__main__":
    main()

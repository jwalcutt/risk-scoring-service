"""Serve the scoring service against the local registry.

Usage (from the repo root):
    python -m risk_scoring.service run [--host HOST] [--port PORT] [--config PATH]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn

from risk_scoring.service.app import create_app
from risk_scoring.service.config import DEFAULT_CONFIG_RELPATH, load_config


def main(argv: list[str] | None = None, runner: Callable[..., Any] = uvicorn.run) -> None:
    parser = argparse.ArgumentParser(prog="python -m risk_scoring.service")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="serve the scoring service against the local registry")
    run.add_argument("--port", type=int, default=8000)
    # Loopback by default: a host process should not be reachable off the
    # machine. The container's command asks for 0.0.0.0 explicitly.
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_RELPATH)

    args = parser.parse_args(argv)
    if args.command == "run":
        repo_root = Path.cwd()
        config = load_config(repo_root / args.config)
        app = create_app(config, repo_root)
        runner(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

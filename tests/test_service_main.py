"""Tests for the service CLI entrypoint.

The rules these tests pin:

- ``python -m risk_scoring.service run`` loads configs/service.toml from
  the working directory, builds the app from it with the working
  directory as the repo root, and hands the app to the server runner
  with the requested port.
- The runner is injectable, so the CLI is testable without binding a
  socket; a missing config file fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

import risk_scoring.service.__main__ as service_main
from risk_scoring.service.config import ServiceConfig


def _write_config(root: Path, version: int = 3) -> None:
    (root / "configs").mkdir()
    (root / "configs" / "service.toml").write_text(
        f'[model]\nname = "readmission-risk"\nversion = {version}\n'
    )


def test_run_builds_app_from_cwd_config_and_hands_it_to_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, version=7)
    monkeypatch.chdir(tmp_path)

    built: dict[str, Any] = {}
    sentinel = FastAPI()

    def fake_create_app(config: ServiceConfig, repo_root: Path) -> FastAPI:
        built["config"] = config
        built["repo_root"] = repo_root
        return sentinel

    served: dict[str, Any] = {}

    def fake_runner(app: FastAPI, **kwargs: Any) -> None:
        served["app"] = app
        served.update(kwargs)

    monkeypatch.setattr(service_main, "create_app", fake_create_app)
    service_main.main(["run", "--port", "9001"], runner=fake_runner)

    assert built["config"] == ServiceConfig(model_name="readmission-risk", model_version=7)
    assert built["repo_root"] == tmp_path
    assert served["app"] is sentinel
    assert served["port"] == 9001


def test_run_defaults_to_port_8000(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_main, "create_app", lambda config, repo_root: FastAPI())

    served: dict[str, Any] = {}

    def fake_runner(app: FastAPI, **kwargs: Any) -> None:
        served.update(kwargs)

    service_main.main(["run"], runner=fake_runner)
    assert served["port"] == 8000


def test_run_without_config_file_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        service_main.main(["run"], runner=lambda app, **kwargs: None)


def test_run_defaults_to_binding_the_loopback_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host process stays local; only the container asks for 0.0.0.0."""
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_main, "create_app", lambda config, repo_root: FastAPI())

    served: dict[str, Any] = {}
    service_main.main(["run"], runner=lambda app, **kwargs: served.update(kwargs))

    assert served["host"] == "127.0.0.1"


def test_run_binds_the_requested_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_main, "create_app", lambda config, repo_root: FastAPI())

    served: dict[str, Any] = {}
    service_main.main(
        ["run", "--host", "0.0.0.0"], runner=lambda app, **kwargs: served.update(kwargs)
    )

    assert served["host"] == "0.0.0.0"

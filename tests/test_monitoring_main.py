"""The monitoring commands as an operator types them.

The rules these tests pin:

- The model version defaults to the pin in configs/service.toml, so the
  reference and the model actually serving cannot silently disagree.
- A missing export or a missing config is refused with a sentence naming
  the remedy, never a traceback.
- Only the subcommands that exist are accepted, and the parser requires
  one. The run, evaluate, ack, and status commands are not built yet, so
  they must be refused rather than quietly doing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from risk_scoring.monitoring.__main__ import build_parser, main


def test_the_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_reference_defaults_the_version_to_the_pin_rather_than_a_literal() -> None:
    """A default of None is what lets the service config decide."""
    args = build_parser().parse_args(["reference"])
    assert args.command == "reference"
    assert args.model_version is None
    assert args.population == "baseline"
    assert args.config == Path("configs/service.toml")


def test_reference_takes_an_explicit_version_for_building_ahead_of_a_promotion() -> None:
    args = build_parser().parse_args(["reference", "--model-version", "7"])
    assert args.model_version == 7


@pytest.mark.parametrize("command", ["run", "evaluate", "ack", "status"])
def test_the_commands_that_do_not_exist_yet_are_refused(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_a_missing_service_config_names_the_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["reference"])
    assert "--model-version" in str(exit_info.value)


def test_a_missing_export_is_refused_before_the_database_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No connection is attempted, so this fails the same way with no Postgres."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["reference", "--model-version", "4", "--population", "nonexistent"])
    assert "generate the population first" in str(exit_info.value)

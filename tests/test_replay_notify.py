"""The desktop notifier, through its injected subprocess runner.

The rules these tests pin:

- On macOS a notification is one ``osascript`` call whose text is quoted
  for AppleScript, and a failure to display never fails the run.
- Anywhere else the notifier prints the text to stderr and runs nothing,
  so a run on a CI box or a Linux host records the pause all the same.
- No real desktop is ever touched by a test.
"""

from __future__ import annotations

from typing import Any

import pytest

from risk_scoring.replay import notify


def test_on_macos_the_notifier_calls_osascript_with_the_text_quoted() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: list[str], **kwargs: Any) -> None:
        calls.append((argv, kwargs))

    notifier = notify.desktop_notifier(platform="darwin", run=run)
    notifier('Replay paused at 2025-03-01T00:00:00Z ("attended")')

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:2] == ["osascript", "-e"]
    assert argv[2] == (
        'display notification "Replay paused at 2025-03-01T00:00:00Z (\\"attended\\")"'
        ' with title "Replay"'
    )
    assert kwargs["check"] is False


def test_elsewhere_the_notifier_prints_to_stderr_and_runs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []
    notifier = notify.desktop_notifier(platform="linux", run=lambda argv, **kw: calls.append(argv))

    notifier("Replay finished at 2026-01-01T00:00:00Z")

    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "notification: Replay finished at 2026-01-01T00:00:00Z\n"


def test_the_default_notifier_reads_the_running_platform() -> None:
    """The default arguments bind to the real platform and the real runner."""
    import subprocess
    import sys

    assert notify.desktop_notifier.__kwdefaults__["platform"] == sys.platform
    assert notify.desktop_notifier.__kwdefaults__["run"] is subprocess.run

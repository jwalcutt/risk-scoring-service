"""Telling the operator, on the desktop, that the clock stopped.

The harness is a host process partly so that a pause can reach the
desktop: the operating model is attended sessions, and a run that pauses
while the operator is in another window should say so where they are
looking. On macOS that is one ``osascript`` call and no new dependency.

Judgment calls this module fixes:

- The notifier is a value the commands take, built here from the platform
  and a subprocess runner. Tests assert the call and its text through the
  injected runner and never touch a real desktop.
- A notification that fails to display never fails the run. The run row
  and the printed summary are the record; the notification is a courtesy.
- Off macOS the text goes to stderr. A run on a CI box or a Linux host
  still records every pause in its output, and nothing is silently lost.
- The text is quoted for AppleScript through ``json.dumps``, whose string
  escaping AppleScript accepts, so a quote in the message cannot break the
  script.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

Notifier = Callable[[str], None]
"""Deliver one line of text to the operator."""

TITLE = "Replay"


def desktop_notifier(
    *, platform: str = sys.platform, run: Callable[..., Any] = subprocess.run
) -> Notifier:
    """A notifier for the platform: ``osascript`` on macOS, stderr elsewhere."""
    if platform == "darwin":

        def notify(text: str) -> None:
            script = f"display notification {json.dumps(text)} with title {json.dumps(TITLE)}"
            run(["osascript", "-e", script], check=False)

        return notify

    def log(text: str) -> None:
        print(f"notification: {text}", file=sys.stderr)

    return log

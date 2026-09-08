"""Monitoring over the tables a replay leaves behind.

The monitor runs as its own process, outside the replay harness: the
harness is the generator layer and is where failure injectors live, so
the thing that detects a failure shares no process, connection, or call
stack with the thing that injects it. See docs/monitoring-notes.md.
"""

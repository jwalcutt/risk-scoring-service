"""The replay harness: a frozen population streamed on a simulated clock.

The scoring service takes events one at a time, in timestamp order. This
package is what feeds it: configuration for which population is replayed
over which simulated span and how fast, the run row that is both the
simulated clock and the checkpoint, the preload that puts history from
before the start into state without scoring it, and the clock arithmetic
the tick loop is built on.
"""

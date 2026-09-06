"""The replay harness: a frozen population streamed on a simulated clock.

The scoring service takes events one at a time, in timestamp order. This
package is what feeds it: configuration for which population is replayed
over which simulated span and how fast, the run row that is both the
simulated clock and the checkpoint, the preload that puts history from
before the start into state without scoring it, the clock arithmetic the
tick loop is built on, the label schedule computed from the export up
front and released 30 simulated days after each discharge, the loop
itself (at each tick, what is due is released and posted in simulated
order and the run row is checkpointed after), the join that reconstructs
realized performance from the log and the labels, and the commands that
operate a run. The pause contract is one field: the loop stops when the
run row's status reads ``paused``, whoever wrote it.
"""

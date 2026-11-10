"""Complete single-case transcripts without changing batch/API logging defaults."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
import sys
from threading import RLock
import traceback


_ACTIVE_LOG = ContextVar("single_case_solve_log", default=None)


class _Transcript:
    def __init__(self, stream, console):
        self.stream = stream
        self.console = console
        self.lock = RLock()

    def write(self, message):
        with self.lock:
            self.stream.write(message)
            self.stream.flush()


class _Tee:
    def __init__(self, console, transcript):
        self.console = console
        self.transcript = transcript

    def write(self, message):
        self.transcript.write(message)
        return self.console.write(message)

    def flush(self):
        self.transcript.stream.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def capture_solve_log(path):
    """Stream UTF-8 output to a fresh log, retaining errors and interrupted runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        transcript = _Transcript(stream, sys.stdout)
        token = _ACTIVE_LOG.set(transcript)
        try:
            with redirect_stdout(_Tee(sys.stdout, transcript)), redirect_stderr(_Tee(sys.stderr, transcript)):
                print(f"Full solve log: {path.resolve()}")
                transcript.write(f"Started: {datetime.now().astimezone().isoformat()}\n")
                try:
                    yield path
                except BaseException:
                    transcript.write(traceback.format_exc())
                    raise
                finally:
                    transcript.write(f"Finished: {datetime.now().astimezone().isoformat()}\n")
        finally:
            _ACTIVE_LOG.reset(token)


def logged_run(log_path):
    """Resolve the configured path at invocation, including imported main() calls."""
    def decorate(main):
        @wraps(main)
        def run(*args, **kwargs):
            with capture_solve_log(log_path()):
                return main(*args, **kwargs)
        return run
    return decorate


def optimize_with_log(model, callback=None):
    """Retain every Gurobi message and forward the existing algorithm callback."""
    transcript = _ACTIVE_LOG.get()
    if transcript is None:
        return model.optimize(callback) if callback is not None else model.optimize()

    import gurobipy as gp

    output_flag = model.Params.OutputFlag
    console = model.Params.LogToConsole
    # Suppress native console output to avoid recording each message twice via Tee.
    # OutputFlag must stay enabled for MESSAGE callbacks, even in silent runs.
    model.Params.LogToConsole = 0
    model.Params.OutputFlag = 1
    transcript.write(f"\n--- Gurobi model: {model.ModelName} ---\n")

    def record(current, where):
        if where == gp.GRB.Callback.MESSAGE:
            message = current.cbGet(gp.GRB.Callback.MSG_STRING)
            transcript.write(message)
            if output_flag and console:
                transcript.console.write(message)
                transcript.console.flush()
        if callback is not None:
            callback(current, where)

    try:
        return model.optimize(record)
    finally:
        model.Params.OutputFlag = output_flag
        model.Params.LogToConsole = console

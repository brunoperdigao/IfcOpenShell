# Bonsai - OpenBIM Blender Add-on
# Copyright (C) 2024
#
# This file is part of Bonsai.
#
# Bonsai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Bonsai is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Bonsai.  If not, see <http://www.gnu.org/licenses/>.

"""
Snap system profiler — collects timing stats for the custom snapping pipeline.

Usage:
    from bonsai.tool.snap_profiler import snap_profiler

    # Enable profiling
    snap_profiler.enabled = True

    # After manual testing, print stats to system console:
    snap_profiler.print_report()

    # Reset for a fresh session:
    snap_profiler.reset()

    # Disable when done:
    snap_profiler.enabled = False

Instrumenting a function without re-indenting its body:
    def some_function(...):
        snap_profiler.start("label")
        ... body ...
        snap_profiler.stop("label")

Or using the context manager (requires indenting the body):
    def some_function(...):
        with snap_profiler.measure("label"):
            ... body ...
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator


class _Timer:
    """Per-label active timer."""

    __slots__ = ("label", "t0")

    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()


class SnapProfiler:
    """Collects per-operation timing stats for the snap pipeline."""

    def __init__(self):
        self.enabled = False
        self._active: dict[str, _Timer] = {}
        self.reset()

    def reset(self):
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._frame_start = 0.0
        self._frame_timings: list[float] = []

    @contextmanager
    def measure(self, label: str) -> Generator[None, None, None]:
        """Context manager: wrap a code block to time it."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._timings[label].append(elapsed)

    def start(self, label: str):
        """Start timing a labelled block (call stop() later with same label)."""
        if not self.enabled:
            return
        self._active[label] = _Timer(label)

    def stop(self, label: str):
        """Stop timing a labelled block previously started with start()."""
        if not self.enabled:
            return
        timer = self._active.pop(label, None)
        if timer is not None:
            elapsed = time.perf_counter() - timer.t0
            self._timings[label].append(elapsed)

    def count(self, label: str, n: int = 1):
        """Increment a named counter."""
        if not self.enabled:
            return
        self._counters[label] += n

    def start_frame(self):
        """Mark the start of a frame (called once per mouse-move cycle)."""
        if not self.enabled:
            return
        self._frame_start = time.perf_counter()

    def end_frame(self):
        """Mark the end of a frame and record total frame time."""
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._frame_start
        self._frame_timings.append(elapsed)

    def print_report(self):
        """Print a formatted report to stdout (visible in Blender system console)."""
        print("\n" + "=" * 70)
        print("  BONSAI SNAP PIPELINE — PROFILING REPORT")
        print("=" * 70)

        frames = len(self._frame_timings)
        if frames == 0:
            print("  No frames recorded. Enable profiling and move the mouse.")
            return

        total_elapsed = sum(self._frame_timings)
        print(f"  Frames captured: {frames}")
        print(f"  Total elapsed:   {total_elapsed:.3f}s")
        print(f"  Avg frame time:  {total_elapsed / frames * 1000:.1f}ms")
        print(f"  Min frame time:  {min(self._frame_timings) * 1000:.1f}ms")
        print(f"  Max frame time:  {max(self._frame_timings) * 1000:.1f}ms")
        print("-" * 70)

        # Per-operation stats, sorted by total time descending
        op_stats = []
        for label, times in self._timings.items():
            total = sum(times)
            avg = total / len(times)
            op_stats.append((label, total, avg, len(times)))

        op_stats.sort(key=lambda x: x[1], reverse=True)

        # Find max label length for alignment
        max_len = max(len(label) for label, _, _, _ in op_stats) if op_stats else 0

        print(f"  {'Operation':<{max_len}}  {'Total (s)':>10}  {'Avg (ms)':>10}  {'Calls':>6}  {'%':>5}")
        print(f"  {'-' * max_len}  {'-' * 10}  {'-' * 10}  {'-' * 6}  {'-' * 5}")

        for label, total, avg, calls in op_stats:
            pct = (total / total_elapsed * 100) if total_elapsed > 0 else 0
            print(f"  {label:<{max_len}}  {total:>10.3f}  {avg * 1000:>10.1f}  {calls:>6}  {pct:>5.1f}%")

        print("-" * 70)

        # Counters
        if self._counters:
            print("\n  Counters:")
            for label, count in sorted(self._counters.items()):
                print(f"    {label}: {count}")

        print("=" * 70 + "\n")


# Singleton
snap_profiler = SnapProfiler()

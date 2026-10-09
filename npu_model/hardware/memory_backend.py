"""Off-chip memory behind the DMA engine's TileLink port.

The DMA engine (``dma.py``) is fixed RTL and is modeled beat for beat. What it
talks to is not: in RTL simulation (EE290SimConfig) the port reaches
DRAMSim2 (DDR3, Chipyard's ``+dramsim`` default) through TileLink buffers and
a width adapter.
That path is an environment choice, so it lives behind this small interface
and is measured on VCS, not derived. Two backends exist: ``curve``
(bandwidth-latency curves after the Mess simulator, the default, loaded from
a curve file measured with DMA probe programs) and ``fixed`` (scripted
latency for tests).

The interface follows TileLink channels A and D as the engine drives them:

- ``can_accept(cycle)`` is ``tl.a.ready`` for this cycle.
- ``issue(request)`` is an A-channel fire.
- ``response(cycle)`` is the beat presented on channel D this cycle, if any.
  It stays presented until ``pop_response`` (a D-channel fire) or is replaced
  only after that fire. ``tl.d.valid`` is ``response(cycle) is not None``.

Cycles use the engine's tick numbering: a request issued at cycle T with
latency L is first presented on D at cycle T + L.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import HardwareConfig


@dataclass(frozen=True)
class BeatRequest:
    """One TileLink A beat: a Get (load) or a PutFullData (store)."""

    source: int
    """TileLink source ID; the engine's per-beat tag."""
    address: int
    """Byte address of the beat in off-chip memory."""
    is_store: bool
    cycle: int
    """Cycle of the A-channel fire."""


class MemoryBackend(ABC):
    """Everything past the DMA engine's channel A/D handshake."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def can_accept(self, cycle: int) -> bool:
        """``tl.a.ready`` this cycle."""

    @abstractmethod
    def issue(self, request: BeatRequest) -> None:
        """Record an A-channel fire."""

    @abstractmethod
    def response(self, cycle: int) -> BeatRequest | None:
        """The beat presented on channel D this cycle, or None if ``tl.d.valid`` is low."""

    @abstractmethod
    def pop_response(self, cycle: int) -> None:
        """A D-channel fire for the beat ``response`` presented."""

    @property
    @abstractmethod
    def outstanding(self) -> int:
        """Beats issued and not yet popped."""


class FixedLatencyBackend(MemoryBackend):
    """Throughput-capped link with a fixed, or scripted, per-beat latency.

    Channel A accepts at most one beat every ``cycles_per_beat`` cycles. A
    beat issued at T is presented on D from T + latency. Responses are
    presented oldest-ready first, so a scripted ``latency_fn`` that varies
    per beat reorders them, as a real fabric may. ``accept_fn`` scripts
    ``tl.a.ready`` directly (for example, buffer backpressure in a harness).

    ``latency`` counts from the A fire to the first cycle D is valid; the
    two TileLink buffers between the engine and the system bus already put
    this at two or more cycles in any Atlas configuration.
    """

    def __init__(
        self,
        latency: int,
        cycles_per_beat: int = 1,
        *,
        store_latency: int | None = None,
        latency_fn: Callable[[BeatRequest], int] | None = None,
        accept_fn: Callable[[int], bool] | None = None,
    ) -> None:
        if latency < 1:
            raise ValueError("memory latency must be at least one cycle")
        if cycles_per_beat < 1:
            raise ValueError("cycles_per_beat must be at least one")
        self.latency = latency
        self.store_latency = latency if store_latency is None else store_latency
        self.cycles_per_beat = cycles_per_beat
        self.latency_fn = latency_fn
        self.accept_fn = accept_fn
        self.reset()

    def reset(self) -> None:
        self._last_accept: int | None = None
        # (ready_cycle, issue_order, request), kept sorted.
        self._pending: list[tuple[int, int, BeatRequest]] = []
        self._issued = 0
        self._presented: BeatRequest | None = None

    def _latency_for(self, request: BeatRequest) -> int:
        if self.latency_fn is not None:
            return max(1, int(self.latency_fn(request)))
        return self.store_latency if request.is_store else self.latency

    def can_accept(self, cycle: int) -> bool:
        if self.accept_fn is not None and not self.accept_fn(cycle):
            return False
        if self._last_accept is None:
            return True
        return cycle - self._last_accept >= self.cycles_per_beat

    def issue(self, request: BeatRequest) -> None:
        if not self.can_accept(request.cycle):
            raise RuntimeError(f"memory backend refused beat {request} on cycle {request.cycle}")
        self._last_accept = request.cycle
        entry = (request.cycle + self._latency_for(request), self._issued, request)
        self._issued += 1
        index = len(self._pending)
        while index > 0 and self._pending[index - 1][:2] > entry[:2]:
            index -= 1
        self._pending.insert(index, entry)

    def response(self, cycle: int) -> BeatRequest | None:
        if self._presented is None and self._pending and self._pending[0][0] <= cycle:
            self._presented = self._pending[0][2]
        return self._presented

    def pop_response(self, cycle: int) -> None:
        if self._presented is None or self.response(cycle) is None:
            raise RuntimeError("D-channel fire with nothing presented")
        self._pending.pop(0)
        self._presented = None

    @property
    def outstanding(self) -> int:
        return len(self._pending)


class CurveMemoryBackend(MemoryBackend):
    """Bandwidth-latency curve memory, after the Mess simulator (MICRO 2024).

    The memory is described only by measured curves: for each read
    percentage, latency as a function of delivered bandwidth. A controller
    watches the beats delivered in a window, turns the window into a
    bandwidth and a read percentage, looks the latency up on the matching
    curve and applies it to every beat issued until the next window closes,
    smoothing the estimate with ``converge`` (1.0 jumps straight to the curve;
    Mess uses 0.05 for CPU workloads over 1000-access windows, far too slow
    for 32-beat Atlas transfers). Because one Atlas command is all loads or
    all stores, the curve a beat is read from follows the mix of the last
    ``window_beats`` beats issued, while the bandwidth estimate comes from the
    delivered window. Responses are also paced so delivered bandwidth never
    exceeds the peak of that curve (``enforce_peak``), which is how a burst
    queues behind the memory port.

    The DMA engine is the memory's only client while the NPU runs, and the
    engine keeps up to 64 beats in flight whatever the latency. A curve
    measured with a probe therefore rises steeply at the peak only because
    the probe waited behind the engine's own backlog, and that backlog is
    exactly what the pacing reproduces. ``latency_mode="knee"`` (default)
    evaluates the curve no further than ``knee`` of the peak bandwidth, so
    the rise before the knee (bank conflicts, clock crossing, contention)
    is applied but the backlog wall is not counted twice. ``"unloaded"``
    applies only each curve's lead-off latency and lets the pacing produce
    all queueing: the right reading of a curve measured with a DMA probe,
    whose loaded latencies are nothing but the probe's wait behind the
    engine's backlog. ``"mess"`` applies the curve verbatim with Mess's
    overflow penalty beyond the peak; use it for curves that describe
    another client's loaded latency.

    A curve file may carry ``latencyMode`` to set the default mode for how
    it was measured; ``scripts/fit_dma_probe_curve.py`` writes ``"unloaded"``.

    ``curves`` is a Mess curve file (path, a name in
    ``npu_model/configs/memory_curves``, or a loaded dict): ``measuredChannels``
    and ``curves`` mapping read percentage to ``[[MB/s, ns], ...]``. Two
    optional fields extend the format for Atlas: ``accessBytes`` (the access
    size the measurement used, Mess assumes 64, DMA-side measurements use 32;
    bandwidth is bytes either way) and ``frequencyGHz`` (the core clock the
    latencies are converted with, overridable by ``frequency_ghz``). A beat
    issued at T with the current latency L is presented on D at T + L or
    later under pacing, oldest ready first.
    """

    def __init__(
        self,
        curves: str | dict,
        *,
        window_beats: int = 32,
        converge: float = 1.0,
        latency_mode: str | None = None,
        knee: float = 0.9,
        frequency_ghz: float | None = None,
        beat_bytes: int = 32,
        channels: int | None = None,
        enforce_peak: bool = True,
        queue_depth: int = 64,
        min_latency: int = 1,
    ) -> None:
        if window_beats < 1:
            raise ValueError("window_beats must be at least one")
        if not 0.0 < converge <= 1.0:
            raise ValueError("converge must be in (0, 1]")
        if not 0.0 < knee <= 1.0:
            raise ValueError("knee must be in (0, 1]")
        data = load_curve_file(curves)
        if latency_mode is None:
            latency_mode = str(data.get("latencyMode", "knee"))
        if latency_mode not in ("knee", "unloaded", "mess"):
            raise ValueError("latency_mode must be 'knee', 'unloaded' or 'mess'")
        self.path = data.get("path")
        self.source = data.get("source", self.path or "<dict>")
        measured_channels = int(data.get("measuredChannels", 1))
        if measured_channels < 1:
            raise ValueError("measuredChannels must be positive")
        self.channels = measured_channels if channels is None else int(channels)
        self.frequency_ghz = float(frequency_ghz if frequency_ghz is not None else data.get("frequencyGHz", 1.0))
        self.access_bytes = int(data.get("accessBytes", 64))
        self.beat_bytes = int(beat_bytes)
        self.window_beats = int(window_beats)
        self.converge = float(converge)
        self.latency_mode = latency_mode
        self.knee = float(knee)
        self.enforce_peak = enforce_peak
        self.queue_depth = int(queue_depth)
        self.min_latency = int(min_latency)
        # Bandwidth counts bytes whatever the measured access size, so beats per
        # cycle = MB/s * 1e6 / beat_bytes / (GHz * 1e9), scaled by channels.
        scale = (self.channels / measured_channels) * 1e6 / self.beat_bytes / (self.frequency_ghz * 1e9)
        raw = data["curves"]
        if not raw:
            raise ValueError("curve file has no curves")
        self._curves: dict[int, list[tuple[float, float]]] = {}
        for key, points in raw.items():
            pct = int(round(float(key)))
            converted = sorted(((float(bw) * scale, float(lat) * self.frequency_ghz) for bw, lat in points),
                               key=lambda point: point[0])
            if not converted:
                raise ValueError(f"curve {key} is empty")
            self._curves[pct] = converted
        self._pcts = sorted(self._curves)
        self.lead_off_latency = max(self.min_latency, min(lat for c in self._curves.values() for _, lat in c))
        self.peak_bandwidth = max(bw for c in self._curves.values() for bw, _ in c)
        self.reset()

    # -- curve lookup ------------------------------------------------------

    def _curve_for(self, read_pct: float) -> list[tuple[float, float]]:
        pct = min(self._pcts, key=lambda p: (abs(p - read_pct), p))
        return self._curves[pct]

    def latency_at(self, bandwidth: float, read_pct: float = 100.0) -> float:
        """Latency (cycles) the curve gives at ``bandwidth`` beats per cycle, no smoothing."""
        curve = self._curve_for(read_pct)
        if bandwidth <= curve[0][0]:
            return max(self.min_latency, curve[0][1])
        for (bw0, lat0), (bw1, lat1) in zip(curve, curve[1:]):
            if bandwidth <= bw1:
                if bw1 == bw0:
                    return max(self.min_latency, max(lat0, lat1))
                return max(self.min_latency, lat0 + (bandwidth - bw0) / (bw1 - bw0) * (lat1 - lat0))
        return max(self.min_latency, curve[-1][1])

    def curve_peak(self, read_pct: float = 100.0) -> tuple[float, float]:
        """(peak bandwidth, latency there) of the curve nearest ``read_pct``."""
        curve = self._curve_for(read_pct)
        bw, lat = max(curve, key=lambda point: point[0])
        return bw, max(self.min_latency, lat)

    # -- controller --------------------------------------------------------

    def reset(self) -> None:
        self.latency = float(self.lead_off_latency)
        self._estimated_bandwidth = 0.0
        self._estimated_latency = float(self.lead_off_latency)
        self._overflow = 0.0
        self._window_count = 0
        self._window_stores = 0
        self._window_start: int | None = None
        self._last_ready = 0.0
        self._recent: list[bool] = []
        self._pending: list[tuple[int, int, BeatRequest]] = []
        self._issued = 0
        self._presented: BeatRequest | None = None
        self.windows: list[tuple[int, float, float, float]] = []
        """(end cycle, delivered bandwidth, read percentage, latency) per closed window."""

    def _close_window(self, end_cycle: int) -> None:
        span = max(1, end_cycle - (self._window_start if self._window_start is not None else end_cycle))
        bandwidth = self._window_count / span
        read_pct = 100.0 * (self._window_count - self._window_stores) / self._window_count
        self._update_latency(bandwidth, read_pct)
        self.windows.append((end_cycle, bandwidth, read_pct, self.latency))
        self._window_count = 0
        self._window_stores = 0
        self._window_start = end_cycle

    def _issued_read_pct(self) -> float:
        if not self._recent:
            return 100.0
        return 100.0 * (len(self._recent) - sum(self._recent)) / len(self._recent)

    def _knee_latency(self, bandwidth: float, read_pct: float) -> float:
        if self.latency_mode == "unloaded":
            return self.latency_at(0.0, read_pct)
        peak_bw, _ = self.curve_peak(read_pct)
        return self.latency_at(min(bandwidth, self.knee * peak_bw), read_pct)

    def _update_latency(self, bandwidth: float, read_pct: float) -> None:
        k = self.converge
        bandwidth = k * bandwidth + (1 - k) * self._estimated_bandwidth
        peak_bw, peak_lat = self.curve_peak(read_pct)
        if self.latency_mode in ("knee", "unloaded"):
            latency = self._knee_latency(bandwidth, read_pct)
        elif bandwidth > 0.99 * peak_bw:
            # Demand beyond the curve: latency climbs past the peak, as in Mess.
            self._overflow += 0.02
            latency = (1 + self._overflow) * peak_lat
            bandwidth = k * peak_bw + (1 - k) * self._estimated_bandwidth
        else:
            latency = self.latency_at(bandwidth, read_pct) * (1 + self._overflow)
            self._overflow = self._overflow - 0.01 if self._overflow > 0.01 else 0.0
        latency = k * latency + (1 - k) * self._estimated_latency
        self._estimated_bandwidth = bandwidth
        self._estimated_latency = latency
        self.latency = max(float(self.lead_off_latency), latency)

    # -- channel A/D -------------------------------------------------------

    def can_accept(self, cycle: int) -> bool:
        return len(self._pending) < self.queue_depth

    def issue(self, request: BeatRequest) -> None:
        if not self.can_accept(request.cycle):
            raise RuntimeError(f"memory backend refused beat {request} on cycle {request.cycle}")
        self._recent.append(request.is_store)
        del self._recent[:-self.window_beats]
        read_pct = self._issued_read_pct()
        if self.latency_mode in ("knee", "unloaded"):
            # The window gives the bandwidth; the beats being issued pick the curve.
            latency = max(self._knee_latency(self._estimated_bandwidth, read_pct),
                          (1 - self.converge) * self._estimated_latency if self.windows else 0.0)
        else:
            latency = self.latency
        ready = request.cycle + max(float(self.min_latency), latency)
        if self.enforce_peak:
            peak_bw, _ = self.curve_peak(read_pct)
            ready = max(ready, self._last_ready + 1.0 / peak_bw)
            self._last_ready = ready
        entry = (math.ceil(ready - 1e-9), self._issued, request)
        self._issued += 1
        index = len(self._pending)
        while index > 0 and self._pending[index - 1][:2] > entry[:2]:
            index -= 1
        self._pending.insert(index, entry)

    def response(self, cycle: int) -> BeatRequest | None:
        if self._presented is None and self._pending and self._pending[0][0] <= cycle:
            self._presented = self._pending[0][2]
        return self._presented

    def pop_response(self, cycle: int) -> None:
        if self._presented is None or self.response(cycle) is None:
            raise RuntimeError("D-channel fire with nothing presented")
        request = self._pending.pop(0)[2]
        self._presented = None
        if self._window_start is None:
            self._window_start = request.cycle      # first delivery: count from its issue
        self._window_count += 1
        if request.is_store:
            self._window_stores += 1
        if self._window_count == self.window_beats:
            self._close_window(cycle)

    @property
    def outstanding(self) -> int:
        return len(self._pending)


def load_curve_file(curves: str | dict) -> dict:
    """Read a Mess curve file (JSON) or pass a loaded dict through, validated."""
    if isinstance(curves, dict):
        data = dict(curves)
    else:
        import json
        from pathlib import Path

        path = Path(curves)
        if not path.is_absolute() and not path.exists():
            candidate = Path(__file__).resolve().parent.parent / "configs" / "memory_curves" / path
            if candidate.exists():
                path = candidate
        with open(path) as handle:
            data = json.load(handle)
        data["path"] = str(path)
    if "curves" not in data or not isinstance(data["curves"], dict):
        raise ValueError("curve file needs a 'curves' object keyed by read percentage")
    return data


def default_fixed_latency(config: HardwareConfig) -> tuple[int, int]:
    """(latency, cycles_per_beat) implied by the spec's serialized-link parameters.

    The frozen baseline charges ``OFFCHIP_LINK_CORE_CYCLES_PER_BEAT`` per
    ``OFFCHIP_LINK_WIDTH_BITS`` on the link, plus ``DMA_OFFCHIP_COMMAND_WORDS``
    of command overhead. Spread over 32-byte DMA beats that is one beat every
    16 cycles after a 4-cycle command delay, with the default values. Phase 2
    calibration against the RTL replaces these with measured numbers.
    """
    link_bytes = config.offchip_link_width_bits // 8
    link_beats_per_dma_beat = -(-config.dma_beat_bytes // link_bytes)
    cycles_per_beat = link_beats_per_dma_beat * config.offchip_link_core_cycles_per_beat
    latency = max(1, config.dma_offchip_command_words * config.offchip_link_core_cycles_per_beat)
    return latency, cycles_per_beat


def make_memory_backend(config: HardwareConfig) -> MemoryBackend:
    """Build the backend ``config.dma_memory_backend`` names.

    ``config.dma_memory_params`` supplies keyword arguments for the backend
    class; for "fixed", ``dma_memory_latency_cycles`` and
    ``dma_memory_cycles_per_beat`` override the spec-derived defaults.
    """
    kind = config.dma_memory_backend
    params = dict(getattr(config, "dma_memory_params", {}) or {})
    if kind == "fixed":
        latency, cycles_per_beat = default_fixed_latency(config)
        if config.dma_memory_latency_cycles is not None:
            latency = config.dma_memory_latency_cycles
        if config.dma_memory_cycles_per_beat is not None:
            cycles_per_beat = config.dma_memory_cycles_per_beat
        params.setdefault("latency", latency)
        params.setdefault("cycles_per_beat", cycles_per_beat)
        return FixedLatencyBackend(**params)
    if kind == "curve":
        params.setdefault("beat_bytes", config.dma_beat_bytes)
        params.setdefault("queue_depth", config.dma_max_in_flight)
        return CurveMemoryBackend(**params)
    raise ValueError(f"unknown DMA memory backend '{kind}'")

"""Channel A/D timing of the off-chip memory backends behind the DMA engine."""
import math

import pytest

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware.memory_backend import (
    BeatRequest, CurveMemoryBackend, FixedLatencyBackend, make_memory_backend,
)


def beat(source, cycle, store=False):
    return BeatRequest(source=source, address=0x1000 + 32 * source, is_store=store, cycle=cycle)


def ready_cycles(backend, requests, horizon=4096):
    """Cycle each response is first presented, popping it immediately."""
    ready = {}
    by_cycle = {}
    for request in requests:
        by_cycle.setdefault(request.cycle, []).append(request)
    for cycle in range(1, horizon):
        for request in by_cycle.get(cycle, []):
            assert backend.can_accept(cycle)
            backend.issue(request)
        response = backend.response(cycle)
        if response is not None:
            ready[response.source] = cycle
            backend.pop_response(cycle)
        if len(ready) == len(requests):
            break
    return ready


def test_fixed_latency_presents_each_beat_latency_cycles_after_issue():
    backend = FixedLatencyBackend(latency=5)
    assert ready_cycles(backend, [beat(0, 1), beat(1, 2), beat(2, 3)]) == {0: 6, 1: 7, 2: 8}


def test_fixed_latency_rate_limit_gates_channel_a():
    backend = FixedLatencyBackend(latency=1, cycles_per_beat=3)
    backend.issue(beat(0, 1))
    assert not backend.can_accept(2) and not backend.can_accept(3) and backend.can_accept(4)
    with pytest.raises(RuntimeError):
        backend.issue(beat(1, 2))


def test_fixed_latency_scripted_latencies_reorder_and_hold_until_popped():
    backend = FixedLatencyBackend(latency=1, latency_fn=lambda r: 10 if r.source == 0 else 2)
    backend.issue(beat(0, 1))
    backend.issue(beat(1, 2))
    assert backend.response(3) is None
    assert backend.response(4).source == 1          # the later beat returns first
    assert backend.response(9).source == 1          # held until the D fire
    backend.pop_response(9)
    assert backend.response(10) is None
    assert backend.response(11).source == 0
    assert backend.outstanding == 1


def test_make_backend_reads_kind_and_params():
    cfg = DefaultHardwareConfig()
    default = make_memory_backend(cfg)
    assert isinstance(default, CurveMemoryBackend)
    assert default.path.endswith("memory_curves/ee290sim_vcs_probe.json")
    assert (default.latency_mode, default.beat_bytes, default.queue_depth) == ("unloaded", 32, 64)
    cfg.dma_memory_params = {"curves": "ee290sim_vcs_probe.json", "window_beats": 16, "latency_mode": "knee"}
    backend = make_memory_backend(cfg)
    assert (backend.window_beats, backend.latency_mode) == (16, "knee")
    cfg.dma_memory_backend = "fixed"
    cfg.dma_memory_params = {}
    cfg.dma_memory_latency_cycles = 7
    assert make_memory_backend(cfg).latency == 7
    cfg.dma_memory_backend = "queue"
    with pytest.raises(ValueError, match="unknown DMA memory backend"):
        make_memory_backend(cfg)


# ---------------------------------------------------------------------------
# Curve backend (Mess-style bandwidth-latency curves)
# ---------------------------------------------------------------------------

GHZ = 0.5
BEAT = 32


def mbps(beats_per_cycle):
    """Beats per cycle at 0.5 GHz and 32 B beats, in the curve file's MB/s."""
    return beats_per_cycle * BEAT * GHZ * 1e9 / 1e6


def ns(cycles):
    return cycles / GHZ


def synthetic_curves(load_lat=(10, 30, 90), store_lat=(8, 24, 72)):
    """Latency 10 -> 30 -> 90 cycles as bandwidth goes 0.01 -> 0.1 -> 0.5 beats/cycle."""
    bws = (0.01, 0.1, 0.5)
    return {
        "measuredChannels": 1,
        "accessBytes": BEAT,
        "frequencyGHz": GHZ,
        "curves": {
            "100": [[mbps(b), ns(l)] for b, l in zip(bws, load_lat)],
            "0": [[mbps(b), ns(l)] for b, l in zip(bws, store_lat)],
        },
    }


def test_curve_units_and_lookup():
    backend = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT)
    assert backend.lead_off_latency == 8                       # minimum over all curves
    assert backend.peak_bandwidth == pytest.approx(0.5)
    assert backend.latency_at(0.0, 100) == 10                  # clamped below the first point
    assert backend.latency_at(0.055, 100) == pytest.approx(20) # linear between points
    assert backend.latency_at(0.3, 100) == pytest.approx(60)
    assert backend.latency_at(5.0, 100) == 90                  # clamped above the last point
    assert backend.latency_at(0.1, 0) == 24                    # store curve
    assert backend.latency_at(0.1, 30) == 24                   # nearest available curve
    assert backend.curve_peak(100) == (pytest.approx(0.5), 90)


def drive(backend, requests, horizon=4096):
    """Issue requests at their cycles and pop responses as they come; returns {source: ready cycle}."""
    ready, by_cycle = {}, {}
    for request in requests:
        by_cycle.setdefault(request.cycle, []).append(request)
    for cycle in range(1, horizon):
        for request in by_cycle.get(cycle, []):
            backend.issue(request)
        response = backend.response(cycle)
        if response is not None:
            ready[response.source] = cycle
            backend.pop_response(cycle)
        if len(ready) == len(requests):
            break
    return ready


def test_curve_beats_pay_their_curve_lead_off_until_a_window_closes():
    backend = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT, window_beats=4)
    # A lone load reads the 100 % curve (lead-off 10), a lone store the 0 % curve (8).
    assert drive(backend, [beat(0, 1), beat(1, 300, store=True)]) == {0: 11, 1: 308}
    assert backend.windows == []                      # two deliveries, window needs four
    backend.reset()
    drive(backend, [beat(i, 1 + 100 * i) for i in range(4)])
    # Window: first beat issued at 1, fourth delivered at 311 -> 4/310 beats/cycle on
    # the load curve, between the 0.01 and 0.1 points.
    end, bw, read_pct, latency = backend.windows[0]
    assert (end, read_pct) == (311, 100.0) and bw == pytest.approx(4 / 310)
    assert latency == pytest.approx(backend.latency_at(4 / 310, 100))
    assert 10 < latency < 11
    assert drive(backend, [beat(9, 400)]) == {9: 400 + math.ceil(latency - 1e-9)}


def test_curve_window_picks_the_store_curve_and_the_next_beats_pay_it():
    backend = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT, window_beats=4, enforce_peak=False)
    stores = [beat(i, 1 + 10 * i, store=True) for i in range(4)]
    drive(backend, stores)                            # issued 1..31, delivered 9..39: 4/38 beats/cycle
    end, bw, read_pct, latency = backend.windows[-1]
    assert (end, read_pct) == (39, 0.0) and bw == pytest.approx(4 / 38)
    assert latency == pytest.approx(backend.latency_at(4 / 38, 0)) and 24 < latency < 72
    assert drive(backend, [beat(9, 100, store=True)]) == {9: 100 + math.ceil(latency - 1e-9)}
    # A load issued now is one of the last four beats issued (25 % reads), so it
    # still reads the nearest curve, the store one, at the same bandwidth estimate.
    expect = backend.latency_at(4 / 38, 25)
    assert drive(backend, [beat(10, 200)]) == {10: 200 + math.ceil(expect - 1e-9)}
    # Four loads in a row flip the mix to the load curve.
    drive(backend, [beat(20 + i, 300 + 50 * i) for i in range(3)])
    expect = backend.latency_at(backend._estimated_bandwidth, 100)
    assert drive(backend, [beat(30, 600)]) == {30: 600 + math.ceil(expect - 1e-9)}


def test_curve_knee_mode_stops_short_of_the_backlog_wall_but_mess_mode_climbs_it():
    curves = synthetic_curves(load_lat=(10, 30, 900))   # wall at the 0.5 peak
    knee = CurveMemoryBackend(curves, beat_bytes=BEAT, window_beats=4, enforce_peak=False, knee=0.5)
    drive(knee, [beat(i, 1 + i) for i in range(4)])     # ~0.3 beats/cycle delivered
    assert knee.latency == pytest.approx(knee.latency_at(0.25, 100))   # capped at knee * peak
    mess = CurveMemoryBackend(curves, beat_bytes=BEAT, window_beats=4, enforce_peak=False, latency_mode="mess")
    drive(mess, [beat(i, 1 + i) for i in range(12)])    # one beat per cycle, three windows
    first, second, third = mess.windows
    assert first[3] == pytest.approx(mess.latency_at(4 / 11, 100))   # delivered 9..12 from issue at 1
    assert second[1] == pytest.approx(1.0) and second[3] == pytest.approx(1.02 * 900)   # past the 0.5 peak
    assert third[3] == pytest.approx(1.04 * 900)                     # the penalty keeps growing


def test_curve_unloaded_mode_pays_only_the_lead_off_and_paces_at_the_peak():
    backend = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT, window_beats=4, latency_mode="unloaded")
    ready = drive(backend, [beat(i, 1 + i) for i in range(8)])   # one beat per cycle, past the peak
    assert backend.windows and backend.windows[-1][3] == 10       # latency stays at the load lead-off
    assert [ready[i] for i in range(8)] == [11, 13, 15, 17, 19, 21, 23, 25]   # 0.5 beats/cycle pacing
    with pytest.raises(ValueError):
        CurveMemoryBackend(synthetic_curves(), latency_mode="other")
    tagged = dict(synthetic_curves(), latencyMode="unloaded")
    assert CurveMemoryBackend(tagged).latency_mode == "unloaded"
    assert CurveMemoryBackend(tagged, latency_mode="knee").latency_mode == "knee"
    assert CurveMemoryBackend("ee290sim_vcs_probe.json").latency_mode == "unloaded"


def test_curve_converge_smooths_the_estimate():
    backend = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT, window_beats=4, converge=0.5, enforce_peak=False)
    drive(backend, [beat(i, 1 + 10 * i) for i in range(4)])
    bw = backend.windows[0][1]
    target = backend.latency_at(0.5 * bw, 100)        # bandwidth is smoothed first, from 0
    assert backend.latency == pytest.approx(0.5 * target + 0.5 * 8)


def test_curve_peak_enforcement_spaces_responses():
    backend = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT, window_beats=64)
    # Peak is 0.5 beats/cycle: a burst drains no faster than one beat per 2 cycles.
    ready = drive(backend, [beat(i, 1) for i in range(6)])
    assert [ready[i] for i in range(6)] == [11, 13, 15, 17, 19, 21]
    free = CurveMemoryBackend(synthetic_curves(), beat_bytes=BEAT, window_beats=64, enforce_peak=False)
    free.issue(beat(0, 1)); free.issue(beat(1, 1)); free.issue(beat(2, 1))
    assert [entry[0] for entry in free._pending] == [11, 11, 11]


def test_curve_file_loads_from_the_bundled_directory_and_the_factory(tmp_path):
    import json
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(synthetic_curves()))
    backend = CurveMemoryBackend(str(path), beat_bytes=BEAT, channels=2)
    assert backend.peak_bandwidth == pytest.approx(1.0)              # two channels double the curve
    cfg = DefaultHardwareConfig()
    cfg.dma_memory_backend = "curve"
    cfg.dma_memory_params = {"curves": str(path)}
    made = make_memory_backend(cfg)
    assert isinstance(made, CurveMemoryBackend)
    assert (made.beat_bytes, made.queue_depth) == (cfg.dma_beat_bytes, cfg.dma_max_in_flight)
    bundled = CurveMemoryBackend("ee290sim_vcs_probe.json")
    assert bundled.path.endswith("memory_curves/ee290sim_vcs_probe.json")
    assert bundled.frequency_ghz == 0.5 and bundled.access_bytes == 32
    assert 40 <= bundled.lead_off_latency <= 50                      # 42-cycle Put lead-off
    assert 44 <= bundled.latency_at(0.0, 100) <= 50                 # 46-cycle Get lead-off
    assert 0.04 <= bundled.peak_bandwidth <= 0.05                   # about 23 cycles per beat
    with pytest.raises(ValueError):
        CurveMemoryBackend({"measuredChannels": 1, "curves": {}})
    with pytest.raises(ValueError):
        CurveMemoryBackend(synthetic_curves(), converge=0)

"""The DMA unit's beat and command traces (hardware/dma.py) and the transaction
programs' timed window (scripts/dma_transactions.py)."""
import sys
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / "scripts"))

from dma_transactions import run_program  # noqa: E402

TXN = MODEL / "calibration/dma/transactions"


@pytest.fixture(scope="module")
def single_load():
    return run_program(TXN / "txn_load_32b.S", None, {})


def test_single_beat_round_trip_is_the_lead_off_latency(single_load):
    assert single_load["beats"] == 1
    assert len(single_load["commands"]) == 1
    # Curve backend: an unloaded beat gets the curve's lead-off latency (46 per Get).
    assert single_load["round_trip_min"] == single_load["round_trip_max"] == 46
    assert single_load["launch_to_first_a"] == 1          # A fires the cycle after issue
    assert single_load["last_d_to_retire"] == 0           # the slot retires on the D fire
    assert single_load["model_dbg1"] == single_load["round_trip_min"] + single_load["engine_overhead"]


def test_timed_window_excludes_the_warm_up():
    res = run_program(TXN / "txn_load_after_load_s5.S", None, {})
    assert res["beats"] == 1 and len(res["commands"]) == 1
    assert res["commands"][0]["channel"] == 0            # the probe, not the ch1 warm-up


def test_burst_issues_one_beat_per_cycle():
    res = run_program(TXN / "txn_load_1024b.S", None, {})
    assert res["beats"] == 32
    issue_offsets = [queue for _, queue, _ in res["beat_trace"]]
    assert issue_offsets == list(range(1, 33))            # A fires at launch+1 .. launch+32
    trips = [trip for _, _, trip in res["beat_trace"]]
    assert trips[0] == 46 and trips == sorted(trips)      # later beats wait behind the pacing
    assert res["commands"][0]["span"] == res["commands"][0]["last_d"] - res["commands"][0]["launched"]


def test_stores_round_trip_faster_than_loads(single_load):
    store = run_program(TXN / "txn_store_32b.S", None, {})
    assert store["round_trip_min"] == 42 < single_load["round_trip_min"]
    assert store["launch_to_first_a"] == 3                # VMEM read, SyncReadMem, store data queue

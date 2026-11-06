"""Per-transaction and per-instruction DMA timing on the model, next to measurements.

Runs the transaction programs (``scripts/gen_dma_transaction_programs.py``)
through the Python model and reports, for the transfer(s) inside each
program's timed window (between its two ``csrrs ... 0xC00``):

* CSR_DBG1 as the program banks it (what VCS and the metasim report too);
* per beat: A-fire to D-fire cycles (the memory round trip of one 32 B
  TileLink transaction), launch to first A (engine queueing), last D to
  retirement, and the latency the curve backend had selected;
* per command: launch, first A, last D, retire.

Measurements are merged by program name: ``--summary`` takes the metasim
runner's ``summary.csv`` (``test``, ``dbg1`` columns) or any CSV with those
columns; ``--measured NAME=CYCLES`` adds single values. For single-beat
programs the measured beat round trip is derived as
``measured_dbg1 - (model_dbg1 - model_round_trip)``: the engine overhead
around one beat is fixed and the model knows it exactly.

    python scripts/dma_transactions.py                       # all transaction programs
    python scripts/dma_transactions.py --summary $PROBE_ROOT/txn/runs/frfcfs/summary.csv --out txn.json
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import statistics
import sys
import tempfile
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(MODEL / "scripts"))

from calibrate_dma_backend import directive, make_config  # noqa: E402
from npu_model.hardware.dma import DmaExecutionUnit  # noqa: E402
from npu_model.logging import LoggerConfig  # noqa: E402
from npu_model.simulation import Simulation  # noqa: E402
from npu_model.software.program import InstantiableProgram  # noqa: E402
from npu_model.util.converter import stream_to_instrs  # noqa: E402

PROGRAMS = MODEL / "calibration/dma/transactions"


def run_program(asm_path: Path, backend: str | None, params: dict) -> dict:
    source = asm_path.read_text()
    program = InstantiableProgram(stream_to_instrs(io.StringIO(source)))
    program.memory_regions = []
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        trace = handle.name
    try:
        sim = Simulation(make_config(backend, params), LoggerConfig(filename=trace), program, verbose=False)
        core = sim.core
        csr_reads: list[int] = []
        max_cycles = directive(source, "TIMEOUT", 1_000_000)
        cycles = 0
        with contextlib.redirect_stdout(io.StringIO()):
            while not core.is_finished() and cycles < max_cycles:
                core.tick()
                cycles += 1
                last = core.last_cycle
                if last.get("s1_fire") and last.get("instruction") == "csrrs":
                    csr_reads.append(last["cycle"])
            core.stop()
            sim.logger.close()
        if not core.arch_state.halted:
            raise RuntimeError(f"{asm_path.name} did not halt in the model")
        dma = next(exu for exu in core.exus if isinstance(exu, DmaExecutionUnit))
        t0, t1 = (csr_reads[0], csr_reads[1]) if len(csr_reads) >= 2 else (0, cycles)
        commands = [c for c in dma.command_trace if t0 <= c.launched <= t1]
        timed_ids = {c.command for c in commands}
        beats = [b for b in dma.beat_trace if b.command in timed_ids]
        trips = [b.round_trip for b in beats if b.round_trip is not None]
        dbg1 = int(core.arch_state.read_csrf(0xC11))
        single = len(beats) == 1 and trips
        return {
            "name": asm_path.stem,
            "model_dbg1": dbg1,
            "engine_overhead": (dbg1 - trips[0]) if single else None,
            "window": [t0, t1],
            "commands": [{
                "channel": c.channel, "store": c.is_store, "beats": c.beats, "address": hex(c.address),
                "launched": c.launched, "first_a": c.first_a, "last_d": c.last_d, "retired": c.retired,
                "span": c.retired - c.launched,
            } for c in commands],
            "beats": len(beats),
            "round_trip_min": min(trips) if trips else None,
            "round_trip_median": statistics.median(trips) if trips else None,
            "round_trip_max": max(trips) if trips else None,
            "first_beat_round_trip": trips[0] if trips else None,
            "launch_to_first_a": (beats[0].a_fire - beats[0].launched) if beats else None,
            "last_d_to_retire": (commands[-1].retired - commands[-1].last_d) if commands and commands[-1].last_d else None,
            "backend_latency": beats[0].backend_latency if beats else None,
            "beat_trace": [[b.index, b.a_fire - b.launched, b.round_trip] for b in beats],
        }
    finally:
        Path(trace).unlink(missing_ok=True)


def read_measurements(summaries: list[Path], values: list[str]) -> dict[str, dict[str, int]]:
    """{program: {source label: dbg1}}."""
    out: dict[str, dict[str, int]] = {}
    for path in summaries:
        label = path.parent.name if path.name == "summary.csv" else path.stem
        with path.open() as handle:
            for row in csv.DictReader(handle):
                name = row.get("test") or row.get("name")
                if name and row.get("dbg1") not in (None, ""):
                    out.setdefault(name, {})[label] = int(float(row["dbg1"]))
    for item in values:
        name, cycles = item.split("=")
        out.setdefault(name, {})["measured"] = int(cycles)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--asm", nargs="*", type=Path, help="programs (default: all transaction programs)")
    parser.add_argument("--backend", help="model memory backend (default: the configured one)")
    parser.add_argument("--param", action="append", default=[], help="backend parameter key=value")
    parser.add_argument("--summary", nargs="*", type=Path, default=[], help="metasim summary.csv files")
    parser.add_argument("--measured", nargs="*", default=[], help="NAME=CYCLES")
    parser.add_argument("--out", type=Path, help="write everything as JSON")
    args = parser.parse_args()
    params = {}
    for item in args.param:
        key, value = item.split("=", 1)
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    programs = args.asm or sorted(PROGRAMS.glob("txn_*.S"))
    measured = read_measurements(args.summary, args.measured)
    labels = sorted({label for sources in measured.values() for label in sources})

    results = []
    header = (f"{'program':34} {'dbg1':>6} {'cmds':>4} {'beats':>5} {'A->D min/med/max':>18} "
              f"{'q':>3} {'tail':>4} {'curve':>6}" + "".join(f" {label[:12]:>12} {'beat':>5}" for label in labels))
    print(header)
    for path in programs:
        res = run_program(path, args.backend, params)
        res["measured"] = measured.get(res["name"], {})
        overhead = res["engine_overhead"]
        res["measured_round_trip"] = ({label: value - overhead for label, value in res["measured"].items()}
                                      if overhead is not None else {})
        results.append(res)
        trips = (f"{res['round_trip_min']}/{res['round_trip_median']:.0f}/{res['round_trip_max']}"
                 if res["beats"] else "-")
        line = (f"{res['name']:34} {res['model_dbg1']:6d} {len(res['commands']):4d} {res['beats']:5d} {trips:>18} "
                f"{res['launch_to_first_a'] if res['launch_to_first_a'] is not None else '-':>3} "
                f"{res['last_d_to_retire'] if res['last_d_to_retire'] is not None else '-':>4} "
                f"{res['backend_latency'] if res['backend_latency'] is not None else '-':>6}")
        for label in labels:
            value = res["measured"].get(label)
            beat = res["measured_round_trip"].get(label)
            line += f" {value if value is not None else '-':>12} {beat if beat is not None else '-':>5}"
        print(line, flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

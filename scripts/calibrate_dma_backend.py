#!/usr/bin/env python3
"""Score (or fit) the DMA memory backend against cycle deltas measured on the RTL.

The programs ``calibration/dma/programs/perf_dma_*.S`` bank the mcycles delta
around a DMA region into CSR_DBG1; the generated C harness prints it as
``dbg1_cycles``. This script runs the same programs through the Python model
and compares the model's CSR_DBG1 with the measurement.

Measurements come from ``calibration/dma/measurements/ee290sim_vcs.json`` by
default (the refresh-off VCS run the bundled curve was made from), or from VCS
logs (``--logs`` / ``--log-dir``), or ``--measured NAME=CYCLES``; later sources
override earlier ones.

Examples:
    # The default backend against the bundled measurements
    python scripts/calibrate_dma_backend.py

    # Fresh VCS logs and a parameter sweep of the fixed backend
    python scripts/calibrate_dma_backend.py --logs /path/to/output/atlas_perf_dma_*.log \
        --backend fixed --sweep latency=2:40 cycles_per_beat=16:24
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))

from npu_model.configs.hardware.default import DefaultHardwareConfig  # noqa: E402
from npu_model.logging import LoggerConfig  # noqa: E402
from npu_model.simulation import Simulation  # noqa: E402
from npu_model.software.program import InstantiableProgram  # noqa: E402
from npu_model.util.converter import stream_to_instrs  # noqa: E402

PROGRAMS = MODEL / "calibration/dma/programs"
MEASUREMENTS = MODEL / "calibration/dma/measurements/ee290sim_vcs.json"


# ---------------------------------------------------------------------------
# Program directives
# ---------------------------------------------------------------------------


def directive(source: str, name: str, default: int) -> int:
    match = re.search(rf"^\s*#\s*@{name}\s+(\S+)", source, re.MULTILINE | re.IGNORECASE)
    return int(match.group(1), 0) if match else default


# ---------------------------------------------------------------------------
# Model execution
# ---------------------------------------------------------------------------


def make_config(backend: str | None, params: dict) -> DefaultHardwareConfig:
    cfg = DefaultHardwareConfig()
    if backend is not None and backend != cfg.dma_memory_backend:
        # The default params belong to the default backend; start another kind clean.
        cfg.dma_memory_backend = backend
        cfg.dma_memory_params = {}
    else:
        cfg.dma_memory_params = dict(cfg.dma_memory_params)
    for key, value in params.items():
        if cfg.dma_memory_backend == "fixed" and key in ("latency", "cycles_per_beat"):
            setattr(cfg, {"latency": "dma_memory_latency_cycles",
                          "cycles_per_beat": "dma_memory_cycles_per_beat"}[key], int(value))
        else:
            cfg.dma_memory_params[key] = value
    return cfg


def model_cycles(asm_path: Path, backend: str | None, params: dict) -> int:
    source = asm_path.read_text()
    program = InstantiableProgram(stream_to_instrs(io.StringIO(source)))
    program.memory_regions = []
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        trace = handle.name
    try:
        sim = Simulation(make_config(backend, params), LoggerConfig(filename=trace),
                         program, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(max_cycles=directive(source, "TIMEOUT", 1_000_000))
        if not sim.core.arch_state.halted:
            raise RuntimeError(f"{asm_path.name} did not halt in the model")
        return sim.core.arch_state.read_csrf(0x801)
    finally:
        Path(trace).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def parse_logs(paths: list[Path]) -> dict[str, int]:
    measured = {}
    for path in paths:
        text = path.read_text(errors="replace")
        match = re.search(r"dbg1_cycles\s*=\s*(\d+)", text)
        if match:
            name = re.sub(r"^atlas_", "", path.stem)
            measured[name] = int(match.group(1))
    return measured


def parse_value(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def parse_range(text: str) -> list:
    """``lo:hi[:step]`` (inclusive) or a comma-separated list; floats allowed."""
    if ":" in text:
        parts = [parse_value(part) for part in text.split(":")]
        lo, hi = parts[0], parts[1]
        step = parts[2] if len(parts) > 2 else 1
        values, value = [], lo
        while value <= hi + 1e-9:
            values.append(round(value, 6))
            value += step
        return values
    return [parse_value(value) for value in text.split(",")]


def parse_assignments(items: list[str], ranges: bool) -> dict:
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        result[key] = parse_range(value) if ranges else parse_value(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asm", nargs="*", type=Path,
                        help=".S files (default: calibration/dma/programs/perf_dma_*.S, probes excluded)")
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS,
                        help="JSON of measured dbg1_cycles (default: the bundled VCS measurements)")
    parser.add_argument("--measurement-set", default="refresh_off",
                        help="Which set of the measurements file to use (default: refresh_off)")
    parser.add_argument("--logs", nargs="*", type=Path, help="VCS logs to read dbg1_cycles from")
    parser.add_argument("--log-dir", type=Path, help="Directory of atlas_<name>.log files")
    parser.add_argument("--measured", nargs="*", default=[], metavar="NAME=CYCLES")
    parser.add_argument("--backend", choices=("fixed", "curve"), help="Backend kind (default: config default)")
    parser.add_argument("--set", nargs="*", default=[], metavar="PARAM=VALUE",
                        help="Backend parameters, e.g. curves=ee290sim_vcs_probe.json window_beats=16")
    parser.add_argument("--sweep", nargs="*", default=[], metavar="PARAM=LO:HI[:STEP]",
                        help="Grid-search these parameters (others from --set)")
    parser.add_argument("--fit", nargs="*", help="Program names used to score the sweep (default: all)")
    args = parser.parse_args()

    programs = args.asm or sorted(path for path in PROGRAMS.glob("perf_dma_*.S")
                                  if not path.name.startswith("perf_dma_probe_"))
    measured: dict[str, int] = {}
    if args.measurements and args.measurements.exists():
        import json
        values = json.loads(args.measurements.read_text())["dbg1_cycles"][args.measurement_set]["values"]
        measured.update({path.stem: int(values[path.stem]) for path in programs if path.stem in values})
    logs = list(args.logs or [])
    if args.log_dir is not None:
        logs += [args.log_dir / f"atlas_{path.stem}.log" for path in programs
                 if (args.log_dir / f"atlas_{path.stem}.log").exists()]
    measured.update(parse_logs(logs))
    for item in args.measured:
        name, value = item.split("=")
        measured[name] = int(value)

    fixed = parse_assignments(args.set, ranges=False)

    def evaluate(params, selected=programs):
        return {path.stem: model_cycles(path, args.backend, params) for path in selected}

    def describe(params):
        from npu_model.hardware.memory_backend import make_memory_backend
        backend = make_memory_backend(make_config(args.backend, params))
        public = {k: v for k, v in vars(backend).items() if not k.startswith("_") and not callable(v)}
        return f"{type(backend).__name__} {public}"

    def report(results, title):
        print(title)
        print(f"  {'program':32} {'rtl':>8} {'model':>8} {'error':>8}")
        total = 0
        for name, cycles in results.items():
            rtl = measured.get(name)
            error = f"{cycles - rtl:+d}" if rtl is not None else "n/a"
            total += abs(cycles - rtl) if rtl is not None else 0
            print(f"  {name:32} {rtl if rtl is not None else '-':>8} {cycles:>8} {error:>8}")
        print(f"  total |error| over measured programs: {total}")

    if args.sweep:
        grid = parse_assignments(args.sweep, ranges=True)
        fit = set(args.fit) if args.fit else set(measured)
        if not fit:
            parser.error("a sweep needs measurements (--logs, --log-dir or --measured)")
        fit_programs = [path for path in programs if path.stem in fit]
        best = None
        for values in itertools.product(*grid.values()):
            params = {**fixed, **dict(zip(grid, values))}
            results = evaluate(params, fit_programs)
            error = sum(abs(results[name] - measured[name]) for name in results if name in measured)
            print(f"{dict(zip(grid, values))}  total |error| = {error}")
            if best is None or error < best[0]:
                best = (error, params)
        error, params = best
        print(f"\nbest: {describe(params)}")
        report(evaluate(params), "all programs at the best setting:")
        return

    report(evaluate(fixed), f"backend: {describe(fixed)}")


if __name__ == "__main__":
    main()

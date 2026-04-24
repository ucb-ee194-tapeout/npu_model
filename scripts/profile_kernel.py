#!/usr/bin/env python3
"""
NPU Performance Model - Kernel Profiler

Usage:
    uv run scripts/profile_kernel.py [program] [options]

Options:
    program          Program class name (omit to list all)
    --max-cycles     Maximum cycles to simulate
    --hardware       HardwareConfig class name
    --list           List all available programs and exit
"""

import argparse
import tempfile
from pathlib import Path

# hardware must be imported before programs to avoid circular import
import npu_model.hardware  # noqa: F401
import npu_model.configs.programs as programs
import npu_model.configs.hardware as hw_configs
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.profiling import print_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile an NPU kernel.")
    parser.add_argument("program", nargs="?", help="Program class name")
    parser.add_argument("--max-cycles", type=int, default=200000)
    parser.add_argument("--hardware", default="DefaultHardwareConfig",
                        help="HardwareConfig class name")
    parser.add_argument("--list", action="store_true",
                        help="List all available programs and exit")
    args = parser.parse_args()

    if args.list or args.program is None:
        names = sorted(getattr(programs, "__all__", []))
        print(f"{len(names)} programs available:")
        for n in names:
            print(f"  {n}")
        return

    prog_cls = getattr(programs, args.program, None)
    if prog_cls is None:
        print(f"Unknown program '{args.program}'. Run with --list to see options.")
        raise SystemExit(1)

    hw_cls = getattr(hw_configs, args.hardware, None)
    if hw_cls is None:
        print(f"Unknown hardware config '{args.hardware}'.")
        raise SystemExit(1)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = f.name

    sim = None
    try:
        print(f"Program : {args.program}")
        print(f"Hardware: {args.hardware}")
        print()

        sim = Simulation(
            hardware_config=hw_cls(),
            logger_config=LoggerConfig(filename=trace_path),
            program=prog_cls(),
            verbose=False,
            record_timeline=True,
        )
        sim.run(max_cycles=args.max_cycles)
        print_stats(sim)
    finally:
        Path(trace_path).unlink(missing_ok=True)
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()

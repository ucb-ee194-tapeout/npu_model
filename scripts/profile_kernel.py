#!/usr/bin/env python3
"""
NPU Performance Model - Kernel Profiler

Usage:
    uv run scripts/profile_kernel.py [program] [options]

    program  can be a Python program class name (e.g. SmolVLAGeluTanhProgram)
             or a path to an assembly file (e.g. npu_model/configs/programs/asm/smolvla_gelu_tanh.S)

Options:
    --output         Output trace file (default: trace.json); open in https://ui.perfetto.dev
    --max-cycles     Maximum cycles to simulate
    --hardware       HardwareConfig class name
    --list           List all available Python programs and exit
"""

import argparse
from pathlib import Path

# hardware must be imported before programs to avoid circular import
import npu_model.hardware  # noqa: F401
import npu_model.configs.programs as programs
import npu_model.configs.hardware as hw_configs
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.profiling import print_stats
from npu_model.util.converter import load_asm
from npu_model.software.program import InstantiableProgram


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile an NPU kernel.")
    parser.add_argument("program", nargs="?", help="Program class name or .S file path")
    parser.add_argument("-o", "--output", default="trace.json",
                        help="Output trace file for Perfetto/Speedscope (default: trace.json)")
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

    hw_cls = getattr(hw_configs, args.hardware, None)
    if hw_cls is None:
        print(f"Unknown hardware config '{args.hardware}'.")
        raise SystemExit(1)

    asm_path = Path(args.program)
    if asm_path.suffix == ".S":
        if not asm_path.exists():
            print(f"Assembly file not found: {asm_path}")
            raise SystemExit(1)
        prog = InstantiableProgram(load_asm(asm_path))
        label = str(asm_path)
    else:
        prog_cls = getattr(programs, args.program, None)
        if prog_cls is None:
            print(f"Unknown program '{args.program}'. Run with --list to see options.")
            raise SystemExit(1)
        prog = prog_cls()
        label = args.program

    sim = None
    try:
        print(f"Program : {label}")
        print(f"Hardware: {args.hardware}")
        print()

        sim = Simulation(
            hardware_config=hw_cls(),
            logger_config=LoggerConfig(filename=args.output),
            program=prog,
            verbose=False,
            record_timeline=True,
        )
        sim.run(max_cycles=args.max_cycles)
        print_stats(sim)
        print(f"Trace   : {args.output}  (open at https://ui.perfetto.dev or https://www.speedscope.app)")
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()

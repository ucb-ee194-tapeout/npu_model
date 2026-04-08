#!/usr/bin/env python3
"""
NPU Performance Model - Simulation Runner

Usage:
    uv run scripts/run.py [options]

Options:
    -o, --output    Output trace file
    --logger        Logger backend: kanata or perfetto
    --max-cycles    Maximum cycles to simulate
"""

import argparse

import torch

import npu_model
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation

from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Performance Model Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run.py
    python scripts/run.py -o my_trace.json
    python scripts/run.py --max-cycles 5000
        """,
    )
    parser.add_argument(
        "--hardware_config",
        type=str,
        default="DefaultHardwareConfig",
        help="Hardware configuration",
    )
    parser.add_argument(
        "-p",
        "--program",
        type=str,
        default="AddiProgram",
        help="Program to run",
    )
    parser.add_argument(
        "-o", "--output", default="trace.json", help="Output trace file"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=10000, help="Maximum cycles to simulate"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NPU Performance Model - Simulation")
    print("=" * 60)

    # Create simulation environment
    try:
        hardware_config = eval(args.hardware_config)()
    except NameError:
        print(f"Hardware config '{args.hardware_config}' not found.")
        print("available options are:")
        print(f"  {', '.join(npu_model.configs.hardware.__all__)}")
        return
    try:
        program = eval(args.program)()
    except NameError:
        print(f"Program '{args.program}' not found.")
        print("available options are:")
        print(f"  {', '.join(npu_model.configs.programs.__all__)}")
        return
    sim = Simulation(
        hardware_config=hardware_config,
        logger_config=LoggerConfig(filename=args.output),
        program=program,
    )
    sim.run(max_cycles=args.max_cycles)

    if hasattr(program, "golden_result") and program.golden_result:
        output_base, golden_tensor = program.golden_result
        size = golden_tensor.numel() * golden_tensor.element_size()
        print(
            sim.core.arch_state.read_dram(output_base, size).view(golden_tensor.dtype)
        )


if __name__ == "__main__":
    main()

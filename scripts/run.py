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

import model_npu
from model_npu.logging import LoggerConfig
from model_npu.simulation import Simulation

from model_npu.configs.programs import *  # noqa: F401, F403
from model_npu.configs.hardware import *  # noqa: F401, F403
from model_npu.configs.isa_definition import *  # noqa: F401, F403


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
        print(f"  {', '.join(model_npu.configs.hardware.__all__)}")
        return
    try:
        program = eval(args.program)()
    except NameError:
        print(f"Program '{args.program}' not found.")
        print("available options are:")
        print(f"  {', '.join(model_npu.configs.programs.__all__)}")
        return
    sim = Simulation(
        hardware_config=hardware_config,
        logger_config=LoggerConfig(filename=args.output),
        program=program,
    )
    sim.run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()

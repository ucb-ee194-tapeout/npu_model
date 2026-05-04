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
from pathlib import Path

import npu_model
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import Program
from npu_model.util.converter import input_to_program
from npu_model.util.importjson import load_json

from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Performance Model Assembler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run scripts/run.py
    uv run scripts/run.py -o my_trace.json
    uv run scripts/run.py -p input.S -m memory.json
    uv run scripts/run.py --max-cycles 5000
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
        "-m",
        "--memory",
        type=str,
        default="",
        help="Path to memory and golden result JSON. Only used if -p does not refer to an in-memory program."
    )
    parser.add_argument(
        "-o", "--output", default="trace.json", help="Output trace file"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=10000, help="Maximum cycles to simulate"
    )
    parser.add_argument(
        "--ignore-runtime-errors",
        action="store_true",
        help="Bypass runtime assertions/exceptions, print bright red warnings, and continue execution",
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
        print("Available options are:")
        print(f"  {', '.join(npu_model.configs.hardware.__all__)}") # type: ignore
        return


    # Try getting the program internally
    try:
        if args.program.startswith(".") or args.program.startswith("/"):
            raise NameError("This is a path")
        program: Program = eval(args.program)()
    except NameError:
        try:
            # If that doesn't work, try opening it as a file and parsing it
            with open(args.program) as p:
                memory_regions: list[tuple[int, torch.Tensor]] = []
                golden_result: list[tuple[int, torch.Tensor]] = []
                timeout: int | None = 10000
                if args.memory != "":
                    try:
                        program_data = load_json(Path(args.memory))
                    except (FileNotFoundError, ValueError) as e:
                        print(f"Error accessing memory file (-m):\n{e}")
                        return
                    memory_regions = program_data.memory_regions
                    golden_result = program_data.golden_result
                    timeout = program_data.timeout

                program = input_to_program(p, memory_regions, golden_result, timeout)

        except (FileNotFoundError, ValueError) as e:
            print(f"Program '{args.program}' not found.")
            print("available options are a path or:")
            print(f"  {', '.join(npu_model.configs.programs.__all__)}")  # type: ignore
            return
        
    sim = Simulation(
        hardware_config=hardware_config,
        logger_config=LoggerConfig(filename=args.output),
        program=program,
        ignore_runtime_errors=args.ignore_runtime_errors,
    )
    sim.run(max_cycles=args.max_cycles)

    if hasattr(program, "golden_result") and program.golden_result and sim.core is not None:
        for result in program.golden_result:
            output_base, golden_tensor = result
            size = golden_tensor.numel() * golden_tensor.element_size()
            print(
                sim.core.arch_state.read_dram(output_base, size).view(golden_tensor.dtype)
            )


if __name__ == "__main__":
    main()

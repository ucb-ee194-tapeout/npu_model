#!/usr/bin/env python3
"""
NPU Performance Model - Delay Annotator

Usage:
    uv run scripts/annotate_delays.py [options]

Options:
    -p, --program   Program to annotate
    -o, --output    Output annotated .S file
"""

import argparse

import npu_model
from npu_model.util.converter import input_to_program, program_to_asm
from npu_model.util.delay_annotate import annotate_delays

from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Performance Model Delay Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/annotate_delays.py -p AddiProgram
    python scripts/annotate_delays.py -p kernel.S -o kernel.annotated.S
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
        help="Program to annotate",
    )
    parser.add_argument(
        "-o", "--output", default="annotated.S", help="Output annotated .S file"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=10000, help="Maximum cycles to simulate while scheduling"
    )

    args = parser.parse_args()

    try:
        hardware_config = eval(args.hardware_config)()
    except NameError:
        print(f"Hardware config '{args.hardware_config}' not found.")
        print("available options are:")
        print(f"  {', '.join(npu_model.configs.hardware.__all__)}")  # type: ignore
        return
    try:
        program = eval(args.program)()
    except NameError:
        try:
            # If that doesn't work, try opening it as a file and parsing it
            with open(args.program) as f:
                program = input_to_program(f)

        except NameError:
            print(f"Program '{args.program}' not found.")
            print("available options are a .S file or:")
            print(f"  {', '.join(npu_model.configs.programs.__all__)}")  # type: ignore
            return

    annotated = annotate_delays(program, hardware_config, max_cycles=args.max_cycles)

    with open(args.output, "w") as f:
        f.write(program_to_asm(annotated))
    print(
        f"Wrote {len(annotated)} instructions "
        f"({len(annotated) - len(program)} delays inserted) to {args.output}"
    )


if __name__ == "__main__":
    main()

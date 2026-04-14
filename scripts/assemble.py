#!/usr/bin/env python3
"""
NPU Performance Model - Assembler

Usage:
    uv run scripts/assemble.py [options]

Options:
    -p, --program   Program
    --out-hex       Output File (hex)
    --out-hex       Output File (binary)
"""

import argparse
import npu_model
import struct

from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.util.converter import input_to_program

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Performance Model Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/assemble.py
    python scripts/assemble.py -p AddiProgram
    python scripts/assemble.py -p input.S
    python scripts/assemble.py --out-bin out1 --out-hex out2
        """,
    )
    parser.add_argument(
        "-p",
        "--program",
        type=str,
        default="AddiProgram",
        help="Program to run",
    )
    parser.add_argument(
        "--out-hex", help="Output hex file"
    )
    parser.add_argument(
        "--out-bin", help="Output binary file"
    )

    args = parser.parse_args()

    # Try getting the program internally
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
            print(f"  {', '.join(npu_model.configs.programs.__all__)}")
            return

    code = program.assemble()

    if args.out_hex:
        with open(args.out_hex, "w") as f:
            for word in code:
                f.write(f"{word & 0xFFFFFFFF:08x}\n")
        print(f"Wrote {len(code)} words to {args.out_hex}")

    if args.out_bin:
        with open(args.out_bin, "wb") as f:
            for word in code:
                f.write(struct.pack("<I", word & 0xFFFFFFFF))
        print(f"Wrote {len(code)} words ({len(code) * 4} bytes) to {args.out_bin}")

    if not args.out_hex and not args.out_bin:
        print(f"Assembled {len(code)} instructions from {args.program}")
        print()
        for i, word in enumerate(code):
            print(f"  [{i:4d}]  0x{word & 0xFFFFFFFF:08x}")

if __name__ == "__main__":
    main()

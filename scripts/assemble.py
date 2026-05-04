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
import os
import re
import struct

from pathlib import Path
import npu_model
from npu_model.configs.programs import *  # noqa: F401, F403
from npu_model.configs.hardware import *  # noqa: F401, F403
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.software.program import Program
from npu_model.util.converter import input_to_program
from npu_model.util.templates.pyprogram import format_python_file
from npu_model.util.templates.cprogram import format_c_header_file
from npu_model.util.importjson import load_json

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Performance Model Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run scripts/assemble.py
    uv run scripts/assemble.py -p AddiProgram
    uv run scripts/assemble.py -p input.S -m memory.json
    uv run scripts/assemble.py --out-bin out1 --out-hex out2
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
        "-m",
        "--memory",
        type=str,
        default="",
        help="Path to memory and golden result JSON. Only used if -p does not refer to an in-memory program."
    )
    parser.add_argument(
        "--out-py", help="Output python file"
    )
    parser.add_argument(
        "--out-c", help="Output c header file"
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
                    print("wtf2")
                    try:
                        program_data = load_json(Path(args.memory))
                    except (FileNotFoundError, ValueError) as e:
                        print(f"Error accessing memory file (-m):\n{e}")
                        return
                    memory_regions = program_data.memory_regions
                    golden_result = program_data.golden_result
                    timeout = program_data.timeout

                program = input_to_program(p, memory_regions, golden_result, timeout)
        except (FileNotFoundError, ValueError):
            print(f"Program '{args.program}' not found.")
            print("available options are a path or:")
            print(f"  {', '.join(npu_model.configs.programs.__all__)}")  # type: ignore
            return

    code = program.assemble()

    if args.out_py:
        with open(args.out_py, "w") as f:
            name = os.path.splitext(args.program)[0]
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            name = ''.join(word.capitalize() for word in name.split('_'))
            name = 'P' + name if name[0].isdigit() else name

            f.write(format_python_file(name, args.program, program.timeout, program.instructions, program.memory_regions, program.golden_result))

    if args.out_c:
        with open(args.out_c, "w") as f:
            name: str = os.path.splitext(args.program)[0]
            name = os.path.basename(name)
            name = 'program_' + name if name[0].isdigit() else name

            f.write(format_c_header_file(name, args.program, program.timeout, code, program.memory_regions, program.golden_result))

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

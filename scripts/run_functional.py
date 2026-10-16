#!/usr/bin/env python3
"""Run a functional assembly (.fs) program."""

import argparse

import torch

from npu_model.functional import (
    FunctionalConfig,
    FunctionalInterpreter,
    load_functional_assembly,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run functional assembly (.fs)")
    parser.add_argument("program", help="path to a .fs source file")
    parser.add_argument("--max-instructions", type=int, default=1_000_000)
    parser.add_argument("--numerics", choices=("pytorch", "rtl"), default="rtl")
    args = parser.parse_args()

    if not args.program.endswith(".fs"):
        parser.error("functional assembly source files must use the .fs extension")
    program = load_functional_assembly(args.program)
    interpreter = FunctionalInterpreter(FunctionalConfig(numerics=args.numerics))
    result = interpreter.run(program, max_instructions=args.max_instructions)
    print(f"Executed {result.instructions_executed} functional instructions")
    print(f"Final PC: {result.final_pc}")
    if result.halt_reason is not None:
        print(f"Halt reason: {result.halt_reason}")
    for register_class, values in result.named_registers(program).items():
        rendered = []
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                value = f"<{tuple(value.shape)} {value.dtype}>"
            rendered.append(f"{name}={value}")
        print(f"{register_class}: {', '.join(rendered)}")


if __name__ == "__main__":
    main()

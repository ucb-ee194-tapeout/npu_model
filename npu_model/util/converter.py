import torch
import re
from typing import TextIO, List
from pathlib import Path

from ..software.instruction import Instruction, x
from ..software.program import InstantiableProgram
from ..isa import IsaSpec
from ..configs.isa_definition import ADDI, LUI
from ..isa_types import ScalarReg

def parse_reg(s: str):
    s = s.strip().rstrip(",").lower()
    if not s.startswith("x"):
        raise ValueError(f"Invalid register: {s}")
    n = int(s[1:])
    if not 0 <= n <= 31:
        raise ValueError(f"Register out of range: {s}")
    return n


def expand_li(rd: int, value: int) -> List[Instruction]:
    """Expand LI pseudo-instruction into LUI+ADDI (32-bit)."""
    if rd < 0 or rd > 31:
        raise ValueError("Invalid RD provided for LI")
    v = value & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    if -2048 <= v <= 2047:
        return [
            ADDI(rd=ScalarReg(rd), rs1=x(0), imm=(v & 0xFFF))
        ]
    lo12 = v & 0xFFF
    if lo12 & 0x800:
        lo12 -= 0x1000
    hi20 = ((v - lo12) >> 12) & 0xFFFFF
    if lo12 == 0:
        return [LUI(rd=ScalarReg(rd), imm=hi20)]
    return [
        LUI(rd=ScalarReg(rd), imm=hi20),
        ADDI(rd=ScalarReg(rd), rs1=ScalarReg(rd), imm=(lo12 & 0xFFF))
    ]

def strip_comment(line: str):
    idx = line.find("#")
    return line[:idx].strip() if idx >= 0 else line.strip()


def tokenize(line: str):
    return [t.rstrip(',') for t in re.split(r"[\s,]+", strip_comment(line)) if t]

def stream_to_instrs(source: TextIO) -> list[Instruction]:
    lines: list[str] = []
    labels: dict[str, int] = {}
    addr: int = 0

    # Pass 1: Figure out where labels are
    for raw_line in source:
        line = strip_comment(raw_line)

        if not line:
            pass

        lines.append(line)
        if line.endswith(":"):
            labels[line[:-1].strip()] = addr
        else:
            tokens = tokenize(line)
            if tokens:
                if tokens[0].lower() == "li":
                    if len(tokens) > 2:
                        addr += len(expand_li(0, int(tokens[2], 0)))
                    else:
                        raise ValueError(f"Malformed Instruction: {line}")
                else:
                    addr += 1

    # Pass 2: Produce a program
    instructions: list[Instruction] = []
    pc = 0

    def resolve(s: str):
        if s in labels:
            return (labels[s] - pc) * 4
        return int(s, 0)

    for line in lines:
        if line.endswith(":"):
            continue

        tokens = tokenize(line)
        if not tokens:
            continue

        mnemonic = tokens[0].lower()

        # Handle pseudoinstructions
        if mnemonic == "nop" and len(tokens) == 1:
            instructions.append(ADDI(x(0), x(0), 0))
            pc += 1
        elif mnemonic == "li" and len(tokens) == 3:
            e = expand_li(parse_reg(tokens[1]), int(tokens[2], 0))
            instructions.extend(e)
            pc += len(e)
        else:
            try:
                # mnemonic should be lowercase
                tokens[0] = tokens[0].lower()
                instr = IsaSpec.operations[mnemonic]

                if len(err := instr.lint(tokens, labels=list(labels.keys()))) != 0:
                    raise ExceptionGroup(f"Error assembling isntr: {", ".join(tokens)}", err)

                instructions.append(instr.from_asm(tokens, resolve))
                pc += 1
            except KeyError:
                raise ValueError(f"Invalid mnemonic provided: {line}")
    
    return instructions

def load_asm(source: Path):
    with open(source) as f:
        return stream_to_instrs(f)

def input_to_program(source: TextIO, memory_regions: list[tuple[int, torch.Tensor]] = [], golden_result: list[tuple[int, torch.Tensor]] = [], timeout: int | None = 10000):
    return InstantiableProgram(stream_to_instrs(source), memory_regions, golden_result, timeout)
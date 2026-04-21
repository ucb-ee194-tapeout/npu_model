import re
from typing import TextIO

from ..software.instruction import Instruction
from ..software.program import InstantiableProgram
from ..isa import InstructionType, ScalarArgs, VectorArgs, MatrixArgs, DmaArgs


MEM_OPERAND_RE = re.compile(r"^(?P<imm>.+)\((?P<rs1>x\d+)\)$")
DMA_CHANNEL_RE = re.compile(
    r"^dma\.(?P<op>load|store|config|wait)\.ch(?P<channel>\d+)$"
)

VECTOR_VR_BINARY_MNEMONICS = {
    "vadd.bf16",
    "vsub.bf16",
    "vmul.bf16",
    "vminimum.bf16",
    "vmaximum.bf16",
    "vmax.bf16",
    "vmin.bf16",
}

MATRIX_VR_TRANSFER_MNEMONICS = {
    "vmatpush.weight.mxu0",
    "vmatpush.weight.mxu1",
    "vmatpush.acc.fp8.mxu0",
    "vmatpush.acc.fp8.mxu1",
    "vmatpush.acc.bf16.mxu0",
    "vmatpush.acc.bf16.mxu1",
    "vmatpop.fp8.acc.mxu0",
    "vmatpop.fp8.acc.mxu1",
    "vmatpop.bf16.acc.mxu0",
    "vmatpop.bf16.acc.mxu1",
}

MATRIX_VR_MATMUL_MNEMONICS = {
    "vmatmul.mxu0",
    "vmatmul.mxu1",
    "vmatmul.acc.mxu0",
    "vmatmul.acc.mxu1",
}


def parse_reg(s: str):
    s = s.strip().rstrip(",").lower()
    if not s.startswith("x"):
        raise ValueError(f"Invalid register: {s}")
    n = int(s[1:])
    if not 0 <= n <= 31:
        raise ValueError(f"Register out of range: {s}")
    return n


def parse_imm(s: str):
    s = s.strip().rstrip(",").lower()
    if s.startswith("0b") or s.startswith("-0b"):
        return int(s, 2)
    if s.startswith("0x") or s.startswith("-0x"):
        return int(s, 16)
    return int(s)


def parse_mem_operand(s: str):
    match = MEM_OPERAND_RE.match(s.strip().rstrip(",").lower())
    if match is None:
        raise ValueError(f"Invalid memory operand: {s}")
    return parse_imm(match.group("imm")), parse_reg(match.group("rs1"))


def parse_dma_channel_mnemonic(mnemonic: str) -> tuple[str, int] | None:
    match = DMA_CHANNEL_RE.match(mnemonic)
    if match is None:
        return None
    channel = int(match.group("channel"))
    if not 0 <= channel <= 7:
        raise ValueError(f"DMA channel out of range: {mnemonic}")
    return match.group("op"), channel


def expand_li(rd: int, value: int):
    """Expand LI pseudo-instruction into LUI+ADDI (32-bit)."""
    v = value & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    if -2048 <= v <= 2047:
        return [
            Instruction(
                mnemonic="addi",
                args=ScalarArgs(rd=rd, rs1=0, imm=(v & 0xFFF)),
            )
        ]
    lo12 = v & 0xFFF
    if lo12 & 0x800:
        lo12 -= 0x1000
    hi20 = ((v - lo12) >> 12) & 0xFFFFF
    if lo12 == 0:
        return [Instruction(mnemonic="lui", args=ScalarArgs(rd=rd, imm=hi20))]
    return [
        Instruction(mnemonic="lui", args=ScalarArgs(rd=rd, imm=hi20)),
        Instruction(
            mnemonic="addi",
            args=ScalarArgs(rd=rd, rs1=rd, imm=(lo12 & 0xFFF)),
        ),
    ]


def strip_comment(line: str):
    idx = line.find("#")
    return line[:idx].strip() if idx >= 0 else line.strip()


def tokenize(line: str):
    return [t for t in re.split(r"[\s,]+", strip_comment(line)) if t]


def input_to_program(source: TextIO):
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
                        addr += len(expand_li(0, parse_imm(tokens[2])))
                    else:
                        raise ValueError(f"Malformed Instruction: {line}")
                else:
                    addr += 1

    # Pass 2: Produce a program
    instructions: list[Instruction] = []
    pc = 0

    def resolve(s: str):
        if s in labels:
            return labels[s] - pc
        return parse_imm(s)

    for line in lines:
        if line.endswith(":"):
            continue

        tokens = tokenize(line)
        if not tokens:
            continue

        mnemonic = tokens[0].lower()

        # Handle pseudoinstructions
        if mnemonic == "nop" and len(tokens) == 1:
            instructions.append(Instruction(mnemonic="addi", args=ScalarArgs()))
            pc += 1
        elif mnemonic == "li" and len(tokens) == 3:
            expanded = expand_li(parse_reg(tokens[1]), parse_imm(tokens[2]))
            instructions.extend(expanded)
            pc += len(expanded)

        # Handle DMA families.
        elif parse_dma_channel_mnemonic(mnemonic) is not None:
            op, channel = parse_dma_channel_mnemonic(mnemonic)
            if op in {"load", "store"} and len(tokens) == 4:
                instructions.append(
                    Instruction(
                        mnemonic=f"dma.{op}.ch<N>",
                        args=DmaArgs(
                            rd=parse_reg(tokens[1]),
                            rs1=parse_reg(tokens[2]),
                            rs2=parse_reg(tokens[3]),
                            channel=channel,
                        ),
                    )
                )
            elif op == "config" and len(tokens) == 2:
                instructions.append(
                    Instruction(
                        mnemonic="dma.config.ch<N>",
                        args=DmaArgs(rs1=parse_reg(tokens[1]), channel=channel),
                    )
                )
            elif op == "wait" and len(tokens) == 1:
                instructions.append(
                    Instruction(
                        mnemonic="dma.wait.ch<N>",
                        args=DmaArgs(channel=channel),
                    )
                )
            else:
                raise ValueError(f"Malformed Instruction: {line}")

        # Handle shifts/ecall/ebreak/fence (custom immediate values only based on name)
        elif (mnemonic == "srli" or mnemonic == "slli") and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(
                        rd=parse_reg(tokens[1]),
                        rs1=parse_reg(tokens[2]),
                        imm=(parse_imm(tokens[3]) & 0x1F),
                    ),
                )
            )
        elif mnemonic == "srai" and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(
                        rd=parse_reg(tokens[1]),
                        rs1=parse_reg(tokens[2]),
                        imm=((parse_imm(tokens[3]) & 0x1F) | 0x400),
                    ),
                )
            )
        elif (mnemonic == "fence" or mnemonic == "ecall") and len(tokens) == 1:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs()))
        elif mnemonic == "ebreak" and len(tokens) == 1:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(imm=0b000000000001),
                )
            )

        # Handle scalar I-type instructions.
        elif (
            mnemonic in InstructionType.SCALAR.I.mnemonics
            or mnemonic in InstructionType.DELAY.I.mnemonics
        ) and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(
                        rd=parse_reg(tokens[1]),
                        rs1=parse_reg(tokens[2]),
                        imm=parse_imm(tokens[3]),
                    ),
                )
            )
        elif mnemonic in InstructionType.SCALAR.I.mnemonics and len(tokens) == 3:
            imm, rs1 = parse_mem_operand(tokens[2])
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(rd=parse_reg(tokens[1]), rs1=rs1, imm=imm),
                )
            )

        # Handle scalar R-type instructions.
        elif mnemonic in InstructionType.SCALAR.R.mnemonics and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(
                        rd=parse_reg(tokens[1]),
                        rs1=parse_reg(tokens[2]),
                        rs2=parse_reg(tokens[3]),
                    ),
                )
            )

        # Handle scalar S-type instructions.
        elif mnemonic in InstructionType.SCALAR.S.mnemonics and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(
                        rs2=parse_reg(tokens[1]),
                        rs1=parse_reg(tokens[2]),
                        imm=parse_imm(tokens[3]),
                    ),
                )
            )
        elif mnemonic in InstructionType.SCALAR.S.mnemonics and len(tokens) == 3:
            imm, rs1 = parse_mem_operand(tokens[2])
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(rs2=parse_reg(tokens[1]), rs1=rs1, imm=imm),
                )
            )

        # Handle scalar SB-type instructions.
        elif mnemonic in InstructionType.SCALAR.SB.mnemonics and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(
                        rs1=parse_reg(tokens[1]),
                        rs2=parse_reg(tokens[2]),
                        imm=resolve(tokens[3]),
                    ),
                )
            )

        # Handle scalar U-type instructions.
        elif mnemonic in InstructionType.SCALAR.U.mnemonics and len(tokens) == 3:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(rd=parse_reg(tokens[1]), imm=parse_imm(tokens[2])),
                )
            )

        # Handle scalar UJ-type instructions.
        elif mnemonic in InstructionType.SCALAR.UJ.mnemonics and len(tokens) == 3:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=ScalarArgs(rd=parse_reg(tokens[1]), imm=resolve(tokens[2])),
                )
            )

        # Handle vector VLS-type instructions.
        elif mnemonic in InstructionType.VECTOR.VLS.mnemonics and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=VectorArgs(
                        vd=parse_reg(tokens[1]),
                        rs1=parse_reg(tokens[2]),
                        imm12=parse_imm(tokens[3]),
                    ),
                )
            )
        elif mnemonic in InstructionType.VECTOR.VLS.mnemonics and len(tokens) == 3:
            imm, rs1 = parse_mem_operand(tokens[2])
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=VectorArgs(vd=parse_reg(tokens[1]), rs1=rs1, imm12=imm),
                )
            )

        # Handle vector VR-type instructions.
        elif mnemonic in VECTOR_VR_BINARY_MNEMONICS and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=VectorArgs(
                        vd=parse_reg(tokens[1]),
                        vs1=parse_reg(tokens[2]),
                        vs2=parse_reg(tokens[3]),
                    ),
                )
            )
        elif mnemonic in InstructionType.VECTOR.VR.mnemonics and len(tokens) == 3:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=VectorArgs(vd=parse_reg(tokens[1]), vs1=parse_reg(tokens[2])),
                )
            )
        elif mnemonic in InstructionType.VECTOR.VR.mnemonics:
            raise ValueError(f"Malformed Instruction: {line}")

        # Handle vector VI-type instructions.
        elif mnemonic in InstructionType.VECTOR.VI.mnemonics and len(tokens) == 3:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=VectorArgs(vd=parse_reg(tokens[1]), imm=parse_imm(tokens[2])),
                )
            )

        # Handle matrix VR-type instructions.
        elif mnemonic in MATRIX_VR_TRANSFER_MNEMONICS and len(tokens) == 3:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=MatrixArgs(vd=parse_reg(tokens[1]), vs1=parse_reg(tokens[2])),
                )
            )
        elif mnemonic in MATRIX_VR_MATMUL_MNEMONICS and len(tokens) == 4:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=MatrixArgs(
                        vd=parse_reg(tokens[1]),
                        vs1=parse_reg(tokens[2]),
                        vs2=parse_reg(tokens[3]),
                    ),
                )
            )
        elif mnemonic in MATRIX_VR_MATMUL_MNEMONICS and len(tokens) == 3:
            instructions.append(
                Instruction(
                    mnemonic=mnemonic,
                    args=MatrixArgs(vs1=parse_reg(tokens[1]), vs2=parse_reg(tokens[2])),
                )
            )

        else:
            raise ValueError(f"Malformed Instruction: {line}")

    return InstantiableProgram(instructions)

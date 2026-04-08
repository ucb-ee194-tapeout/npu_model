import re
from typing import TextIO

from ..software.instruction import Instruction
from ..software.program import InstantiableProgram
from ..isa import InstructionType, ScalarArgs, VectorArgs, MatrixArgs, DmaArgs

# ─── Parsing helpers ────────────────────────────────────────────────────────

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
        return int(s,2)
    if s.startswith("0x") or s.startswith("-0x"):
        return int(s,16)
    return int(s)

def expand_li(rd: int, value: int):
    """Expand LI pseudo-instruction into LUI+ADDI (32-bit)."""
    v = value & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    if -2048 <= v <= 2047:
        return [Instruction(mnemonic="rd", args=ScalarArgs(rd=rd, imm=(v & 0xFFF)))]
    lo12 = v & 0xFFF
    if lo12 & 0x800:
        lo12 -= 0x1000
    hi20 = ((v - lo12) >> 12) & 0xFFFFF
    if lo12 == 0:
        return [Instruction(mnemonic="lui", args=ScalarArgs(rd=rd, imm=hi20))]
    return [Instruction(mnemonic="lui",args=ScalarArgs(rd=rd, imm=hi20)), Instruction(mnemonic="addi", args=ScalarArgs(rd=rd, rs2=rd, imm=(lo12 & 0xFFF)))]

def strip_comment(line: str):
    idx = line.find("#")
    return line[:idx].strip() if idx >= 0 else line.strip()

def tokenize(line: str):
    return [t for t in re.split(r"[\s,]+", strip_comment(line)) if t]

def input_to_program(source: TextIO):
    lines: list[str] = []
    labels: dict[str,int] = {}
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
                        addr += len(expand_li(0,parse_imm(tokens[2])))
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

        tokens=tokenize(line)
        if not tokens:
            continue

        mnemonic = tokens[0].lower()

        # Handle pseudoinstructions
        if mnemonic == "nop" and len(tokens) == 1:
            instructions.append(Instruction(mnemonic="addi", args=ScalarArgs()))
            pc += 1
        elif mnemonic == "li" and len(tokens) == 3:
            e = expand_li(parse_reg(tokens[1]), parse_imm(tokens[2]))
            instructions.extend(e)
            pc += len(e)
        
        # Handle weird representations (DMA R/I)
        elif ((mnemonic.startswith("dma.load.ch") and len(mnemonic) == 11) or
              (mnemonic.startswith("dma.store.ch") and len(mnemonic) == 13)) and mnemonic[-1].isdigit() and len(tokens) == 4:
            # FIXME: This allows any single-digit channel to work.
            # I'm leaving this in since I think this should be fixed in Instruction()
            channel = int(mnemonic[-1])
            instructions.append(Instruction(mnemonic=f"{mnemonic[:-1]}<N>", args=DmaArgs(rd=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), rs2=parse_reg(tokens[3]), channel=channel)))
        elif mnemonic.startswith("dma.config.ch") and len(mnemonic) == 14 and mnemonic[-1].isdigit() and len(tokens) == 2:
            channel = int(mnemonic[-1])
            instructions.append(Instruction(mnemonic=f"dma.config.ch<N>", args=DmaArgs(rs1=parse_reg(tokens[2]), channel=channel)))
        elif mnemonic.startswith("dma.wait.ch") and len(mnemonic) == 12 and mnemonic[-1].isdigit() and len(tokens) == 2:
            channel = int(mnemonic[-1])
            instructions.append(Instruction(mnemonic=f"dma.wait.ch<N>", args=DmaArgs(channel=channel)))
        
        # Handle shifts/ecall/ebreak/fence (custom immediate values only based on name)
        elif (mnemonic == "srli" or mnemonic == "slli") and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), imm=(parse_imm(tokens[3]) & 0x1f))))
        elif mnemonic == "srai" and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), imm=((parse_imm(tokens[3]) & 0x1f) | 0x400))))
        elif (mnemonic == "fence" or mnemonic == "ecall") and len(tokens) == 1:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs()))
        elif mnemonic == "ebreak" and len(tokens) == 1:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(imm=0b000000000001)))

        # Handle all scalar I-type instructions
        # FIXME: currently loads don't support the syntax of lw rd, offset(rs1) — they're just lw, rd, rs1, offset
        elif (mnemonic in InstructionType.SCALAR.I.mnemonics or mnemonic in InstructionType.DELAY.I.mnemonics) and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), imm=parse_imm(tokens[3]))))

        # Handle all scalar R-type instructions
        elif mnemonic in InstructionType.SCALAR.R.mnemonics and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), rs2=parse_reg(tokens[3]))))

        # Handle all scalar S-type instructions
        # FIXME: Same thing as loads.
        elif mnemonic in InstructionType.SCALAR.S.mnemonics and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rs2=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), imm=parse_imm(tokens[3]))))

        # Handle all scalar SB-type instructions
        elif mnemonic in InstructionType.SCALAR.SB.mnemonics and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rs1=parse_reg(tokens[1]), rs2=parse_reg(tokens[2]), imm=resolve(tokens[3]))))

        # Handle all scalar U-type instructions
        elif mnemonic in InstructionType.SCALAR.U.mnemonics and len(tokens) == 3:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), imm=parse_imm(tokens[2]))))

        # Handle all scalar UJ-type instructions
        elif mnemonic in InstructionType.SCALAR.UJ.mnemonics and len(tokens) == 3:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), imm=resolve(tokens[2]))))

        # Handle all vector VLS-type instructions
        elif mnemonic in InstructionType.VECTOR.VLS.mnemonics and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=VectorArgs(vd=parse_reg(tokens[1]), rs1=parse_reg(tokens[2]), imm12=parse_imm(tokens[3]))))

        # Handle all vector VR-type instructions
        # some instructions use vs2
        elif mnemonic in InstructionType.VECTOR.VR.mnemonics and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=VectorArgs(vd=parse_reg(tokens[1]), vs1=parse_reg(tokens[2]), vs2=parse_reg(tokens[3]))))

        # some don't
        # FIXME: we don't currently error if you try to use 3 arguments on a 2-argument instr. (and vice versa)
        elif mnemonic in InstructionType.VECTOR.VR.mnemonics and len(tokens) == 3:
            instructions.append(Instruction(mnemonic=mnemonic, args=VectorArgs(vd=parse_reg(tokens[1]), vs1=parse_reg(tokens[2]))))


        # Handle all vector VI-type instructions
        elif mnemonic in InstructionType.VECTOR.VI.mnemonics and len(tokens) == 3:
            instructions.append(Instruction(mnemonic=mnemonic, args=ScalarArgs(rd=parse_reg(tokens[1]), imm=parse_imm(tokens[2]))))

        # Handle all matrix VR-type instructions — only matmul so all use both args
        elif mnemonic in InstructionType.MATRIX_SYSTOLIC.VR.mnemonics or \
            mnemonic in InstructionType.MATRIX_IPT.VR.mnemonics and len(tokens) == 4:
            instructions.append(Instruction(mnemonic=mnemonic, args=MatrixArgs(vd=parse_reg(tokens[1]), vs1=parse_reg(tokens[2]), vs2=parse_reg(tokens[3]))))

        else:
            raise ValueError(f"Malformed Instruction: {line}")
    
    return InstantiableProgram(instructions)
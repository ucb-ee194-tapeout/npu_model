import pytest

from npu_model.configs.hardware import DefaultHardwareConfig
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.hardware.dma import (
    dma_offchip_cycles,
    dma_transfer_cycles,
    vmem_transfer_cycles,
)
from npu_model.isa import MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software.instruction import Instruction
from npu_model.software.program import InstantiableProgram
from tests.helpers import run_simulation


def encode_s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm &= 0xFFF
    return (
        (((imm >> 5) & 0x7F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((imm & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_sb_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm &= 0x1FFF
    return (
        (((imm >> 12) & 0x1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 0x1) << 7)
        | (opcode & 0x7F)
    )


def encode_vr_type(opcode: int, funct7: int, vd: int, vs1: int, vs2: int) -> int:
    return (
        ((funct7 & 0x7F) << 25)
        | ((vs2 & 0x3F) << 19)
        | ((vs1 & 0x3F) << 13)
        | ((vd & 0x3F) << 7)
        | (opcode & 0x7F)
    )


def test_representative_instruction_encodings_match_spec_bit_packing() -> None:
    cases = [
        (
            Instruction("sw", ScalarArgs(rs2=3, rs1=2, imm=16)),
            encode_s_type(0b0100011, 0b010, rs1=2, rs2=3, imm=16),
        ),
        (
            Instruction("beq", ScalarArgs(rs1=1, rs2=2, imm=-4)),
            encode_sb_type(0b1100011, 0b000, rs1=1, rs2=2, imm=-4),
        ),
        (
            Instruction("vadd.bf16", VectorArgs(vd=2, vs1=4, vs2=6)),
            encode_vr_type(0b1010111, 0b0000000, vd=2, vs1=4, vs2=6),
        ),
        (
            Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=1, vs2=0)),
            encode_vr_type(0b1110111, 0b0001010, vd=0, vs1=1, vs2=0),
        ),
    ]

    for insn, expected in cases:
        word = insn.assemble()
        assert 0 <= word < (1 << 32), f"{insn.mnemonic} assembled outside 32 bits"
        assert word == expected, (
            f"{insn.mnemonic} encoding mismatch: got 0x{word:08x}, "
            f"expected 0x{expected:08x}"
        )


def test_dma_transfer_latency_uses_offchip_and_vmem_formulas() -> None:
    cfg = DefaultHardwareConfig()

    assert dma_offchip_cycles(cfg, 1024) == 516
    assert vmem_transfer_cycles(cfg, 1024) == 16
    assert dma_transfer_cycles(cfg, 1024) == 516


def test_taken_branch_executes_exactly_two_delay_slots_before_redirect() -> None:
    program = InstantiableProgram(
        [
            Instruction("addi", ScalarArgs(rd=1, rs1=0, imm=1)),
            Instruction("addi", ScalarArgs(rd=4, rs1=0, imm=0)),
            Instruction("beq", ScalarArgs(rs1=1, rs2=1, imm=4)),
            Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=11)),
            Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=22)),
            Instruction("addi", ScalarArgs(rd=4, rs1=0, imm=99)),
            Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=44)),
        ]
    )

    sim = run_simulation(program, DefaultHardwareConfig(), max_cycles=64)

    assert sim.core.arch_state.read_xrf(2) == 11
    assert sim.core.arch_state.read_xrf(3) == 22
    assert sim.core.arch_state.read_xrf(4) == 0
    assert sim.core.arch_state.read_xrf(5) == 44


def test_control_flow_instruction_in_delay_slot_is_illegal() -> None:
    program = InstantiableProgram(
        [
            Instruction("addi", ScalarArgs(rd=1, rs1=0, imm=1)),
            Instruction("beq", ScalarArgs(rs1=1, rs2=1, imm=4)),
            Instruction("jal", ScalarArgs(rd=0, imm=2)),
            Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=22)),
            Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=33)),
            Instruction("addi", ScalarArgs(rd=4, rs1=0, imm=44)),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="Illegal control-flow instruction 'jal' decoded in a delay-slot position",
    ):
        run_simulation(program, DefaultHardwareConfig(), max_cycles=64)

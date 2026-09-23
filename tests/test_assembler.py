import io
import importlib.util
from pathlib import Path

import pytest

from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.util.converter import input_to_program
from npu_model.lsp.linter import lint_text


_baremetal_path = Path(__file__).resolve().parents[2] / "baremetal" / "assembler.py"
_baremetal_spec = importlib.util.spec_from_file_location("atlas_baremetal_assembler", _baremetal_path)
assert _baremetal_spec is not None and _baremetal_spec.loader is not None
_baremetal_assembler = importlib.util.module_from_spec(_baremetal_spec)
_baremetal_spec.loader.exec_module(_baremetal_assembler)


def test_li_expands_to_valid_addi_and_lui_addi_sequences() -> None:
    program = input_to_program(
        io.StringIO(
            """
            li x3, 5
            li x4, 0x12345
            """
        )
    )

    addi0 = program.instructions[0]
    assert isinstance(addi0, ADDI)
    assert addi0.rd == 3
    assert addi0.rs1 == 0
    assert addi0.imm == 5
    assert isinstance(program.instructions[1], LUI)
    addi2 = program.instructions[2]
    assert isinstance(addi2, ADDI)
    assert addi2.rs1 == 4


def test_dma_config_and_wait_parse_correctly() -> None:
    program = input_to_program(
        io.StringIO(
            """
            dma.config.ch0 x7
            dma.wait.ch1
            """
        )
    )

    config = program.instructions[0]
    assert isinstance(config, DMA_CONFIG_CH0)
    assert config.rs1 == 7
    assert isinstance(program.instructions[1], DMA_WAIT_CH1)


def test_offset_addressing_and_vi_immediates_parse_correctly() -> None:
    program = input_to_program(
        io.StringIO(
            """
            lw x1, 16(x2)
            sw x3, 20(x4)
            vload m5, 7(x6)
            vli.all m7, -3
            """
        )
    )

    lw = program.instructions[0]
    assert isinstance(lw, LW)
    assert lw.rd == 1
    assert lw.rs1 == 2
    assert lw.imm == 16
    sw = program.instructions[1]
    assert isinstance(sw, SW)
    assert sw.rs2 == 3
    assert sw.rs1 == 4
    assert sw.imm == 20
    vload = program.instructions[2]
    assert isinstance(vload, VLOAD)
    assert vload.vd == 5
    assert vload.rs1 == 6
    assert vload.imm == 7
    vli = program.instructions[3]
    assert isinstance(vli, VLI_ALL)
    assert vli.vd == 7
    assert vli.imm == -3


def test_matrix_transfer_and_matmul_parse_correctly() -> None:
    program = input_to_program(
        io.StringIO(
            """
            vmatpush.weight.mxu0 w0, m1
            vmatpop.bf16.acc.mxu0 m4, acc1
            vmatmul.mxu0 acc0, m5, w0
            """
        )
    )

    push = program.instructions[0]
    assert isinstance(push, VMATPUSH_WEIGHT_MXU0)
    assert push.vd == 0
    assert push.vs1 == 1
    pop = program.instructions[1]
    assert isinstance(pop, VMATPOP_BF16_ACC_MXU0)
    assert pop.vd == 4
    assert pop.vs2 == 1
    mul = program.instructions[2]
    assert isinstance(mul, VMATMUL_MXU0)
    assert mul.vd == 0
    assert mul.vs1 == 5
    assert mul.vs2 == 0


@pytest.mark.parametrize("mnemonic", ["beq", "bne", "blt", "bge", "bltu", "bgeu"])
@pytest.mark.parametrize("offset", [-2048, -1, 0, 1, 2047])
def test_branch_source_word_offsets_match_rtl_assembler(mnemonic: str, offset: int) -> None:
    source = f"{mnemonic} x1, x2, {offset}"
    insn = input_to_program(io.StringIO(source)).instructions[0]
    expected = getattr(_baremetal_assembler, mnemonic.upper())(1, 2, offset)
    assert insn.imm == offset * 2
    assert insn.to_bytecode() == expected
    assert lint_text(source) == []


@pytest.mark.parametrize("offset", [-524288, -1, 0, 1, 524287])
def test_jal_source_word_offsets_match_rtl_assembler(offset: int) -> None:
    source = f"jal x1, {offset}"
    insn = input_to_program(io.StringIO(source)).instructions[0]
    assert insn.imm == offset * 2
    assert insn.to_bytecode() == _baremetal_assembler.JAL(1, offset)
    assert lint_text(source) == []


@pytest.mark.parametrize("offset", [-2048, -1, 0, 1, 2047])
def test_jalr_uses_unscaled_word_offset_and_accepts_odd_values(offset: int) -> None:
    source = f"jalr x1, x2, {offset}"
    insn = input_to_program(io.StringIO(source)).instructions[0]
    assert insn.imm == offset
    assert insn.to_bytecode() == _baremetal_assembler.JALR(1, 2, offset)
    assert lint_text(source) == []


def test_control_flow_labels_account_for_li_expansion_in_word_units() -> None:
    program = input_to_program(io.StringIO("""
        start:
        li x3, 0x12345
        beq x1, x2, target
        nop
        target:
        jal x4, start
        nop
    """))
    assert program.instructions[2].to_bytecode() == _baremetal_assembler.BEQ(1, 2, 2)
    assert program.instructions[4].to_bytecode() == _baremetal_assembler.JAL(4, -4)


@pytest.mark.parametrize("source", [
    "beq x1, x2, -2049", "beq x1, x2, 2048",
    "jal x1, -524289", "jal x1, 524288",
])
def test_source_control_flow_offsets_outside_rtl_range_are_rejected(source: str) -> None:
    assert any("Instruction-word offset" in diagnostic.message for diagnostic in lint_text(source))
    with pytest.raises(ExceptionGroup):
        input_to_program(io.StringIO(source))


def test_instruction_constructors_retain_raw_encoded_immediate_convention() -> None:
    assert BEQ(rs1=1, rs2=2, imm=-4096).to_bytecode() == _baremetal_assembler.BEQ(1, 2, -2048)
    assert JAL(rd=1, imm=-1048576).to_bytecode() == _baremetal_assembler.JAL(1, -524288)
    assert BEQ.from_asm(["beq", "x1", "x2", "3"]).imm == 6
    assert JAL.from_asm(["jal", "x1", "3"]).imm == 6


@pytest.mark.parametrize("source, expected", [
    ("addi x3, x1, 17", _baremetal_assembler.ADDI(3, 1, 17)),
    ("lw x7, 16(x2)", _baremetal_assembler.LW(7, 2, 16)),
    ("jalr x3, x2, 5", _baremetal_assembler.JALR(3, 2, 5)),
    ("delay 7", _baremetal_assembler.DELAY_INSN(7)),
    ("seli e3, 127", _baremetal_assembler.SELI(3, 127)),
])
def test_i_type_encoding_keeps_destination_separate_from_immediate(source: str, expected: int) -> None:
    assert input_to_program(io.StringIO(source)).assemble() == [expected]

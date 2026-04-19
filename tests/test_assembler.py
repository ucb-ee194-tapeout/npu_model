import io

from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.util.converter import input_to_program


def test_li_expands_to_valid_addi_and_lui_addi_sequences() -> None:
    program = input_to_program(
        io.StringIO(
            """
            li x3, 5
            li x4, 0x12345
            """
        )
    )

    assert program.instructions[0].mnemonic == "addi"
    assert program.instructions[0].args == ScalarArgs(rd=3, rs1=0, imm=5)
    assert program.instructions[1].mnemonic == "lui"
    assert program.instructions[2].mnemonic == "addi"
    assert program.instructions[2].args.rs1 == 4


def test_dma_config_and_wait_parse_correctly() -> None:
    program = input_to_program(
        io.StringIO(
            """
            dma.config.ch0 x7
            dma.wait.ch1
            """
        )
    )

    assert program.instructions[0].args == DmaArgs(rs1=7, channel=0)
    assert program.instructions[1].args == DmaArgs(channel=1)


def test_offset_addressing_and_vi_immediates_parse_correctly() -> None:
    program = input_to_program(
        io.StringIO(
            """
            lw x1, 16(x2)
            sw x3, 20(x4)
            vload x5, 7(x6)
            vli.all x7, -3
            """
        )
    )

    assert program.instructions[0].args == ScalarArgs(rd=1, rs1=2, imm=16)
    assert program.instructions[1].args == ScalarArgs(rs2=3, rs1=4, imm=20)
    assert program.instructions[2].args == VectorArgs(vd=5, rs1=6, imm12=7)
    assert program.instructions[3].args == VectorArgs(vd=7, imm=-3)


def test_matrix_transfer_and_matmul_parse_correctly() -> None:
    program = input_to_program(
        io.StringIO(
            """
            vmatpush.weight.mxu0 x0, x1
            vmatpop.bf16.acc.mxu0 x4, x2
            vmatmul.mxu0 x0, x5, x6
            """
        )
    )

    assert program.instructions[0].args == MatrixArgs(vd=0, vs1=1)
    assert program.instructions[1].args == MatrixArgs(vd=4, vs1=2)
    assert program.instructions[2].args == MatrixArgs(vd=0, vs1=5, vs2=6)

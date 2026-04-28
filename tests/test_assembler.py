import io

from npu_model.configs.isa_definition import *  # noqa: F401, F403
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

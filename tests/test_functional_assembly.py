"""Parser and execution coverage for the channel-free ``.fs`` dialect."""

import pytest
import torch

from npu_model.functional import (
    FunctionalAssemblyError,
    FunctionalConfig,
    FunctionalInterpreter,
    load_functional_assembly,
    parse_functional_assembly,
)
from npu_model.isa_types import Accumulator, MatrixReg, ScalarReg, WeightBuffer


def test_virtual_registers_execute_scalar_operations_and_strip_comments():
    program = parse_functional_assembly(
        """
        # Named virtual registers are allocated on first use.
        addi x.left, x.zero, 9   # inline comments work too
        addi x.right, x.zero, 7
        add x.total, x.left, x.right
        """
    )

    result = FunctionalInterpreter().run(program)

    assert result.instructions_executed == 3
    assert result.named_registers(program)["x"] == {
        "left": 9,
        "zero": 0,
        "right": 7,
        "total": 16,
    }
    assert program.register_ids[ScalarReg] == {
        "left": 1,
        "zero": 2,
        "right": 3,
        "total": 4,
    }


def test_parser_assigns_disjoint_virtual_matrix_backing_slots():
    program = parse_functional_assembly(
        "vli.all m.first, 0\nvli.all m.second, 0"
    )

    assert program.register_ids[MatrixReg] == {"first": 0, "second": 2}


def test_named_vector_matrix_registers_execute_and_report_values():
    program = parse_functional_assembly(
        """
        vli.all m.seed, 0x3f80
        vadd.bf16 m.doubled, m.seed, m.seed
        """
    )

    result = FunctionalInterpreter().run(program)
    named = result.named_registers(program)["m"]
    seed = result.state.read_mrf_bf16_tile(program.register_ids[MatrixReg]["seed"])
    doubled = result.state.read_mrf_bf16_tile(program.register_ids[MatrixReg]["doubled"])

    assert program.register_ids[MatrixReg] == {"seed": 0, "doubled": 2}
    assert named["seed"].numel() == 32 * 32  # one 32x32-byte matrix slot
    assert torch.all(named["seed"].view(torch.bfloat16) == 1.0)
    assert torch.all(named["doubled"].view(torch.bfloat16) == 2.0)
    assert torch.all(seed == 1.0)
    assert torch.all(doubled == 2.0)


def test_named_weight_buffer_and_accumulator_flow_through_mxu():
    fp8_ones = torch.ones((32, 32), dtype=torch.float8_e4m3fn).view(torch.uint8)
    program = parse_functional_assembly(
        """
        addi x.vmem_base, x.zero, 0
        vload m.activation, 0(x.vmem_base)
        vload m.weights, 32(x.vmem_base)
        vmatpush.weight.mxu0 w.kernel, m.weights
        vmatmul.mxu0 acc.product, m.activation, w.kernel
        vmatpop.bf16.acc.mxu0 m.output, acc.product
        """
    )

    result = FunctionalInterpreter().run(program, vmem={0: fp8_ones, 4096: fp8_ones})
    named = result.named_registers(program)
    acc_id = program.register_ids[Accumulator]["product"]
    matrix_id = program.register_ids[MatrixReg]["output"]

    assert program.register_ids[WeightBuffer] == {"kernel": 0}
    assert named["w.mxu0"]["kernel"].numel() == 32 * 32
    assert torch.equal(named["w.mxu0"]["kernel"], fp8_ones.flatten())
    assert torch.all(named["acc.mxu0"]["product"] == 32.0)
    assert torch.all(named["acc.mxu1"]["product"] == 0.0)
    assert torch.all(result.state.read_mrf_bf16_tile(matrix_id) == 32.0)
    assert result.state.read_acc_bf16("mxu0", acc_id).shape == (32, 32)


def test_labels_drive_functional_control_flow_without_delay_slots():
    program = parse_functional_assembly(
        """
        addi x.i, x.zero, 0
        addi x.limit, x.zero, 3
        loop:
        addi x.i, x.i, 1
        blt x.i, x.limit, loop
        addi x.done, x.i, 0
        """
    )

    result = FunctionalInterpreter().run(program)

    assert program.labels == {"loop": 2}
    assert result.instructions_executed == 9
    assert result.named_registers(program)["x"]["i"] == 3
    assert result.named_registers(program)["x"]["done"] == 3


def test_csr_register_and_immediate_forms_read_and_update_state():
    program = parse_functional_assembly(
        """
        addi x.value, x.zero, 165
        csrrw x.old, 0xC10, x.value
        addi x.mask, x.zero, 15
        csrrs x.old_set, 0xC10, x.mask
        csrrc x.old_clear, 0xC10, x.mask
        csrrwi x.old_imm, 0xC11, 7
        csrrsi x.old_imm_set, 0xC11, 2
        csrrci x.old_imm_clear, 0xC11, 1
        """
    )

    result = FunctionalInterpreter().run(program)
    registers = result.named_registers(program)["x"]

    assert [registers[name] for name in (
        "old", "old_set", "old_clear", "old_imm", "old_imm_set", "old_imm_clear"
    )] == [0, 165, 175, 0, 7, 7]
    assert result.state.read_csrf(0xC10) == 160
    assert result.state.read_csrf(0xC11) == 6


def test_channel_free_dma_config_load_store_copy_bytes_atomically():
    payload = bytes(range(32))
    program = parse_functional_assembly(
        """
        addi x.base, x.zero, 0
        dma.config x.base
        addi x.vmem, x.zero, 0
        addi x.source, x.zero, 64
        addi x.destination, x.zero, 128
        addi x.length, x.zero, 32
        dma.load x.vmem, x.source, x.length
        dma.store x.destination, x.vmem, x.length
        """
    )

    result = FunctionalInterpreter().run(program, dram={64: payload})

    assert result.instructions_executed == 8
    assert result.state.vmem[0:32].tolist() == list(payload)
    assert result.state.read_dram(128, 32).tolist() == list(payload)


def test_channel_free_dma_respects_configured_dram_base():
    dram_base = 1 << 32
    payload = b"base-aware DMA"
    program = parse_functional_assembly(
        """
        addi x.base, x.zero, 1
        dma.config x.base
        addi x.vmem, x.zero, 0
        addi x.offset, x.zero, 64
        addi x.length, x.zero, 14
        dma.load x.vmem, x.offset, x.length
        """
    )
    config = FunctionalConfig(dram_size=dram_base + 4096, vmem_size=64)

    result = FunctionalInterpreter(config).run(
        program, dram={dram_base + 64: payload}
    )

    assert result.state.vmem[0:len(payload)].tolist() == list(payload)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("same:\naddi x.a, x.zero, 1\nsame:\naddi x.b, x.zero, 2", "duplicate label"),
        ("beq x.a, x.b, missing", "undefined label"),
        ("addi x.a, e.scale, 1", "expected virtual scalar register"),
        ("not_an_instruction x.a", "unsupported functional mnemonic"),
        ("dma.wait 0", "channel-specific or a synchronization operation"),
        ("dma.load.ch0 x.a, x.b, x.c", "channel-specific or a synchronization operation"),
        ("delay 4", "not part of the functional instruction subset"),
    ],
)
def test_parser_reports_invalid_or_unsupported_assembly(source, message):
    with pytest.raises(FunctionalAssemblyError, match=message):
        parse_functional_assembly(source)


def test_parser_checks_operand_count_and_preserves_source_line():
    with pytest.raises(FunctionalAssemblyError, match=r"line 2: 'addi' expects 3 operand"):
        parse_functional_assembly("# comment\naddi x.a, x.zero")


def test_memory_access_failures_include_the_instruction_line():
    program = parse_functional_assembly(
        """
        addi x.vmem, x.zero, 0
        addi x.offset, x.zero, 64
        addi x.length, x.zero, 1
        dma.load x.vmem, x.offset, x.length
        """
    )
    config = FunctionalConfig(dram_size=64, vmem_size=64)

    with pytest.raises(FunctionalAssemblyError, match=r"line 5: executing 'dma.load'.*DRAM read out of bounds"):
        FunctionalInterpreter(config).run(program)


def test_load_functional_assembly_reads_a_file(tmp_path):
    source = tmp_path / "kernel.fs"
    source.write_text("addi x.result, x.zero, 12\n", encoding="utf-8")

    program = load_functional_assembly(source)
    result = FunctionalInterpreter().run(program)

    assert result.named_registers(program)["x"]["result"] == 12

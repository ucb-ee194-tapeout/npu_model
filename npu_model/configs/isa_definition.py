import torch
from typing import Callable, Any

from npu_model.isa import instr, InstructionType
from npu_model.hardware.arch_state import ArchState


"""
Scalar operations
"""

PIPELINE_LATENCY = 2

# Mask for 64-bit unsigned comparison (RISC-V RV64)
_MASK64 = 0xFFFFFFFFFFFFFFFF


def define_isa(instr: Callable) -> None:
    @instr("delay", instruction_type=InstructionType.SCALAR)
    def delay(state: ArchState, args: Any) -> None:
        """
        Delay for a specified number of cycles.

        a `delay 0` is equivalent to a `nop` operation.
        """
        # delay is handled at IDU stage
        pass

    @instr("addi", instruction_type=InstructionType.SCALAR.I)
    def addi(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] + args.imm)

    @instr("slli", instruction_type=InstructionType.SCALAR.I)
    def slli(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] << args.imm)

    @instr("slti", instruction_type=InstructionType.SCALAR.I)
    def slti(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < args.imm else 0)

    @instr("sltiu", instruction_type=InstructionType.SCALAR.I)
    def sltiu(state: ArchState, args: Any) -> None:
        a = state.xrf[args.rs1] & _MASK64
        b = args.imm & _MASK64
        state.write_xrf(args.rd, 1 if a < b else 0)

    @instr("xori", instruction_type=InstructionType.SCALAR.I)
    def xori(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] ^ args.imm)

    @instr("srli", instruction_type=InstructionType.SCALAR.I)
    def srli(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] >> args.imm)

    @instr("srai", instruction_type=InstructionType.SCALAR.I)
    def srai(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] >> args.imm)

    @instr("ori", instruction_type=InstructionType.SCALAR.I)
    def ori(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] | args.imm)

    @instr("andi", instruction_type=InstructionType.SCALAR.I)
    def andi(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] & args.imm)

    @instr("add", instruction_type=InstructionType.SCALAR.R)
    def add(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] + state.xrf[args.rs2])

    @instr("sub", instruction_type=InstructionType.SCALAR.R)
    def sub(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] - state.xrf[args.rs2])

    @instr("sll", instruction_type=InstructionType.SCALAR.R)
    def sll(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] << state.xrf[args.rs2])

    @instr("slt", instruction_type=InstructionType.SCALAR.R)
    def slt(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < state.xrf[args.rs2] else 0)

    @instr("sltu", instruction_type=InstructionType.SCALAR.R)
    def sltu(state: ArchState, args: Any) -> None:
        a = state.xrf[args.rs1] & _MASK64
        b = state.xrf[args.rs2] & _MASK64
        state.write_xrf(args.rd, 1 if a < b else 0)

    @instr("xor", instruction_type=InstructionType.SCALAR.R)
    def xor(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] ^ state.xrf[args.rs2])

    @instr("srl", instruction_type=InstructionType.SCALAR.R)
    def srl(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])

    @instr("sra", instruction_type=InstructionType.SCALAR.R)
    def sra(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])

    @instr("or", instruction_type=InstructionType.SCALAR.R)
    def or_(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] | state.xrf[args.rs2])

    @instr("and", instruction_type=InstructionType.SCALAR.R)
    def and_(state: ArchState, args: Any) -> None:
        state.write_xrf(args.rd, state.xrf[args.rs1] & state.xrf[args.rs2])

    @instr("lui", instruction_type=InstructionType.SCALAR.U)
    def lui(state: ArchState, args: Any) -> None:
        """Load upper immediate: rd = imm << 12 (RISC-V LUI semantics)."""
        state.write_xrf(args.rd, (args.imm << 12) & _MASK64)

    @instr("jal", instruction_type=InstructionType.SCALAR.J)
    def jal(state: ArchState, args: Any) -> None:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay

    @instr("beq", instruction_type=InstructionType.SCALAR.B)
    def beq(state: ArchState, args: Any) -> None:
        if state.xrf[args.rs1] == state.xrf[args.rs2]:
            state.set_npc(
                state.pc + args.imm - PIPELINE_LATENCY
            )  # FIXME: this is a hack to compensate for the IF->EX delay

    @instr("bne", instruction_type=InstructionType.SCALAR.B)
    def bne(state: ArchState, args: Any) -> None:
        if state.xrf[args.rs1] != state.xrf[args.rs2]:
            state.set_npc(
                state.pc + args.imm - PIPELINE_LATENCY
            )  # FIXME: this is a hack to compensate for the IF->EX delay

    @instr("blt", instruction_type=InstructionType.SCALAR.B)
    def blt(state: ArchState, args: Any) -> None:
        if state.xrf[args.rs1] < state.xrf[args.rs2]:
            state.set_npc(
                state.pc + args.imm - PIPELINE_LATENCY
            )  # FIXME: this is a hack to compensate for the IF->EX delay

    @instr("bge", instruction_type=InstructionType.SCALAR.B)
    def bge(state: ArchState, args: Any) -> None:
        """Branch if rs1 >= rs2 (signed)."""
        if state.xrf[args.rs1] >= state.xrf[args.rs2]:
            state.set_npc(
                state.pc + args.imm - PIPELINE_LATENCY
            )  # FIXME: this is a hack to compensate for the IF->EX delay

    @instr("bltu", instruction_type=InstructionType.SCALAR.B)
    def bltu(state: ArchState, args: Any) -> None:
        """Branch if rs1 < rs2 (unsigned)."""
        a = state.xrf[args.rs1] & _MASK64
        b = state.xrf[args.rs2] & _MASK64
        if a < b:
            state.set_npc(
                state.pc + args.imm - PIPELINE_LATENCY
            )  # FIXME: this is a hack to compensate for the IF->EX delay

    @instr("bgeu", instruction_type=InstructionType.SCALAR.B)
    def bgeu(state: ArchState, args: Any) -> None:
        """Branch if rs1 >= rs2 (unsigned)."""
        a = state.xrf[args.rs1] & _MASK64
        b = state.xrf[args.rs2] & _MASK64
        if a >= b:
            state.set_npc(
                state.pc + args.imm - PIPELINE_LATENCY
            )  # FIXME: this is a hack to compensate for the IF->EX delay

    """
    Matrix operations
    """

    @instr("mv.mw", instruction_type=InstructionType.MATRIX)
    def mv_mw(state: ArchState, args: Any) -> None:
        """
        Vector/matrix move from matrix registers to weight buffer.
        """
        # TODO: check register dimensions
        state.write_wb_bf16("mxu0", args.rd, state.read_mrf_bf16(args.rs1))

    @instr("matmul.mxu0", instruction_type=InstructionType.MATRIX.MATRIX_SYSTOLIC)
    def matmul_mxu0(state: ArchState, args: Any) -> None:
        """
        Matrix multiplication using MXU0, the systolic array.
        Weights are read from MXU0's weight buffer.
        """
        activation_fp8 = state.read_mrf_fp8(args.rs1)
        weight_fp8 = state.read_wb_fp8("mxu0", args.rs2)

        activation_fp16 = activation_fp8.to(torch.float16)
        weight_fp16 = weight_fp8.to(torch.float16)

        product_fp16 = activation_fp16 @ weight_fp16

        acc_bf16 = state.read_mrf_bf16_tile(args.rd)
        acc_fp16 = acc_bf16.to(torch.float16)

        accumulation_fp16 = acc_fp16 + product_fp16

        output_bf16 = accumulation_fp16.to(torch.bfloat16)
        state.write_mrf_bf16_tile(args.rd, output_bf16)

    @instr("matmul.mxu1", instruction_type=InstructionType.MATRIX.MATRIX_IPT)
    def matmul_mxu1(state: ArchState, args: Any) -> None:
        activation_fp8 = state.read_mrf_fp8(args.rs1)
        weight_fp8 = state.read_wb_fp8("mxu1", args.rs2)

        activation_fp16 = activation_fp8.to(torch.float16)
        weight_fp16 = weight_fp8.to(torch.float16)

        product_fp16 = activation_fp16 @ weight_fp16

        acc_bf16 = state.read_mrf_bf16_tile(args.rd)
        acc_fp16 = acc_bf16.to(torch.float16)

        accumulation_fp16 = acc_fp16 + product_fp16

        output_bf16 = accumulation_fp16.to(torch.bfloat16)
        state.write_mrf_bf16_tile(args.rd, output_bf16)

    """
    Vector operations (bfloat16)
    """

    @instr("vadd", instruction_type=InstructionType.VECTOR)
    def vadd(state: ArchState, args: Any) -> None:
        a = state.read_mrf_bf16(args.vs1)
        b = state.read_mrf_bf16(args.vs2)
        state.write_mrf_bf16(args.vrd, (a + b).to(torch.bfloat16))

    @instr("vsub", instruction_type=InstructionType.VECTOR)
    def vsub(state: ArchState, args: Any) -> None:
        a = state.read_mrf_bf16(args.vs1)
        b = state.read_mrf_bf16(args.vs2)
        state.write_mrf_bf16(args.vrd, (a - b).to(torch.bfloat16))

    @instr("vmul", instruction_type=InstructionType.VECTOR)
    def vmul(state: ArchState, args: Any) -> None:
        a = state.read_mrf_bf16(args.vs1)
        b = state.read_mrf_bf16(args.vs2)
        result = (a * b).to(torch.bfloat16)
        state.write_mrf_bf16(args.vrd, result)

    @instr("vsqrt", instruction_type=InstructionType.VECTOR)
    def vsqrt(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.sqrt(x).to(torch.bfloat16))

    @instr("vrcp", instruction_type=InstructionType.VECTOR)
    def vrcp(state: ArchState, args: Any) -> None:
        """Elementwise reciprocal: 1 / x."""
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, (1.0 / x).to(torch.bfloat16))

    @instr("vexp", instruction_type=InstructionType.VECTOR)
    def vexp(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.exp(x).to(torch.bfloat16))

    @instr("vlog2", instruction_type=InstructionType.VECTOR)
    def vlog2(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.log2(x).to(torch.bfloat16))

    @instr("vexp2", instruction_type=InstructionType.VECTOR)
    def vexp2(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.exp2(x).to(torch.bfloat16))

    @instr("vsin", instruction_type=InstructionType.VECTOR)
    def vsin(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.sin(x).to(torch.bfloat16))

    @instr("vcos", instruction_type=InstructionType.VECTOR)
    def vcos(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.cos(x).to(torch.bfloat16))

    @instr("vtanh", instruction_type=InstructionType.VECTOR)
    def vtanh(state: ArchState, args: Any) -> None:
        x = state.read_mrf_bf16(args.vs1)
        state.write_mrf_bf16(args.vrd, torch.tanh(x).to(torch.bfloat16))

    @instr("vreduce.sum", instruction_type=InstructionType.VECTOR)
    def vreduce_sum(state: ArchState, args: Any) -> None:
        """Reduce sum over second-to-last (across columns) dimension. For (rows, cols) in, gives (1, cols) broadcast."""
        x = state.read_mrf_bf16(args.vs1)
        sum_val = torch.sum(x.float(), dim=0, keepdim=True)
        out = sum_val.expand_as(x).to(torch.bfloat16)
        state.write_mrf_bf16(args.vrd, out)

    @instr("vrot.reduce.sum", instruction_type=InstructionType.VECTOR)
    def vrot_reduce_sum(state: ArchState, args: Any) -> None:
        """Reduce sum over last (across rows) dimension. For (rows, cols) in, gives (rows, 1) broadcast."""
        # TODO: implementation cost?
        x = state.read_mrf_bf16(args.vs1)
        sum_val = torch.sum(x.float(), dim=-1, keepdim=True)
        out = sum_val.expand_as(x).to(torch.bfloat16)
        state.write_mrf_bf16(args.vrd, out)

    @instr("mv.mm", instruction_type=InstructionType.VECTOR)
    def mv_mm(state: ArchState, args: Any) -> None:
        """
        Vector/matrix move between matrix registers.
        """
        state.write_mrf_f32(args.rd, state.read_mrf_f32(args.rs1))

    """
    Transpose operations
    """

    @instr("vtrpose.h", instruction_type=InstructionType.VECTOR)
    def vtrpose_h(state: ArchState, args: Any) -> None:
        """Transpose upper half: block = x[:, 0:half], write (cols, rows) with first half rows = block.T. Use with vtrpose.l + vadd for full transpose."""
        # TODO: check correctness
        x = state.read_mrf_bf16(args.vs1)
        half = x.shape[0] // 2
        block = x[0:half, :]
        transposed = block.T.contiguous()
        out = torch.zeros_like(x)
        out[0:half, :] = transposed
        state.write_mrf_bf16(args.vrd, out)

    @instr("vtrpose.l", instruction_type=InstructionType.VECTOR)
    def vtrpose_l(state: ArchState, args: Any) -> None:
        """Transpose lower half: block = x[:, half:], write (cols, rows) with second half rows = block.T. Use with vtrpose.h + vadd for full transpose."""
        # TODO: check correctness
        x = state.read_mrf_bf16(args.vs1)
        half = x.shape[0] // 2
        block = x[half:, :]
        transposed = block.T.contiguous()
        out = torch.zeros_like(x)
        out[half:, :] = transposed
        state.write_mrf_bf16(args.vrd, out)

    """
    Memory operations
    """

    def _tensor_register_bytes(state: ArchState) -> int:
        return state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize

    def _vls_base_register(args: Any) -> int:
        if hasattr(args, "rs1"):
            return args.rs1
        if hasattr(args, "base"):
            return args.base
        if hasattr(args, "vs1"):
            return args.vs1
        raise KeyError("vload/vstore requires rs1, base, or vs1")

    def _vls_offset(args: Any) -> int:
        if hasattr(args, "imm"):
            return args.imm
        if hasattr(args, "imm12"):
            return args.imm12
        return getattr(args, "offset", 0)

    def _vls_address(state: ArchState, args: Any) -> int:
        return state.read_xrf(_vls_base_register(args)) + (_vls_offset(args) << 5)

    def _acc_dest_index(args: Any) -> int:
        if hasattr(args, "vd"):
            return args.vd
        raise KeyError("MXU local write requires a destination accumulator selector")

    def _acc_source_index(args: Any) -> int:
        if hasattr(args, "vs2"):
            return args.vs2
        if hasattr(args, "vs1"):
            return args.vs1
        if hasattr(args, "vd"):
            return args.vd
        raise KeyError("MXU local read requires an accumulator selector")

    @instr("vload", instruction_type=InstructionType.VECTOR)
    def vload(state: ArchState, args: Any) -> None:
        """
        Load one full tensor register from VMEM.
        """
        base = _vls_address(state, args)
        data = state.read_memory(base, _tensor_register_bytes(state)).to(torch.uint8)
        state.write_mrf_u8(args.vd, data)

    @instr("vstore", instruction_type=InstructionType.VECTOR)
    def vstore(state: ArchState, args: Any) -> None:
        """
        Store one full tensor register to VMEM.
        """
        base = _vls_address(state, args)
        register_index = getattr(
            args, "vd", getattr(args, "rs2", getattr(args, "vs1", None))
        )
        data = state.mrf[register_index].view(torch.uint8)
        state.write_memory(base, data)

    @instr("vmatpush.weight.mxu0", instruction_type=InstructionType.VECTOR)
    def vmatpush_weight_mxu0(state: ArchState, args: Any) -> None:
        state.write_wb_u8("mxu0", args.vd, state.mrf[args.vs1].view(torch.uint8))

    @instr("vmatpush.weight.mxu1", instruction_type=InstructionType.VECTOR)
    def vmatpush_weight_mxu1(state: ArchState, args: Any) -> None:
        state.write_wb_u8("mxu1", args.vd, state.mrf[args.vs1].view(torch.uint8))

    @instr("vmatpush.acc.fp8.mxu0", instruction_type=InstructionType.VECTOR)
    def vmatpush_acc_fp8_mxu0(state: ArchState, args: Any) -> None:
        state.write_acc_bf16(
            "mxu0",
            _acc_dest_index(args),
            state.read_mrf_fp8(args.vs1).to(torch.bfloat16),
        )

    @instr("vmatpush.acc.fp8.mxu1", instruction_type=InstructionType.VECTOR)
    def vmatpush_acc_fp8_mxu1(state: ArchState, args: Any) -> None:
        state.write_acc_bf16(
            "mxu1",
            _acc_dest_index(args),
            state.read_mrf_fp8(args.vs1).to(torch.bfloat16),
        )

    @instr("vmatpush.acc.bf16.mxu0", instruction_type=InstructionType.VECTOR)
    def vmatpush_acc_bf16_mxu0(state: ArchState, args: Any) -> None:
        state.write_acc_bf16(
            "mxu0", _acc_dest_index(args), state.read_mrf_bf16_tile(args.vs1)
        )

    @instr("vmatpush.acc.bf16.mxu1", instruction_type=InstructionType.VECTOR)
    def vmatpush_acc_bf16_mxu1(state: ArchState, args: Any) -> None:
        state.write_acc_bf16(
            "mxu1", _acc_dest_index(args), state.read_mrf_bf16_tile(args.vs1)
        )

    @instr("vmatpop.fp8.acc.mxu0", instruction_type=InstructionType.VECTOR)
    def vmatpop_fp8_acc_mxu0(state: ArchState, args: Any) -> None:
        quantized = state.read_acc_bf16("mxu0", _acc_source_index(args)).to(
            torch.float8_e4m3fn
        )
        state.write_mrf_u8(args.vd, quantized.view(torch.uint8))

    @instr("vmatpop.fp8.acc.mxu1", instruction_type=InstructionType.VECTOR)
    def vmatpop_fp8_acc_mxu1(state: ArchState, args: Any) -> None:
        quantized = state.read_acc_bf16("mxu1", _acc_source_index(args)).to(
            torch.float8_e4m3fn
        )
        state.write_mrf_u8(args.vd, quantized.view(torch.uint8))

    @instr("vmatpop.bf16.acc.mxu0", instruction_type=InstructionType.VECTOR)
    def vmatpop_bf16_acc_mxu0(state: ArchState, args: Any) -> None:
        state.write_mrf_bf16_tile(
            args.vd, state.read_acc_bf16("mxu0", _acc_source_index(args))
        )

    @instr("vmatpop.bf16.acc.mxu1", instruction_type=InstructionType.VECTOR)
    def vmatpop_bf16_acc_mxu1(state: ArchState, args: Any) -> None:
        state.write_mrf_bf16_tile(
            args.vd, state.read_acc_bf16("mxu1", _acc_source_index(args))
        )

    @instr("vmatpop.mxu0", instruction_type=InstructionType.VECTOR)
    def vmatpop_mxu0(state: ArchState, args: Any) -> None:
        state.write_mrf_bf16_tile(
            args.vd, state.read_acc_bf16("mxu0", _acc_source_index(args))
        )

    @instr("vmatpop.mxu1", instruction_type=InstructionType.VECTOR)
    def vmatpop_mxu1(state: ArchState, args: Any) -> None:
        state.write_mrf_bf16_tile(
            args.vd, state.read_acc_bf16("mxu1", _acc_source_index(args))
        )

    def _vmatmul(state: ArchState, unit: str, args: Any, accumulate: bool) -> None:
        activation_fp16 = state.read_mrf_fp8(args.vs1).to(torch.float16)
        weight_fp16 = state.read_wb_fp8(unit, args.vs2).to(torch.float16)
        result_fp16 = activation_fp16 @ weight_fp16
        if accumulate:
            result_fp16 = result_fp16 + state.read_acc_bf16(
                unit, _acc_dest_index(args)
            ).to(torch.float16)
        state.write_acc_bf16(
            unit, _acc_dest_index(args), result_fp16.to(torch.bfloat16)
        )

    @instr("vmatmul.mxu0", instruction_type=InstructionType.MATRIX.MATRIX_SYSTOLIC)
    def vmatmul_mxu0(state: ArchState, args: Any) -> None:
        _vmatmul(state, "mxu0", args, accumulate=False)

    @instr("vmatmul.mxu1", instruction_type=InstructionType.MATRIX.MATRIX_IPT)
    def vmatmul_mxu1(state: ArchState, args: Any) -> None:
        _vmatmul(state, "mxu1", args, accumulate=False)

    @instr("vmatmul.acc.mxu0", instruction_type=InstructionType.MATRIX.MATRIX_SYSTOLIC)
    def vmatmul_acc_mxu0(state: ArchState, args: Any) -> None:
        _vmatmul(state, "mxu0", args, accumulate=True)

    @instr("vmatmul.acc.mxu1", instruction_type=InstructionType.MATRIX.MATRIX_IPT)
    def vmatmul_acc_mxu1(state: ArchState, args: Any) -> None:
        _vmatmul(state, "mxu1", args, accumulate=True)

    @instr("dma.load", instruction_type=InstructionType.DMA)
    def dma_load(state: ArchState, args: Any) -> None:
        """
        DMA load from memory to matrix registers.
        """
        base = args.base
        size = args.size
        data = state.read_memory(base, size)
        # zero pad the data to the size of the MRF
        if data.numel() < _tensor_register_bytes(state):
            data = torch.nn.functional.pad(
                data,
                (
                    0,
                    _tensor_register_bytes(state) - data.numel(),
                ),
            )
        state.write_mrf_u8(args.rd, data)

    @instr("dma.load.mxu0", instruction_type=InstructionType.DMA)
    def dma_load_mxu0(state: ArchState, args: Any) -> None:
        """
        DMA load from memory to weight buffer at MXU0.
        """
        base = args.base
        size = args.size
        data = state.read_memory(base, size).to(torch.uint8)
        # zero pad the data to the size of the WB
        if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
            data = torch.nn.functional.pad(
                data,
                (
                    0,
                    state.cfg.wb_width // torch.uint8.itemsize - data.numel(),
                ),
            )
        state.write_wb_u8("mxu0", args.rd, data)

    @instr("dma.load.mxu1", instruction_type=InstructionType.DMA)
    def dma_load_mxu1(state: ArchState, args: Any) -> None:
        """
        DMA load from memory to weight buffer at MXU1.
        """
        base = args.base
        size = args.size
        data = state.read_memory(base, size).to(torch.uint8)
        # zero pad the data to the size of the WB
        if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
            data = torch.nn.functional.pad(
                data,
                (
                    0,
                    state.cfg.wb_width // torch.uint8.itemsize - data.numel(),
                ),
            )
        state.write_wb_u8("mxu1", args.rd, data)

    @instr("dma.store", instruction_type=InstructionType.DMA)
    def dma_store(state: ArchState, args: Any) -> None:
        """
        DMA store from matrix registers to memory.
        """
        base = args.base
        size = args.size
        data = state.mrf[args.rs1].view(torch.uint8)
        state.write_memory(base, data[:size])

    @instr("dma.wait", instruction_type=InstructionType.BARRIER)
    def dma_wait(state: ArchState, args: Any) -> None:
        """
        Wait for target DMA operations to complete.
        """
        pass

import torch

from npu_model.isa import instr, InstructionType
from npu_model.hardware.arch_state import ArchState


"""
Scalar operations
"""

PIPELINE_LATENCY = 2

# Mask for 64-bit unsigned comparison (RISC-V RV64)
_MASK64 = 0xFFFFFFFFFFFFFFFF


@instr("delay", instruction_type=InstructionType.SCALAR)
def delay(state: ArchState, args: dict[str, int]) -> None:
    """
    Delay for a specified number of cycles.

    a `delay 0` is equivalent to a `nop` operation.
    """
    # delay is handled at IDU stage
    pass


@instr("addi", instruction_type=InstructionType.SCALAR)
def addi(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] + args["imm"])


@instr("slli", instruction_type=InstructionType.SCALAR)
def slli(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] << args["imm"])


@instr("slti", instruction_type=InstructionType.SCALAR)
def slti(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], 1 if state.xrf[args["rs1"]] < args["imm"] else 0)


@instr("sltiu", instruction_type=InstructionType.SCALAR)
def sltiu(state: ArchState, args: dict[str, int]) -> None:
    a = state.xrf[args["rs1"]] & _MASK64
    b = args["imm"] & _MASK64
    state.write_xrf(args["rd"], 1 if a < b else 0)


@instr("xori", instruction_type=InstructionType.SCALAR)
def xori(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] ^ args["imm"])


@instr("srli", instruction_type=InstructionType.SCALAR)
def srli(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> args["imm"])


@instr("srai", instruction_type=InstructionType.SCALAR)
def srai(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> args["imm"])


@instr("ori", instruction_type=InstructionType.SCALAR)
def ori(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] | args["imm"])


@instr("andi", instruction_type=InstructionType.SCALAR)
def andi(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] & args["imm"])


@instr("add", instruction_type=InstructionType.SCALAR)
def add(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] + state.xrf[args["rs2"]])


@instr("sub", instruction_type=InstructionType.SCALAR)
def sub(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] - state.xrf[args["rs2"]])


@instr("sll", instruction_type=InstructionType.SCALAR)
def sll(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] << state.xrf[args["rs2"]])


@instr("slt", instruction_type=InstructionType.SCALAR)
def slt(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(
        args["rd"], 1 if state.xrf[args["rs1"]] < state.xrf[args["rs2"]] else 0
    )


@instr("sltu", instruction_type=InstructionType.SCALAR)
def sltu(state: ArchState, args: dict[str, int]) -> None:
    a = state.xrf[args["rs1"]] & _MASK64
    b = state.xrf[args["rs2"]] & _MASK64
    state.write_xrf(args["rd"], 1 if a < b else 0)


@instr("xor", instruction_type=InstructionType.SCALAR)
def xor(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] ^ state.xrf[args["rs2"]])


@instr("srl", instruction_type=InstructionType.SCALAR)
def srl(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> state.xrf[args["rs2"]])


@instr("sra", instruction_type=InstructionType.SCALAR)
def sra(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> state.xrf[args["rs2"]])


@instr("or", instruction_type=InstructionType.SCALAR)
def or_(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] | state.xrf[args["rs2"]])


@instr("and", instruction_type=InstructionType.SCALAR)
def and_(state: ArchState, args: dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] & state.xrf[args["rs2"]])


@instr("lui", instruction_type=InstructionType.SCALAR)
def lui(state: ArchState, args: dict[str, int]) -> None:
    """Load upper immediate: rd = imm << 12 (RISC-V LUI semantics)."""
    state.write_xrf(args["rd"], (args["imm"] << 12) & _MASK64)


@instr("jal", instruction_type=InstructionType.SCALAR)
def jal(state: ArchState, args: dict[str, int]) -> None:
    state.set_npc(
        state.pc + args["imm"] - PIPELINE_LATENCY
    )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("beq", instruction_type=InstructionType.SCALAR)
def beq(state: ArchState, args: dict[str, int]) -> None:
    if state.xrf[args["rs1"]] == state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bne", instruction_type=InstructionType.SCALAR)
def bne(state: ArchState, args: dict[str, int]) -> None:
    if state.xrf[args["rs1"]] != state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("blt", instruction_type=InstructionType.SCALAR)
def blt(state: ArchState, args: dict[str, int]) -> None:
    if state.xrf[args["rs1"]] < state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bge", instruction_type=InstructionType.SCALAR)
def bge(state: ArchState, args: dict[str, int]) -> None:
    """Branch if rs1 >= rs2 (signed)."""
    if state.xrf[args["rs1"]] >= state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bltu", instruction_type=InstructionType.SCALAR)
def bltu(state: ArchState, args: dict[str, int]) -> None:
    """Branch if rs1 < rs2 (unsigned)."""
    a = state.xrf[args["rs1"]] & _MASK64
    b = state.xrf[args["rs2"]] & _MASK64
    if a < b:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bgeu", instruction_type=InstructionType.SCALAR)
def bgeu(state: ArchState, args: dict[str, int]) -> None:
    """Branch if rs1 >= rs2 (unsigned)."""
    a = state.xrf[args["rs1"]] & _MASK64
    b = state.xrf[args["rs2"]] & _MASK64
    if a >= b:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


"""
Matrix operations
"""


@instr("mv.mw.mxu0", instruction_type=InstructionType.MATRIX)
def mv_mw_mxu0(state: ArchState, args: dict[str, int]) -> None:
    """
    Vector/matrix move from matrix registers to weight buffer.
    Takes first 512 elements from MRF (fp8) and writes to WB.
    Zero-pads automatically since MRF read always returns full register.
    """
    mrf_data = state.read_mrf_fp8(args["rs1"])  # (64, 32) = 2048 elements

    # Take first 512 elements (WB capacity)
    wb_capacity = state.cfg.wb_width // torch.float8_e4m3fn.itemsize  # 512
    wb_data_flat = mrf_data.flatten()[:wb_capacity]  # First 512 elements

    # Reshape to (32, 16) for WB
    num_rows = state.cfg.mrf_width // torch.float8_e4m3fn.itemsize  # 32
    num_cols = wb_capacity // num_rows  # 16
    wb_data_reshaped = wb_data_flat.reshape(num_rows, num_cols)

    state.write_wb_fp8("mxu0", args["rd"], wb_data_reshaped)


@instr("mv.mw.mxu1", instruction_type=InstructionType.MATRIX)
def mv_mw_mxu1(state: ArchState, args: dict[str, int]) -> None:
    """
    Vector/matrix move from matrix registers to weight buffer.
    Takes first 512 elements from MRF (fp8) and writes to WB.
    Zero-pads automatically since MRF read always returns full register.
    """
    mrf_data = state.read_mrf_fp8(args["rs1"])  # (64, 32) = 2048 elements

    # Take first 512 elements (WB capacity)
    wb_capacity = state.cfg.wb_width // torch.float8_e4m3fn.itemsize  # 512
    wb_data_flat = mrf_data.flatten()[:wb_capacity]  # First 512 elements

    # Reshape to (32, 16) for WB
    num_rows = state.cfg.mrf_width // torch.float8_e4m3fn.itemsize  # 32
    num_cols = wb_capacity // num_rows  # 16
    wb_data_reshaped = wb_data_flat.reshape(num_rows, num_cols)

    state.write_wb_fp8("mxu1", args["rd"], wb_data_reshaped)


@instr("matmul.mxu0", instruction_type=InstructionType.MATRIX_SYSTOLIC)
def matmul_mxu0(state: ArchState, args: dict[str, int]) -> None:
    """
    Matrix multiplication using MXU0, the systolic array.
    Weights are read from MXU0's weight buffer.
    """
    activation_fp8 = state.read_mrf_fp8(args["rs1"])
    weight_fp8 = state.read_wb_fp8("mxu0", args["rs2"])

    activation_fp16 = activation_fp8.to(torch.float16)
    weight_fp16 = weight_fp8.to(torch.float16)

    product_fp16 = activation_fp16 @ weight_fp16

    acc_bf16 = state.read_mrf_bf16(args["rd"])
    acc_fp16 = acc_bf16.to(torch.float16)

    accumulation_fp16 = acc_fp16 + product_fp16

    output_bf16 = accumulation_fp16.to(torch.bfloat16)
    state.write_mrf_bf16(args["rd"], output_bf16)


@instr("matmul.mxu1", instruction_type=InstructionType.MATRIX_INNER)
def matmul_mxu1(state: ArchState, args: dict[str, int]) -> None:
    activation_fp8 = state.read_mrf_fp8(args["rs1"])
    weight_fp8 = state.read_wb_fp8("mxu1", args["rs2"])

    activation_fp16 = activation_fp8.to(torch.float16)
    weight_fp16 = weight_fp8.to(torch.float16)

    product_fp16 = activation_fp16 @ weight_fp16

    acc_bf16 = state.read_mrf_bf16(args["rd"])
    acc_fp16 = acc_bf16.to(torch.float16)

    accumulation_fp16 = acc_fp16 + product_fp16

    output_bf16 = accumulation_fp16.to(torch.bfloat16)
    state.write_mrf_bf16(args["rd"], output_bf16)


"""
Vector operations (bfloat16)
"""


@instr("vadd", instruction_type=InstructionType.VECTOR)
def vadd(state: ArchState, args: dict[str, int]) -> None:
    a = state.read_mrf_bf16(args["vs1"])
    b = state.read_mrf_bf16(args["vs2"])
    state.write_mrf_bf16(args["vrd"], (a + b).to(torch.bfloat16))


@instr("vsub", instruction_type=InstructionType.VECTOR)
def vsub(state: ArchState, args: dict[str, int]) -> None:
    a = state.read_mrf_bf16(args["vs1"])
    b = state.read_mrf_bf16(args["vs2"])
    state.write_mrf_bf16(args["vrd"], (a - b).to(torch.bfloat16))


@instr("vmul", instruction_type=InstructionType.VECTOR)
def vmul(state: ArchState, args: dict[str, int]) -> None:
    a = state.read_mrf_bf16(args["vs1"])
    b = state.read_mrf_bf16(args["vs2"])
    result = (a * b).to(torch.bfloat16)
    state.write_mrf_bf16(args["vrd"], result)


@instr("vsqrt", instruction_type=InstructionType.VECTOR)
def vsqrt(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.sqrt(x).to(torch.bfloat16))


@instr("vrcp", instruction_type=InstructionType.VECTOR)
def vrcp(state: ArchState, args: dict[str, int]) -> None:
    """Elementwise reciprocal: 1 / x."""
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], (1.0 / x).to(torch.bfloat16))


@instr("vexp", instruction_type=InstructionType.VECTOR)
def vexp(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.exp(x).to(torch.bfloat16))


@instr("vlog2", instruction_type=InstructionType.VECTOR)
def vlog2(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.log2(x).to(torch.bfloat16))


@instr("vexp2", instruction_type=InstructionType.VECTOR)
def vexp2(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.exp2(x).to(torch.bfloat16))


@instr("vsin", instruction_type=InstructionType.VECTOR)
def vsin(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.sin(x).to(torch.bfloat16))


@instr("vcos", instruction_type=InstructionType.VECTOR)
def vcos(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.cos(x).to(torch.bfloat16))


@instr("vtanh", instruction_type=InstructionType.VECTOR)
def vtanh(state: ArchState, args: dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.tanh(x).to(torch.bfloat16))


@instr("vzero", instruction_type=InstructionType.VECTOR)
def vzero(state: ArchState, args: dict[str, int]) -> None:
    """Zero out a register (fill with zeros)."""
    zeros = torch.zeros((state.cfg.mrf_depth, state.cfg.mrf_width // torch.bfloat16.itemsize), dtype=torch.bfloat16)
    state.write_mrf_bf16(args["vrd"], zeros)


@instr("vmax", instruction_type=InstructionType.VECTOR)
def vmax(state: ArchState, args: dict[str, int]) -> None:
    """Element-wise max between two registers."""
    x = state.read_mrf_bf16(args["vs1"])  # (64, 16)
    y = state.read_mrf_bf16(args["vs2"])  # (64, 16)
    result = torch.maximum(x, y)
    state.write_mrf_bf16(args["vrd"], result)


@instr("vrowmax", instruction_type=InstructionType.VECTOR)
def vrowmax(state: ArchState, args: dict[str, int]) -> None:
    """Row-wise max: compute max across each row, broadcast to all columns."""
    x = state.read_mrf_bf16(args["vs1"])  # (64, 16)
    row_max = x.max(dim=1, keepdim=True)  # (64, 1)
    # Broadcast to (64, 16)
    result = row_max.expand(64, 16).contiguous()
    state.write_mrf_bf16(args["vrd"], result)


@instr("vcast.down", instruction_type=InstructionType.VECTOR)
def vcast_down(state: ArchState, args: dict[str, int]) -> None:
    """Convert bfloat16 to float8: vrd = fp8(vs1)."""
    x_bf16 = state.read_mrf_bf16(args["vs1"])  # (64, 16) = 1024 elements
    x_fp8 = x_bf16.to(torch.float8_e4m3fn)     # 1024 fp8 elements

    # Pad with zeros to fill fp8 register (needs 2048 elements)
    fp8_capacity = (state.cfg.mrf_depth * state.cfg.mrf_width) // torch.float8_e4m3fn.itemsize
    x_fp8_padded = torch.zeros(fp8_capacity, dtype=torch.float8_e4m3fn)
    x_fp8_padded[:x_fp8.numel()] = x_fp8.flatten()

    # Reshape to (64, 32)
    x_fp8_reshaped = x_fp8_padded.reshape(state.cfg.mrf_depth, state.cfg.mrf_width // torch.float8_e4m3fn.itemsize)
    state.write_mrf_fp8(args["vrd"], x_fp8_reshaped)


@instr("vcast.up", instruction_type=InstructionType.VECTOR)
def vcast_up(state: ArchState, args: dict[str, int]) -> None:
    """Convert float8 to bfloat16: vrd = bf16(vs1)."""
    x_fp8 = state.read_mrf_fp8(args["vs1"])  # (64, 32) = 2048 elements

    # Only convert as many elements as fit in bf16 register (1024 elements)
    bf16_capacity = (state.cfg.mrf_depth * state.cfg.mrf_width) // torch.bfloat16.itemsize
    x_fp8_trimmed = x_fp8.flatten()[:bf16_capacity]  # Take first 1024 elements
    x_bf16 = x_fp8_trimmed.to(torch.bfloat16)  # 1024 bf16 elements

    # Reshape to (64, 16) to match bf16 register shape
    x_bf16_reshaped = x_bf16.reshape(state.cfg.mrf_depth, state.cfg.mrf_width // torch.bfloat16.itemsize)
    state.write_mrf_bf16(args["vrd"], x_bf16_reshaped)


@instr("vreduce.sum", instruction_type=InstructionType.VECTOR)
def vreduce_sum(state: ArchState, args: dict[str, int]) -> None:
    """Reduce sum over second-to-last (across columns) dimension. For (rows, cols) in, gives (1, cols) broadcast."""
    x = state.read_mrf_bf16(args["vs1"])
    sum_val = torch.sum(x.float(), dim=0, keepdim=True)
    out = sum_val.expand_as(x).to(torch.bfloat16)
    state.write_mrf_bf16(args["vrd"], out)


@instr("vrot.reduce.sum", instruction_type=InstructionType.VECTOR)
def vrot_reduce_sum(state: ArchState, args: dict[str, int]) -> None:
    """Reduce sum over last (across rows) dimension. For (rows, cols) in, gives (rows, 1) broadcast."""
    # TODO: implementation cost?
    x = state.read_mrf_bf16(args["vs1"])
    sum_val = torch.sum(x.float(), dim=-1, keepdim=True)
    out = sum_val.expand_as(x).to(torch.bfloat16)
    state.write_mrf_bf16(args["vrd"], out)


@instr("mv.mm", instruction_type=InstructionType.VECTOR)
def mv_mm(state: ArchState, args: dict[str, int]) -> None:
    """
    Vector/matrix move between matrix registers.
    """
    state.write_mrf_f32(args["rd"], state.read_mrf_f32(args["rs1"]))


"""
Transpose operations
"""


@instr("vtrpose", instruction_type=InstructionType.VECTOR)
def vtrpose(state: ArchState, args: dict[str, int]) -> None:
    """
    Transpose a 16×16 block from source register.

    Extracts rows [imm*16:(imm+1)*16, :] from vs1 (a 16×16 block),
    transposes it to 16×16, and stores in vrd's first 16 rows.
    Rows 16-63 of vrd are zeroed.

    Args:
        vrd: destination register
        vs1: source register (64×16 bf16)
        imm: block index (0-3) selecting which 16-row block to transpose

    Example: To transpose entire K (64×16) for attention:
        vtrpose vrd=1, vs1=0, imm=0  # K[0:16,:].T  → MRF[1][0:16,:]
        vtrpose vrd=2, vs1=0, imm=1  # K[16:32,:].T → MRF[2][0:16,:]
        vtrpose vrd=3, vs1=0, imm=2  # K[32:48,:].T → MRF[3][0:16,:]
        vtrpose vrd=4, vs1=0, imm=3  # K[48:64,:].T → MRF[4][0:16,:]
    """
    x = state.read_mrf_bf16(args["vs1"])
    block_idx = args.get("imm", 0)

    # Validate block index
    assert 0 <= block_idx <= 3, f"Block index must be 0-3, got {block_idx}"

    # Extract 16×16 block
    start_row = block_idx * 16
    end_row = start_row + 16
    block = x[start_row:end_row, :]

    # Transpose
    transposed = block.T.contiguous()  # (16, 16)

    # Pad to (64, 16) register size
    out = torch.zeros((64, 16), dtype=torch.bfloat16)
    out[0:16, :] = transposed

    state.write_mrf_bf16(args["vrd"], out)


@instr("vtrpose.h", instruction_type=InstructionType.VECTOR)
def vtrpose_h(state: ArchState, args: dict[str, int]) -> None:
    """Transpose upper half: block = x[:, 0:half], write (cols, rows) with first half rows = block.T. Use with vtrpose.l + vadd for full transpose."""
    # TODO: check correctness
    x = state.read_mrf_bf16(args["vs1"])
    half = x.shape[0] // 2
    block = x[0:half, :]
    transposed = block.T.contiguous()
    out = torch.zeros_like(x)
    out[0:half, :] = transposed
    state.write_mrf_bf16(args["vrd"], out)


@instr("vtrpose.l", instruction_type=InstructionType.VECTOR)
def vtrpose_l(state: ArchState, args: dict[str, int]) -> None:
    """Transpose lower half: block = x[:, half:], write (cols, rows) with second half rows = block.T. Use with vtrpose.h + vadd for full transpose."""
    # TODO: check correctness
    x = state.read_mrf_bf16(args["vs1"])
    half = x.shape[0] // 2
    block = x[half:, :]
    transposed = block.T.contiguous()
    out = torch.zeros_like(x)
    out[half:, :] = transposed
    state.write_mrf_bf16(args["vrd"], out)


"""
Memory operations
"""


@instr("dma.load", instruction_type=InstructionType.DMA)
def dma_load(state: ArchState, args: dict[str, int]) -> None:
    """
    DMA load from memory to matrix registers.
    """
    base = args["base"]
    size = args["size"]
    data = state.read_memory(base, size)
    # zero pad the data to the size of the MRF
    if data.numel() < state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize:
        data = torch.nn.functional.pad(
            data,
            (
                0,
                state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize
                - data.numel(),
            ),
        )
    state.write_mrf_u8(args["rd"], data)


@instr("dma.load.mxu0", instruction_type=InstructionType.DMA)
def dma_load_mxu0(state: ArchState, args: dict[str, int]) -> None:
    """
    DMA load from memory to weight buffer at MXU0.
    """
    base = args["base"]
    size = args["size"]
    data = state.read_memory(base, size).to(torch.uint8)
    # zero pad the data to the size of the WB
    if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
        data = torch.nn.functional.pad(
            data,
            (
                0,
                state.cfg.wb_width // torch.uint8.itemsize
                - data.numel(),
            ),
        )
    state.write_wb_u8("mxu0", args["rd"], data)


@instr("dma.load.mxu1", instruction_type=InstructionType.DMA)
def dma_load_mxu1(state: ArchState, args: dict[str, int]) -> None:
    """
    DMA load from memory to weight buffer at MXU1.
    """
    base = args["base"]
    size = args["size"]
    data = state.read_memory(base, size).to(torch.uint8)
    # zero pad the data to the size of the WB
    if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
        data = torch.nn.functional.pad(
            data,
            (
                0,
                state.cfg.wb_width // torch.uint8.itemsize
                - data.numel(),
            ),
        )
    state.write_wb_u8("mxu1", args["rd"], data)


@instr("dma.store", instruction_type=InstructionType.DMA)
def dma_store(state: ArchState, args: dict[str, int]) -> None:
    """
    DMA store from matrix registers to memory.
    """
    base = args["base"]
    size = args["size"]
    data = state.mrf[args["rs1"]].view(torch.uint8)
    state.write_memory(base, data[:size])


@instr("dma.wait", instruction_type=InstructionType.BARRIER)
def dma_wait(state: ArchState, args: dict[str, int]) -> None:
    """
    Wait for target DMA operations to complete.
    """
    pass

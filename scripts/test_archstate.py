import torch
from model_npu.hardware.arch_state import ArchState
from model_npu.hardware.config import ArchStateConfig

if __name__ == "__main__":
    cfg = ArchStateConfig(
        mrf_depth=64,  # each instruction is 64 cycles
        mrf_width=1 * 32 * 2,  # each cycle, we read out 64 B of activation row
        wb_width=16 * 32 * 2,  # each cycle, we read out 1024 B of weight
        num_x_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        memory_size=1048576,
    )

    state = ArchState(cfg)

    state.write_mrf_f32(0, torch.ones(64, 16))
    print(state.read_mrf_f32(0))

    state.write_mrf_bf16(0, torch.ones(64, 32))
    print(state.read_mrf_bf16(0))

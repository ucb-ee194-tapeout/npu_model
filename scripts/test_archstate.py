import torch
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.config import ArchStateConfig

if __name__ == "__main__":
    cfg = ArchStateConfig(
        mrf_depth=32,
        mrf_width=32,
        wb_width=32 * 32,
        num_x_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        memory_size=1048576,
    )

    state = ArchState(cfg)

    state.write_mrf_f32(0, torch.ones(32, 8))
    print(state.read_mrf_f32(0))

    state.write_mrf_bf16(0, torch.ones(32, 16, dtype=torch.bfloat16))
    print(state.read_mrf_bf16(0))

    state.write_mrf_bf16_tile(2, torch.ones(32, 32, dtype=torch.bfloat16))
    print(state.read_mrf_bf16_tile(2))

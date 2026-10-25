# perf_dma_load_1k.S — DMA calibration: one 1 KiB load
# Banks the mcycles delta around the DMA region into CSR_DBG1 so the
# npu-model memory backend can be fitted (npu-model/scripts/calibrate_dma_backend.py).
# @TIMEOUT 100000
# @DRAM_BASE 0x90000000
# @PERF_REPORT

    li    x5, 0x00000000
    dma.config x5, ch0
    li    x12, 32
    li    x13, 1024
    li    x14, 4096
    csrrs x20, 0xC00, x0        # x20 = cycle count at start
    addi  x6, x0, 0
    li    x10, 0x90000000
    dma.load  x6, x10, x13, ch0
    dma.wait  ch0
    csrrs x21, 0xC00, x0        # x21 = cycle count at end
    sub   x22, x21, x20
    csrrw x0,  0x801, x22       # stash in CSR_DBG1
    addi  x30, x0, 1
    csrrw x0, 0x800, x30
    ecall

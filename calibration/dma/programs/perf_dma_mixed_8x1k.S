# perf_dma_mixed_8x1k.S — DMA calibration: alternating 1 KiB loads and stores on all channels
# Banks the mcycles delta around the DMA region into CSR_DBG1 so the
# npu-model memory backend can be fitted (npu-model/scripts/calibrate_dma_backend.py).
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PERF_REPORT

    li    x5, 0x00000000
    dma.config x5, 0
    li    x12, 32
    li    x13, 1024
    li    x14, 4096
    csrrs x20, 0xC00, x0        # x20 = cycle count at start
    li    x6, 0
    li    x10, 0x90000000
    dma.load  x6, x10, x13, 0
    li    x6, 256
    li    x10, 0x90004400
    dma.store x10, x6, x13, 1
    li    x6, 512
    li    x10, 0x90000800
    dma.load  x6, x10, x13, 2
    li    x6, 768
    li    x10, 0x90004C00
    dma.store x10, x6, x13, 3
    li    x6, 1024
    li    x10, 0x90001000
    dma.load  x6, x10, x13, 4
    li    x6, 1280
    li    x10, 0x90005400
    dma.store x10, x6, x13, 5
    li    x6, 1536
    li    x10, 0x90001800
    dma.load  x6, x10, x13, 6
    li    x6, 1792
    li    x10, 0x90005C00
    dma.store x10, x6, x13, 7
    dma.wait  0
    dma.wait  1
    dma.wait  2
    dma.wait  3
    dma.wait  4
    dma.wait  5
    dma.wait  6
    dma.wait  7

    csrrs x21, 0xC00, x0        # x21 = cycle count at end
    sub   x22, x21, x20
    csrrw x0,  0xC11, x22       # stash in CSR_DBG1
    addi  x30, x0, 1
    csrrw x0, 0xC10, x30
    ecall

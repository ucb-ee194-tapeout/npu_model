# perf_dma_store_8x1k.S — DMA calibration: eight 1 KiB stores on all channels, then eight waits
# Banks the mcycles delta around the DMA region into CSR_DBG1 so the
# npu-model memory backend can be fitted (npu-model/scripts/calibrate_dma_backend.py).
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PERF_REPORT

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 32
    LI    x13, 1024
    LI    x14, 4096
    CSRRS x20, 0xC00, x0        # x20 = cycle count at start
    LI    x6, 0
    LI    x10, 0x90004000
    DMA.STORE x10, x6, x13, 0
    LI    x6, 256
    LI    x10, 0x90004400
    DMA.STORE x10, x6, x13, 1
    LI    x6, 512
    LI    x10, 0x90004800
    DMA.STORE x10, x6, x13, 2
    LI    x6, 768
    LI    x10, 0x90004C00
    DMA.STORE x10, x6, x13, 3
    LI    x6, 1024
    LI    x10, 0x90005000
    DMA.STORE x10, x6, x13, 4
    LI    x6, 1280
    LI    x10, 0x90005400
    DMA.STORE x10, x6, x13, 5
    LI    x6, 1536
    LI    x10, 0x90005800
    DMA.STORE x10, x6, x13, 6
    LI    x6, 1792
    LI    x10, 0x90005C00
    DMA.STORE x10, x6, x13, 7
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

    CSRRS x21, 0xC00, x0        # x21 = cycle count at end
    SUB   x22, x21, x20
    CSRRW x0,  0xC11, x22       # stash in CSR_DBG1
    ADDI  x30, x0, 1
    CSRW  x30, 0xC10
    ECALL

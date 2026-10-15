# perf_dma_load_8x1k.S — DMA calibration: eight 1 KiB loads on all channels, then eight waits
# Banks the mcycles delta around the DMA region into CSR_DBG1 so the
# npu-model memory backend can be fitted (npu-model/scripts/calibrate_dma_backend.py).
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PERF_REPORT

    li    x5, 0x00000000
    dma.config x5, ch0
    li    x12, 32
    li    x13, 1024
    li    x14, 4096
    csrrs x20, 0xC00, x0        # x20 = cycle count at start
    li    x6, 0
    li    x10, 0x90000000
    dma.load  x6, x10, x13, ch0
    li    x6, 256
    li    x10, 0x90000400
    dma.load  x6, x10, x13, ch1
    li    x6, 512
    li    x10, 0x90000800
    dma.load  x6, x10, x13, ch2
    li    x6, 768
    li    x10, 0x90000C00
    dma.load  x6, x10, x13, ch3
    li    x6, 1024
    li    x10, 0x90001000
    dma.load  x6, x10, x13, ch4
    li    x6, 1280
    li    x10, 0x90001400
    dma.load  x6, x10, x13, ch5
    li    x6, 1536
    li    x10, 0x90001800
    dma.load  x6, x10, x13, ch6
    li    x6, 1792
    li    x10, 0x90001C00
    dma.load  x6, x10, x13, ch7
    dma.wait  ch0
    dma.wait  ch1
    dma.wait  ch2
    dma.wait  ch3
    dma.wait  ch4
    dma.wait  ch5
    dma.wait  ch6
    dma.wait  ch7

    csrrs x21, 0xC00, x0        # x21 = cycle count at end
    sub   x22, x21, x20
    csrrw x0,  0xC11, x22       # stash in CSR_DBG1
    addi  x30, x0, 1
    csrrw x0, 0xC10, x30
    ecall

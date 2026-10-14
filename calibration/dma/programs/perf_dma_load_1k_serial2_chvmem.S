# perf_dma_load_1k_serial2_chvmem.S — DMA calibration: two dependent 1 KiB loads, different channels, same VMEM slot
# Banks the mcycles delta around the DMA region into CSR_DBG1 so the
# npu-model memory backend can be fitted (npu-model/scripts/calibrate_dma_backend.py).
# @TIMEOUT 100000
# @DRAM_BASE 0x90000000
# @PERF_REPORT

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 32
    LI    x13, 1024
    LI    x14, 4096
    CSRRS x20, 0xC00, x0        # x20 = cycle count at start
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x13, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x13, 1
    DMA.WAIT  1
    CSRRS x21, 0xC00, x0        # x21 = cycle count at end
    SUB   x22, x21, x20
    CSRRW x0,  0xC11, x22       # stash in CSR_DBG1
    ADDI  x30, x0, 1
    CSRW  x30, 0xC10
    ECALL

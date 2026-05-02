# Matmul kernel
# 32x32 FP8 activation @ 32x32 FP8 weight -> 32x32 BF16 output
#
# Register map:
#   x1 = VMEM_ACTIVATION_BASE = 0x2000
#   x2 = VMEM_WEIGHT_BASE     = 0x2400
#   x3 = VMEM_OUTPUT_BASE     = 0x2800
#   x4 = DRAM_ACTIVATION_BASE = 0x0000
#   x5 = DRAM_WEIGHT_BASE     = 0x0400
#   x6 = 1024  (tile byte length)

# --- Register Setup ---
lui x4, 0x80000     # DRAM_ACTIVATION_BASE = 0x80000000
addi x5, x4, 1024   # DRAM_WEIGHT_BASE = 0x80000400
lui x1, 0x20002     # VMEM_ACTIVATION_BASE = 0x20002000
addi x2, x1, 1024   # VMEM_WEIGHT_BASE = 0x20002400
addi x3, x2, 1024   # VMEM_OUTPUT_BASE = 0x20002800
addi x6, x0, 1024   # tile byte length

# --- DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x6    # activations -> VMEM[x1]
dma.load.ch1 x2, x5, x6    # weights -> VMEM[x2]
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)     # activations
delay 34
vload m1, 0(x2)     # weights
delay 34

# --- Matmul ---
vmatpush.weight.mxu0 w0, m1
# VPU local transfer (1KB) is ~16 cycles at 64B/cycle
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m2, acc0
delay 32

# --- MRF -> VMEM ---
vstore m2, 0(x3)
delay 34
vstore m3, 32(x3)
# two vstores are ~2x16 cycles; add slack before DMA reads VMEM
delay 34

# --- VMEM -> DRAM ---
# DRAM_OUTPUT_BASE = 0x80000800: LUI 0x80001 (0x80001000) + ADDI -2048 = 0x80000800
lui x10, 0x80001
addi x10, x10, -2048        # x10 = 0x0800 (DRAM output, first half)
addi x11, x10, 1024         # x11 = 0x0C00 (DRAM output, second half)
# IMPORTANT: DMA ops read XRF at *execute* time. Since DMA now has non-trivial
# latency (based on XRF[rs2] length), don't mutate x3 between the two stores.
dma.store.ch0 x10, x3, x6
addi x12, x3, 1024           # x12 = VMEM second half; can't mutate x3 during ch0
dma.store.ch1 x11, x12, x6
dma.wait.ch0
dma.wait.ch1

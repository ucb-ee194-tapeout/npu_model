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
lui x4, 0x80000  # DRAM_ACTIVATION_BASE = 0x0000
lui x5, 0x80000  # DRAM_WEIGHT_BASE = 0x0400
addi x5, x5, 1024
li x1, 0x2000       # VMEM_ACTIVATION_BASE
li x2, 0x2400       # VMEM_WEIGHT_BASE
li x3, 0x2800       # VMEM_OUTPUT_BASE
# Note: 0x800 is -2048 in signed 12-bit, so 0x2800 = lui 3 + addi -2048
addi x6, x0, 1024   # tile byte length

# --- DRAM -> VMEM ---
dma.config x0

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
# DRAM_OUTPUT_BASE = 0x0800: LUI 1 (0x1000) + ADDI -2048 = 0x0800
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

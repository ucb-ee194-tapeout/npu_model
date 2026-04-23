# SmolVLA matmul kernel
# C = A_fp8 @ B_fp8 → bf16, single 32x32 tile.
# Output written as two 32x16 halves stacked in DRAM.
#
# Scalar register map:
#   x1 = DRAM_A   = 0x0000    x2 = DRAM_B    = 0x0500
#   x3 = DRAM_OUT = 0x0B00    x4 = VMEM_A    = 0x2000
#   x5 = VMEM_B   = 0x2400    x6 = VMEM_OUT_H0 = 0x2800
#   x7 = 1024 (fp8 tile bytes)  x8 = 2048 (bf16 output bytes)
#   x9 = VMEM_OUT_H1 = 0x2C00  (computed mid-program)

# --- Scalar setup ---
addi x1, x0, 0              # x1 = 0x0000 DRAM_A
addi x2, x0, 1280           # x2 = 0x0500 DRAM_B
lui x3, 0x1                 # x3 = 0x1000
addi x3, x3, -1280          # x3 = 0x0B00 DRAM_OUT
lui x4, 0x2                 # x4 = 0x2000 VMEM_A
addi x5, x4, 1024           # x5 = 0x2400 VMEM_B
addi x6, x5, 1024           # x6 = 0x2800 VMEM_OUT_H0
addi x7, x0, 1024           # x7 = 1024 (fp8 tile bytes)
lui x8, 0x1                 # x8 = 0x1000
addi x8, x8, -2048          # x8 = 2048 (bf16 output bytes)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x4, x1, x7    # A
dma.load.ch1 x5, x2, x7    # B
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x4)             # A fp8
delay 34
vload m1, 0(x5)             # B fp8
delay 34

# --- fp8 matmul via MXU ---
vmatpush.weight.mxu0 w0, m1
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m2, acc0     # (m2, m3) = C bf16
delay 32

# --- MRF -> VMEM -> DRAM ---
vstore m2, 0(x6)            # OUT_H0
delay 34
addi x9, x6, 1024           # x9 = 0x2C00 VMEM_OUT_H1
vstore m3, 0(x9)            # OUT_H1
delay 34
dma.store.ch0 x3, x6, x8
dma.wait.ch0
ecall

# SmolVLA reduction_sum kernel
# Row-wise sum of a 32x32 bf16 tile, broadcast to a 32x16 output.
# vredsum.row.bf16 operates on the full (m0,m1) pair in one shot.
#
# Scalar register map:
#   x1 = VMEM_X     = 0x2000    x2 = VMEM_OUT = 0x3000
#   x3 = DRAM_X_H0  = 0x0000    x4 = DRAM_X_H1 = 0x0400
#   x5 = DRAM_OUT   = 0x0B00    x6 = 1024 (per-half bytes)
#   x7 = VMEM_X_H1  = 0x2400    (computed before DMA)
#
# MRF register map:
#   (m0, m1) = X   (m4, m5) = row-sum broadcast

# --- Scalar setup ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_X
lui x2, 0x3                 # x2 = 0x3000 VMEM_OUT
addi x3, x0, 0              # x3 = 0x0000 DRAM_X_H0
addi x4, x3, 1024           # x4 = 0x0400 DRAM_X_H1
lui x5, 0x1                 # x5 = 0x1000
addi x5, x5, -1280          # x5 = 0x0B00 DRAM_OUT
addi x6, x0, 1024           # x6 = 1024 (per-half bytes)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
addi x7, x1, 1024           # x7 = 0x2400 VMEM_X_H1
dma.load.ch0 x1, x3, x6    # X_H0
dma.load.ch1 x7, x4, x6    # X_H1
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # X h0
delay 34
vload m1, 32(x1)            # X h1
delay 34

# --- Row-sum broadcast: (m4, m5) = rowsum(m0, m1) ---
vredsum.row.bf16 m4, m0
delay 69

# --- MRF -> VMEM -> DRAM (only h0; output is 32x16) ---
vstore m4, 0(x2)
delay 34
dma.store.ch0 x5, x2, x6
dma.wait.ch0

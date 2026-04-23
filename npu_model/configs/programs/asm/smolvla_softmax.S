# SmolVLA softmax kernel
# Row-wise stable softmax on a 32x32 bf16 tile.
# softmax(x) = exp(x - rowmax) / rowsum(exp(x - rowmax))
#
# Scalar register map:
#   x1 = VMEM_X     = 0x2000    x2 = VMEM_OUT   = 0x3000
#   x3 = DRAM_X_H0  = 0x0000    x4 = DRAM_X_H1  = 0x0400
#   x5 = DRAM_OUT_H0 = 0x0B00   x6 = 1024 (per-half bytes)
#   x7 = VMEM_X_H1  = 0x2400    (for DMA load)
#   x8 = DRAM_OUT_H1 = 0x0F00   x9 = VMEM_OUT_H1 = 0x3400 (computed mid-program)
#
# MRF register map:
#   (m0, m1)   = X              (m2, m3)   = rowmax(X)
#   (m4, m5)   = X - rowmax     (m6, m7)   = exp(X - rowmax)
#   (m8, m9)   = rowsum(exp)    (m10, m11) = 1 / rowsum
#   (m12, m13) = softmax(X)

# --- Scalar setup ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_X
lui x2, 0x3                 # x2 = 0x3000 VMEM_OUT
addi x3, x0, 0              # x3 = 0x0000 DRAM_X_H0
addi x4, x3, 1024           # x4 = 0x0400 DRAM_X_H1
lui x5, 0x1                 # x5 = 0x1000
addi x5, x5, -1280          # x5 = 0x0B00 DRAM_OUT_H0
addi x6, x0, 1024           # x6 = 1024 (per-half bytes)
addi x7, x1, 1024           # x7 = 0x2400 VMEM_X_H1

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x3, x6    # X_H0
dma.load.ch1 x7, x4, x6    # X_H1
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # X h0
delay 34
vload m1, 32(x1)            # X h1
delay 34

# --- Stable softmax ---
vredmax.row.bf16 m2, m0     # (m2, m3) = rowmax(X)
delay 34
vsub.bf16 m4, m0, m2        # (m4, m5) = X - rowmax
delay 66
vexp.bf16 m6, m4            # (m6, m7) = exp(X - rowmax)
delay 66
vredsum.row.bf16 m8, m6     # (m8, m9) = rowsum(exp)
delay 39
vrecip.bf16 m10, m8         # (m10, m11) = 1 / rowsum
delay 66
vmul.bf16 m12, m6, m10      # (m12, m13) = softmax(X)
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m12, 0(x2)
delay 34
vstore m13, 32(x2)
delay 34
addi x8, x5, 1024           # x8 = 0x0F00 DRAM_OUT_H1
addi x9, x2, 1024           # x9 = 0x3400 VMEM_OUT_H1
dma.store.ch0 x5, x2, x6
dma.store.ch1 x8, x9, x6
dma.wait.ch0
dma.wait.ch1

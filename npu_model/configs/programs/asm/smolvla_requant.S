# SmolVLA requant kernel
# bf16 -> fp8 requantization with unit scale.
# Reads two 32x16 bf16 halves, packs to a 32x32 fp8 tile via vpack.bf16.fp8.
#
# Scalar register map:
#   x1 = VMEM_X_H0  = 0x2000    x2 = VMEM_X_H1 = 0x2400
#   x3 = VMEM_OUT   = 0x3000    x4 = DRAM_X_H0 = 0x0000
#   x5 = DRAM_X_H1  = 0x0400    x6 = DRAM_OUT  = 0x0B00
#   x7 = 1024 (transfer size)
#   e5 = 1 (unit scale for vpack)
#
# MRF register map:
#   (m0, m1) = X bf16 pair    m2 = packed fp8 output

# --- Scalar setup ---
lui x1, 0x20002             # x1 = 0x20002000 VMEM_X_H0
addi x2, x1, 1024           # x2 = 0x20002400 VMEM_X_H1
lui x3, 0x20003             # x3 = 0x20003000 VMEM_OUT
lui x4, 0x80000             # x4 = 0x80000000 DRAM_X_H0
addi x5, x4, 1024           # x5 = 0x80000400 DRAM_X_H1
lui x6, 0x80001
addi x6, x6, -1280          # x6 = 0x80000B00 DRAM_OUT
addi x7, x0, 1024           # x7 = 1024 (transfer size)

# unit scale for bf16->fp8 pack
seli e5, 1

# --- DMA: DRAM -> VMEM ---
dma.config x0
dma.load.ch0 x1, x4, x7    # X_H0
dma.load.ch1 x2, x5, x7    # X_H1
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # X h0
delay 34
vload m1, 0(x2)             # X h1
delay 34

# --- Pack bf16 pair (m0,m1) -> fp8 tile m2 ---
vpack.bf16.fp8 m2, m0, e5
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m2, 0(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0

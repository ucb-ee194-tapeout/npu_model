# SmolVLA attention kernel
# Single-tile scaled dot-product attention: softmax((Q @ K) * scale) @ V
# Q, K, V are fp8 32x32 tiles; scale is an elementwise bf16 mask/scale.
# Writes the bf16 output as two 32x16 halves.
#
# MRF layout:
#   m0      = Q (fp8 tile)
#   m2      = K (fp8 tile)
#   m4      = V (fp8 tile)
#   m6,m7   = scale (bf16 pair, duplicated halves)
#   m10,m11 = scores (bf16 pair) = Q @ K
#   m12,m13 = scaled
#   m14,m15 = rowmax (broadcast)
#   m16,m17 = scaled - rowmax
#   m18,m19 = exp(scaled - rowmax)
#   m20,m21 = rowsum(exp)
#   m22,m23 = 1 / rowsum
#   m24,m25 = probs (bf16)
#   m26     = probs (fp8, packed from m24,m25)
#   m28,m29 = out = probs @ V (bf16 pair)
#
# VMEM staging (each tile = 1024 B):
#   x1 = 0x2000 Q           x2 = 0x2400 K          x3 = 0x2800 V
#   x4 = 0x2C00 scale h0    x5 = 0x3000 scale h1
#   x6 = 0x4000 OUT h0      x7 = 0x4400 OUT h1

# --- VMEM addresses ---
lui x1, 0x2                 # x1 = 0x2000 Q
addi x2, x1, 1024           # x2 = 0x2400 K
addi x3, x2, 1024           # x3 = 0x2800 V
addi x4, x3, 1024           # x4 = 0x2C00 scale h0
lui x5, 0x3                 # x5 = 0x3000 scale h1
lui x6, 0x4                 # x6 = 0x4000 OUT h0
addi x7, x6, 1024           # x7 = 0x4400 OUT h1

# --- DRAM addresses ---
addi x8, x0, 0              # x8  = 0x0000 DRAM_Q
addi x9, x8, 1024           # x9  = 0x0400 DRAM_K
addi x10, x9, 1024          # x10 = 0x0800 DRAM_V
addi x11, x10, 1024         # x11 = 0x0C00 DRAM_SCALE
lui x12, 0x1                # x12 = 0x1000 DRAM_OUT_H0
addi x13, x12, 1024         # x13 = 0x1400 DRAM_OUT_H1
addi x14, x0, 1024          # x14 = 1024 per-half size

# --- DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x8, x14   # Q
dma.load.ch1 x2, x9, x14   # K
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x10, x14  # V
dma.load.ch1 x4, x11, x14  # scale h0
dma.wait.ch0
dma.wait.ch1
# scale is 32x32 bf16 (2048 B); load second 1024 B half
addi x15, x11, 1024         # x15 = DRAM_SCALE + 1024
dma.load.ch0 x5, x15, x14  # scale h1
dma.wait.ch0

# --- VMEM -> MRF ---
vload m0, 0(x1)             # Q
delay 34
vload m2, 0(x2)             # K
delay 34
vload m4, 0(x3)             # V
delay 34
vload m6, 0(x4)             # scale h0
delay 34
vload m7, 0(x5)             # scale h1
delay 34

# --- Matmul 1: scores = Q @ K (push K as weight; activation = Q) ---
vmatpush.weight.mxu0 w0, m2
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m10, acc0    # (m10, m11)
delay 32

# --- Scale: (m12, m13) = scores * scale ---
vmul.bf16 m12, m10, m6
delay 66

# --- Stable softmax ---
vredmax.row.bf16 m14, m12
delay 69
vsub.bf16 m16, m12, m14
delay 66
vexp.bf16 m18, m16
delay 66
vredsum.row.bf16 m20, m18
delay 69
vrecip.bf16 m22, m20
delay 66
vmul.bf16 m24, m18, m22
delay 66

# --- Pack probs BF16 -> FP8 into m26 with unit scale ---
seli e0, 1
vpack.bf16.fp8 m26, m24, e0
delay 66

# --- Matmul 2: out = packed_probs @ V ---
vmatpush.weight.mxu0 w0, m4
delay 32
vmatmul.mxu0 acc0, m26, w0
delay 96
vmatpop.bf16.acc.mxu0 m28, acc0    # (m28, m29)
delay 32

# --- MRF -> VMEM -> DRAM ---
vstore m28, 0(x6)
delay 34
vstore m29, 0(x7)
delay 34
dma.store.ch0 x12, x6, x14
dma.store.ch1 x13, x7, x14
dma.wait.ch0
dma.wait.ch1

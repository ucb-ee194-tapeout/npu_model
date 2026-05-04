# SmolVLA fused_norm_scale kernel
# out[i,j] = matrix[i,j] * rsqrt(variance[i,j]) on 32x32 bf16 tiles.
# rsqrt implemented as vrecip(vsqrt(x)) — no dedicated VRSQRT.
#
# Scalar register map:
#   x1 = VMEM_VAR = 0x2000    x2 = VMEM_MAT = 0x3000    x3 = VMEM_OUT = 0x4000
#   x4 = DRAM_VAR = 0x0000    x5 = DRAM_MAT = 0x0800    x6 = DRAM_OUT = 0x1000
#   x7 = 2048 (TILE_BYTES)
#
# MRF register map:
#   (m0, m1) = variance        (m2, m3) = matrix
#   (m4, m5) = sqrt(variance)  (m6, m7) = rsqrt(variance)
#   (m8, m9) = out = matrix * rsqrt

# --- Scalar setup ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_VAR
lui x2, 0x3                 # x2 = 0x3000 VMEM_MAT
lui x3, 0x4                 # x3 = 0x4000 VMEM_OUT
lui x4, 0x80000  # x4 = 0x0000 DRAM_VAR
lui x5, 0x80001  # x5 = 0x1000
addi x5, x5, -2048          # x5 = 0x0800 DRAM_MAT
lui x6, 0x80001  # x6 = 0x1000 DRAM_OUT
lui x7, 0x1                 # x7 = 0x1000
addi x7, x7, -2048          # x7 = 2048 (TILE_BYTES)

# --- DMA: DRAM -> VMEM ---
dma.config x0

dma.wait.ch0
dma.load.ch0 x1, x4, x7    # variance
dma.load.ch1 x2, x5, x7    # matrix
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # variance h0
delay 34
vload m1, 32(x1)            # variance h1
delay 34
vload m2, 0(x2)             # matrix h0
delay 34
vload m3, 32(x2)            # matrix h1
delay 34

# --- op_a: rsqrt = 1 / sqrt(variance) ---
vsqrt.bf16 m4, m0           # (m4, m5) = sqrt(variance)
delay 66
vrecip.bf16 m6, m4          # (m6, m7) = rsqrt(variance)
delay 66

# --- op_b: out = matrix * rsqrt ---
vmul.bf16 m8, m2, m6        # (m8, m9) = out
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m8, 0(x3)
delay 34
vstore m9, 32(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0

# VPU arithmetic correctness test
# Computes: vmul(vsquare(vsub(vadd(x,x), x)), vcube(x))
# = (x+x - x)^2 * x^3 = x^2 * x^3 = x^5 (in bf16)
#
# Scalar register map:
#   x1 = VMEM_INPUT  = 0x2000    x2 = VMEM_OUTPUT = 0x2800
#   x3 = DRAM_INPUT  = 0x0000    x4 = DRAM_OUTPUT = 0x0800
#   x5 = 2048 (TILE_BYTES)
#
# MRF register map:
#   (m0, m1)   = x              (m2, m3)   = x + x
#   (m4, m5)   = identity(x)   (m6, m7)   = identity^2
#   (m8, m9)   = x^3            (m10, m11) = result

# --- Scalar setup ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_INPUT
lui x2, 0x3                 # x2 = 0x3000
addi x2, x2, -2048          # x2 = 0x2800 VMEM_OUTPUT
lui x3, 0x80000  # x3 = 0x0000 DRAM_INPUT
lui x4, 0x80001  # x4 = 0x1000
addi x4, x4, -2048          # x4 = 0x0800 DRAM_OUTPUT
lui x5, 0x1                 # x5 = 0x1000
addi x5, x5, -2048          # x5 = 2048 (TILE_BYTES)

# --- DMA: DRAM -> VMEM ---
dma.config x0

dma.wait.ch0
dma.load.ch0 x1, x3, x5
dma.wait.ch0

# --- VMEM -> MRF ---
vload m0, 0(x1)             # x h0
delay 34
vload m1, 32(x1)            # x h1
delay 34

# --- Compute ---
vadd.bf16 m2, m0, m0        # (m2, m3) = x + x
delay 66
vsub.bf16 m4, m2, m0        # (m4, m5) = identity(x) = (x+x) - x
delay 66
vsquare.bf16 m6, m4         # (m6, m7) = identity^2
delay 66
vcube.bf16 m8, m0           # (m8, m9) = x^3
delay 66
vmul.bf16 m10, m6, m8       # (m10, m11) = result
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m10, 0(x2)
delay 34
vstore m11, 32(x2)
delay 34
dma.store.ch0 x4, x2, x5
dma.wait.ch0

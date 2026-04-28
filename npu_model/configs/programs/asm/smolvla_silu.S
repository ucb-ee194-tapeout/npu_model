# SmolVLA SiLU kernel
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) on a 32x32 bf16 tile.
#
# Scalar register map:
#   x1 = VMEM_INPUT  = 0x2000    x2 = VMEM_OUTPUT = 0x2800
#   x3 = DRAM_INPUT  = 0x0000    x4 = DRAM_OUTPUT = 0x0800
#   x5 = 2048 (TILE_BYTES)
#
# MRF register map:
#   (m0, m1)   = x              (m2, m3)   = -1.0 const
#   (m4, m5)   = +1.0 const     (m6, m7)   = -x
#   (m8, m9)   = exp(-x)        (m10, m11) = 1 + exp(-x)
#   (m12, m13) = sigmoid(x)     (m14, m15) = silu(x)

# --- Scalar setup ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_INPUT
lui x2, 0x3                 # x2 = 0x3000
addi x2, x2, -2048          # x2 = 0x2800 VMEM_OUTPUT
addi x3, x0, 0              # x3 = 0x0000 DRAM_INPUT
lui x4, 0x1                 # x4 = 0x1000
addi x4, x4, -2048          # x4 = 0x0800 DRAM_OUTPUT
lui x5, 0x1                 # x5 = 0x1000
addi x5, x5, -2048          # x5 = 2048 (TILE_BYTES)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x3, x5
dma.wait.ch0

# --- VMEM -> MRF ---
vload m0, 0(x1)             # x h0
delay 34
vload m1, 32(x1)            # x h1
delay 34

# --- Broadcast constants ---
vli.all m2, 0xbf80              # -1.0 h0
delay 65
vli.all m3, 0xbf80              # -1.0 h1
delay 65
vli.all m4, 0x3f80          # +1.0 h0
delay 65
vli.all m5, 0x3f80          # +1.0 h1
delay 65

# --- SiLU: x / (1 + exp(-x)) ---
vmul.bf16 m6, m0, m2        # (m6, m7) = -x
delay 66
vexp.bf16 m8, m6            # (m8, m9) = exp(-x)
delay 66
vadd.bf16 m10, m8, m4       # (m10, m11) = 1 + exp(-x)
delay 66
vrecip.bf16 m12, m10        # (m12, m13) = sigmoid(x)
delay 66
vmul.bf16 m14, m0, m12      # (m14, m15) = silu(x)
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m14, 0(x2)
delay 34
vstore m15, 32(x2)
delay 34
dma.store.ch0 x4, x2, x5
dma.wait.ch0

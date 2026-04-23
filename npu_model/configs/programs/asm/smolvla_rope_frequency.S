# SmolVLA rope_frequency kernel
# y = cos(x) elementwise on a 32x32 bf16 tile (RoPE building block).
#
# Scalar register map:
#   x1 = VMEM_X   = 0x2000    x4 = DRAM_X   = 0x0000
#   x5 = DRAM_OUT = 0x0B00    x6 = 2048 (TILE_BYTES)
#
# MRF register map:
#   (m0, m1) = X    (m2, m3) = cos(X)

# --- Scalar setup ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_X
addi x4, x0, 0              # x4 = 0x0000 DRAM_X
lui x5, 0x1                 # x5 = 0x1000
addi x5, x5, -1280          # x5 = 0x0B00 DRAM_OUT
lui x6, 0x1                 # x6 = 0x1000
addi x6, x6, -2048          # x6 = 2048 (TILE_BYTES)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x6
dma.wait.ch0

# --- VMEM -> MRF ---
vload m0, 0(x1)             # X h0
delay 34
vload m1, 32(x1)            # X h1
delay 34

# --- cos: pair(m2,m3) = cos(pair(m0,m1)) ---
vcos.bf16 m2, m0            # (m2, m3) = cos(X)
delay 66
# FIXME: pair semantics already wrote m3 = cos(m1) above; second vcos
# overwrites m3 with cos(m1) again (same value) and writes spurious m4.
vcos.bf16 m3, m1
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m2, 0(x1)
delay 34
vstore m3, 32(x1)
delay 34
dma.store.ch0 x5, x1, x6
dma.wait.ch0

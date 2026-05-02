# SmolVLA elementwise multiply kernel
# y = a * b on two 32x32 bf16 tiles.
#
# Register map:
#   x1 = VMEM_A   = 0x4000
#   x2 = VMEM_B   = 0x5000
#   x3 = VMEM_OUT = 0x6000
#   x4 = DRAM_A   = 0x0000
#   x5 = DRAM_B   = 0x1000
#   x6 = DRAM_OUT = 0x2000
#   x7 = 2048  (TILE_BYTES)

# --- Scalar setup ---
lui x1, 0x20004             # x1 = VMEM_A = 0x20004000
lui x2, 0x20005             # x2 = VMEM_B = 0x20005000
lui x3, 0x20006             # x3 = VMEM_OUT = 0x20006000
lui x4, 0x80000             # x4 = DRAM_A = 0x80000000
lui x5, 0x80001             # x5 = DRAM_B = 0x80001000
lui x6, 0x80002             # x6 = DRAM_OUT = 0x80002000
lui x7, 0x1                 # x7 = 0x1000
addi x7, x7, -2048          # x7 = 2048

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x7    # VMEM[x1] <- DRAM[x4]
dma.load.ch1 x2, x5, x7    # VMEM[x2] <- DRAM[x5]
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # A h0
delay 34
vload m1, 32(x1)            # A h1
delay 34
vload m2, 0(x2)             # B h0
delay 34
vload m3, 32(x2)            # B h1
delay 34

# --- Elementwise multiply ---
vmul.bf16 m4, m0, m2        # pair(m4,m5) = pair(m0,m1) * pair(m2,m3) = (A*B h0, A*B h1)
delay 66
# FIXME: pair semantics already wrote m5 = A_h1*B_h1 above
# vmul.bf16 m5, m1, m3      
# delay 4

# --- MRF -> VMEM -> DRAM ---
vstore m4, 0(x3)
delay 34
vstore m5, 32(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0

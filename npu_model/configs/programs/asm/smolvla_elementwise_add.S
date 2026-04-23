# SmolVLA elementwise addition kernel
# y = a + b on two 32x32 bf16 tiles.
#
# Register map:
#   x1 = VMEM_A   = 0x4000
#   x2 = VMEM_B   = 0x5000
#   x3 = VMEM_OUT = 0x6000
#   x4 = DRAM_A   = 0x0000
#   x5 = DRAM_B   = 0x1000
#   x6 = DRAM_OUT = 0x2000
#   x7 = 2048  (TILE_BYTES)

# --- Scalar setup: VMEM / DRAM addresses + transfer size ---
lui x1, 0x4                 # x1 = VMEM_A = 0x4000
lui x2, 0x5                 # x2 = VMEM_B = 0x5000
lui x3, 0x6                 # x3 = VMEM_OUT = 0x6000
addi x4, x0, 0              # x4 = DRAM_A = 0x0000
lui x5, 0x1                 # x5 = DRAM_B = 0x1000
lui x6, 0x2                 # x6 = DRAM_OUT = 0x2000
# Transfer size = 2048 (= 0x800). Split via lui + addi because
# 2048 exceeds the signed 12-bit addi immediate range.
lui x7, 0x1                 # x7 = 0x1000
addi x7, x7, -2048          # x7 = 2048

# --- DMA: DRAM -> VMEM (channels 0 and 1 in parallel) ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x7    # VMEM[x1] <- DRAM[x4]
dma.load.ch1 x2, x5, x7    # VMEM[x2] <- DRAM[x5]
dma.wait.ch0
dma.wait.ch1

# --- Load 2 halves per operand (imm12 in units of 32 bytes) ---
vload m0, 0(x1)             # A h0
delay 34
vload m1, 32(x1)            # A h1 (next 1024 B)
delay 34
vload m2, 0(x2)             # B h0
delay 34
vload m3, 32(x2)            # B h1
delay 34

# --- Per-half add ---
vadd.bf16 m4, m0, m2
delay 66
# FIXME: In OG kernel but seems like a bug.
# New semantics = one add necessary to sum 32x32
# vadd.bf16 m5, m1, m3
# delay 4

# --- Store: MRF -> VMEM -> DRAM ---
vstore m4, 0(x3)
delay 34
vstore m5, 32(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0

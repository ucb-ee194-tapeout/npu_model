# SmolVLA elementwise division kernel
# y = a / b via vrecip + vmul (no dedicated divide op).
#
# Register map:
#   x1  = VMEM_A_H0   = 0x2000    x2  = VMEM_A_H1   = 0x2400
#   x3  = VMEM_B_H0   = 0x2800    x4  = VMEM_B_H1   = 0x2C00
#   x5  = VMEM_OUT_H0 = 0x3000    x6  = VMEM_OUT_H1 = 0x3400
#   x7  = DRAM_A_H0   = 0x0000    x8  = DRAM_A_H1   = 0x0400
#   x9  = DRAM_B_H0   = 0x0800    x10 = DRAM_B_H1   = 0x0C00
#   x11 = DRAM_OUT_H0 = 0x0C00 (B_H1 overwritten in place after load)
#   x12 = DRAM_OUT_H1 = 0x1000    x13 = 1024 (per-half size)

# --- Scalar setup: VMEM addresses ---
lui x1, 2                   # x1 = 0x2000 VMEM_A_H0
addi x2, x1, 1024           # x2 = 0x2400 VMEM_A_H1
addi x3, x2, 1024           # x3 = 0x2800 VMEM_B_H0
addi x4, x3, 1024           # x4 = 0x2C00 VMEM_B_H1
lui x5, 3                   # x5 = 0x3000 VMEM_OUT_H0
addi x6, x5, 1024           # x6 = 0x3400 VMEM_OUT_H1

# --- Scalar setup: DRAM addresses + transfer size ---
lui x7, 0x80000  # x7  = 0x0000 DRAM_A_H0
lui x8, 0x80000  # x8  = 0x0400 DRAM_A_H1
addi x8, x8, 1024
addi x9, x8, 1024           # x9  = 0x0800 DRAM_B_H0
addi x10, x9, 1024          # x10 = 0x0C00 DRAM_B_H1
lui x11, 0x80001  # x11 = 0x1000
addi x11, x11, -1024        # x11 = 0x0C00 DRAM_OUT_H0 (= B_H1, overwritten after load)
addi x12, x11, 1024         # x12 = 0x1000 DRAM_OUT_H1
addi x13, x0, 1024          # x13 = 1024 per-half size

# --- DMA: DRAM -> VMEM ---
dma.config x0

dma.config.ch1 x0

dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x1, x7, x13   # A_H0 -> VMEM[x1]
dma.load.ch1 x2, x8, x13   # A_H1 -> VMEM[x2]
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x9, x13   # B_H0 -> VMEM[x3]
dma.load.ch1 x4, x10, x13  # B_H1 -> VMEM[x4]
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # A_H0
delay 34
vload m1, 0(x2)             # A_H1
delay 34
vload m2, 0(x3)             # B_H0
delay 34
vload m3, 0(x4)             # B_H1
delay 34

# --- Compute: a / b = a * (1/b) ---
vrecip.bf16 m4, m2          # pair(m4,m5) = 1/pair(m2,m3) = (1/B_H0, 1/B_H1)
delay 66
# FIXME: pair semantics already wrote m5 = 1/B_H1 above
# vrecip.bf16 m5, m3        
# delay 16
vmul.bf16 m6, m0, m4        # pair(m6,m7) = pair(m0,m1) * pair(m4,m5) = (A/B_H0, A/B_H1)
delay 66
# FIXME: pair semantics already wrote m7 = A_H1/B_H1 above
# vmul.bf16 m7, m1, m5     
# delay 4

# --- MRF -> VMEM -> DRAM ---
vstore m6, 0(x5)            # OUT_H0
delay 34
vstore m7, 0(x6)            # OUT_H1
delay 34
dma.store.ch0 x11, x5, x13
dma.store.ch1 x12, x6, x13
dma.wait.ch0
dma.wait.ch1

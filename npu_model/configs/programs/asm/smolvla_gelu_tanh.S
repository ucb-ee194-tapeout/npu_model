# SmolVLA gelu_tanh kernel
# y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# NOTE: Autocomp-generated. No golden_result — tabulated constants (C1-C3)
# must be populated from the original manifest before numerical validation.
#
# DRAM layout (each half = 32x16 bf16 = 1024 B):
#   x8=DRAM_X_H0=0x0000  x9=DRAM_X_H1=0x0400
#   x10=DRAM_C1=0x0800   x11=DRAM_C2=0x0C00  x12=DRAM_C3=0x1000
#   x14=DRAM_OUT_H0=0x1400  x15=DRAM_OUT_H1=0x1800
#
# VMEM layout:
#   x1=VMEM_X_H0=0x2000  x2=VMEM_X_H1=0x2400
#   x3=VMEM_C1=0x2800    x4=VMEM_C2=0x2C00   x5=VMEM_C3=0x3000
#   x6=VMEM_OUT_H0=0x3400  x7=VMEM_OUT_H1=0x3800
#   x16=1024 (per-half transfer size)
#
# MRF: m0=X_H0  m1=X_H1  m2=C1(0.044715)  m3=C2(sqrt(2/pi))
#       m4=C3(0.5?)  m5=1.0

# --- Scalar setup: VMEM addresses ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_X_H0
addi x2, x1, 1024           # x2 = 0x2400 VMEM_X_H1
addi x3, x2, 1024           # x3 = 0x2800 VMEM_C1
addi x4, x3, 1024           # x4 = 0x2C00 VMEM_C2
lui x5, 0x3                 # x5 = 0x3000 VMEM_C3
addi x6, x5, 1024           # x6 = 0x3400 VMEM_OUT_H0
addi x7, x6, 1024           # x7 = 0x3800 VMEM_OUT_H1

# --- Scalar setup: DRAM addresses + transfer size ---
addi x8, x0, 0              # x8  = 0x0000 DRAM_X_H0
addi x9, x0, 1024           # x9  = 0x0400 DRAM_X_H1
addi x10, x9, 1024          # x10 = 0x0800 DRAM_C1
addi x11, x10, 1024         # x11 = 0x0C00 DRAM_C2
lui x12, 0x1                # x12 = 0x1000 DRAM_C3
lui x14, 0x1                # x14 = 0x1000
addi x14, x14, 1024         # x14 = 0x1400 DRAM_OUT_H0
addi x15, x14, 1024         # x15 = 0x1800 DRAM_OUT_H1
addi x16, x0, 1024          # x16 = 1024 (per-half size)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x8, x16   # X_H0
dma.load.ch1 x2, x9, x16   # X_H1
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x10, x16  # C1
dma.load.ch1 x4, x11, x16  # C2
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x5, x12, x16  # C3
dma.wait.ch0

# --- VMEM -> MRF ---
vload m0, 0(x1)             # X_H0
delay 34
vload m1, 0(x2)             # X_H1
delay 34
vload m2, 0(x3)             # C1 (0.044715)
delay 34
vload m3, 0(x4)             # C2 (sqrt(2/pi) ~= 0.7979)
delay 34
vload m4, 0(x5)             # C3 (0.5)
delay 34
vli.all m5, 1               # m5 = 1.0
delay 65

# --- GELU compute: h0 (pair ops act on (m0,m1) but h0 output lands in m14) ---
vmul.bf16 m6, m0, m0        # (m6, m7) = x^2
delay 66
vmul.bf16 m7, m6, m0        # (m7, m8) = x^3
delay 66
vmul.bf16 m8, m7, m3        # (m8, m9) = 0.044715 * x^3
delay 66
vadd.bf16 m9, m0, m8        # (m9, m10) = x + 0.044715*x^3
delay 66
vmul.bf16 m10, m9, m4       # (m10, m11) = sqrt(2/pi) * (x + 0.044715*x^3)
delay 66
vtanh.bf16 m11, m10         # (m11, m12) = tanh(...)
delay 66
vadd.bf16 m12, m5, m11      # (m12, m13) = 1 + tanh(...)
delay 66
vmul.bf16 m13, m0, m2       # (m13, m14) = 0.5 * x
delay 66
vmul.bf16 m14, m13, m12     # (m14, m15) = 0.5 * x * (1 + tanh(...)) = GELU h0
delay 66

# --- GELU compute: h1 (explicit vs1=m1; pair ops also write neighboring regs) ---
# FIXME: Autocomp-generated second-half pass; may be redundant given pair semantics above.
vmul.bf16 m6, m1, m1        # (m6, m7) = x_h1^2
delay 66
vmul.bf16 m7, m6, m1        # (m7, m8) = x_h1^3
delay 66
vmul.bf16 m8, m7, m3        # (m8, m9) = 0.044715 * x_h1^3
delay 66
vadd.bf16 m9, m1, m8        # (m9, m10) = x_h1 + 0.044715*x_h1^3
delay 66
vmul.bf16 m10, m9, m4       # (m10, m11) = sqrt(2/pi) * (...)
delay 66
vtanh.bf16 m11, m10         # (m11, m12) = tanh(...)
delay 66
vadd.bf16 m12, m5, m11      # (m12, m13) = 1 + tanh(...)
delay 66
vmul.bf16 m13, m1, m2       # (m13, m14) = 0.5 * x_h1
delay 66
vmul.bf16 m15, m13, m12     # (m15, m16) = GELU h1
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m14, 0(x6)
delay 34
vstore m15, 0(x7)
delay 34
dma.store.ch0 x14, x6, x16
dma.store.ch1 x15, x7, x16
dma.wait.ch0
dma.wait.ch1

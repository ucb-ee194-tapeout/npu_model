# SmolVLA gelu_tanh kernel
# y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#
# Scalar register map:
#   x1 = VMEM_X      = 0x2000   x2 = VMEM_C044  = 0x3000
#   x3 = VMEM_CSQRT  = 0x4000   x4 = VMEM_CHALF = 0x5000
#   x5 = VMEM_OUT    = 0x6000
#   x6 = DRAM_X_H0   = 0x0000   x7 = DRAM_X_H1  = 0x0400
#   x8 = DRAM_C044   = 0x0800   x9 = DRAM_CSQRT = 0x0C00
#   x10 = DRAM_CHALF = 0x1000   x11 = DRAM_OUT_H0 = 0x1400
#   x12 = DRAM_OUT_H1 = 0x1800  x13 = 1024 (tile bytes)
#   x14 = x1+1024 (VMEM x_h1)  x15 = x2+1024 (VMEM C044 pair)
#   x16 = x3+1024 (VMEM Csqrt pair)  x17 = x4+1024 (VMEM Chalf pair)
#   x18 = x5+1024 (VMEM OUT_H1)
#
# MRF register map:
#   (m0, m1) = x           (m2, m3) = 0.044715 pair
#   (m4, m5) = sqrt(2/pi)  (m6, m7) = 0.5 pair
#   (m8, m9) = 1.0
#   (m10, m11) = x^2            (m12, m13) = x^3
#   (m14, m15) = 0.044715*x^3   (m16, m17) = x + 0.044715*x^3
#   (m18, m19) = sqrt(2/pi)*(...) (m20, m21) = tanh(...)
#   (m22, m23) = 1 + tanh(...)   (m24, m25) = 0.5*x
#   (m26, m27) = GELU(x)

# --- Scalar setup: VMEM addresses ---
li x1, 0x2000       # VMEM_X
li x2, 0x3000       # VMEM_C044
li x3, 0x4000       # VMEM_CSQRT
li x4, 0x5000       # VMEM_CHALF
li x5, 0x6000       # VMEM_OUT

# --- Scalar setup: DRAM addresses + transfer size ---
li x6, 0            # DRAM_X_H0
li x7, 0x400        # DRAM_X_H1
li x8, 0x800        # DRAM_C044
li x9, 0xC00        # DRAM_CSQRT
li x10, 0x1000      # DRAM_CHALF
li x11, 0x1400      # DRAM_OUT_H0
li x12, 0x1800      # DRAM_OUT_H1
li x13, 1024        # tile bytes

# --- Secondary VMEM offsets for second halves of each pair ---
addi x14, x1, 1024  # x14 = 0x2400 VMEM x_h1
addi x15, x2, 1024  # x15 = 0x3400 VMEM C044 pair
addi x16, x3, 1024  # x16 = 0x4400 VMEM Csqrt pair
addi x17, x4, 1024  # x17 = 0x5400 VMEM Chalf pair

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x6, x13   # x_h0 -> VMEM[x1]
dma.load.ch1 x14, x7, x13  # x_h1 -> VMEM[x14]
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x2, x8, x13   # C044 -> VMEM[x2]
dma.load.ch1 x15, x8, x13  # C044 -> VMEM[x15] (pair)
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x9, x13   # Csqrt -> VMEM[x3]
dma.load.ch1 x16, x9, x13  # Csqrt -> VMEM[x16] (pair)
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x4, x10, x13  # Chalf -> VMEM[x4]
dma.load.ch1 x17, x10, x13 # Chalf -> VMEM[x17] (pair)
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # m0 = x_h0
delay 34
vload m1, 32(x1)            # m1 = x_h1
delay 34
vload m2, 0(x2)             # m2 = 0.044715
delay 34
vload m3, 32(x2)            # m3 = 0.044715 (pair)
delay 34
vload m4, 0(x3)             # m4 = sqrt(2/pi)
delay 34
vload m5, 32(x3)            # m5 = sqrt(2/pi) (pair)
delay 34
vload m6, 0(x4)             # m6 = 0.5
delay 34
vload m7, 32(x4)            # m7 = 0.5 (pair)
delay 34
vli.all m8, 0x3f80               # m8 = 1.0
delay 65
vli.all m9, 0x3f80               # m9 = 1.0 (pair)
delay 65

# --- GELU compute (pair semantics over m0..m7) ---
vmul.bf16 m10, m0, m0       # (m10, m11) = x^2
delay 66
vmul.bf16 m12, m10, m0      # (m12, m13) = x^3
delay 66
vmul.bf16 m14, m12, m2      # (m14, m15) = 0.044715 * x^3
delay 66
vadd.bf16 m16, m0, m14      # (m16, m17) = x + 0.044715*x^3
delay 66
vmul.bf16 m18, m16, m4      # (m18, m19) = sqrt(2/pi) * (x + 0.044715*x^3)
delay 66
vtanh.bf16 m20, m18         # (m20, m21) = tanh(...)
delay 66
vadd.bf16 m22, m8, m20      # (m22, m23) = 1 + tanh(...)
delay 66
vmul.bf16 m24, m0, m6       # (m24, m25) = 0.5 * x
delay 66
vmul.bf16 m26, m24, m22     # (m26, m27) = GELU(x)
delay 66

# --- MRF -> VMEM -> DRAM ---
addi x18, x5, 1024          # x18 = 0x6400 VMEM OUT_H1
vstore m26, 0(x5)
delay 34
vstore m27, 32(x5)
delay 34
dma.store.ch0 x11, x5, x13
dma.store.ch1 x12, x18, x13
dma.wait.ch0
dma.wait.ch1

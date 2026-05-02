# SmolVLA RMS-norm kernel
# y = x * rsqrt(mean(x^2) + eps) on a 32x32 bf16 tile.
# rsqrt implemented as vrecip(vsqrt(x)); no learnable scale.
#
# Scalar register map:
#   VMEM:  x1=X=0x2000  x2=INV_DIM=0x3000  x3=EPS=0x4000  x4=OUT=0x5000
#          x11=X_H1=0x2400  x12=INV_DIM_H1=0x3400  x14=EPS_H1=0x4400
#          x15=OUT_H1=0x5400
#   DRAM:  x5=X_H0=0x0000  x6=X_H1=0x0400  x7=INV_DIM=0x0800  x8=EPS=0x0C00
#          x9=OUT_H0=0x1000  x10=OUT_H1=0x1400
#   x13=1024 (per-half bytes)
#
# MRF register map:
#   (m0, m1)   = X            (m2, m3)   = X^2
#   (m4, m5)   = rowsum(X^2) / inv_rms (reused)
#   (m6, m7)   = inv_dim / Y (reused)   (m8, m9)   = eps
#   (m10, m11) = mean(X^2)   (m12, m13) = mean + eps
#   (m14, m15) = sqrt(mean + eps)

# --- Scalar setup: VMEM addresses ---
lui x1, 0x20002             # x1 = 0x20002000 VMEM_X
lui x2, 0x20003             # x2 = 0x20003000 VMEM_INV_DIM
lui x3, 0x20004             # x3 = 0x20004000 VMEM_EPS
lui x4, 0x20005             # x4 = 0x20005000 VMEM_OUT

# --- Scalar setup: DRAM addresses ---
lui x5, 0x80000             # x5  = 0x80000000 DRAM_X_H0
addi x6, x5, 1024           # x6  = 0x80000400 DRAM_X_H1
addi x7, x6, 1024           # x7  = 0x80000800 DRAM_INV_DIM
addi x8, x7, 1024           # x8  = 0x80000C00 DRAM_EPS
lui x9, 0x80001             # x9  = 0x80001000 DRAM_OUT_H0
addi x10, x9, 1024          # x10 = 0x80001400 DRAM_OUT_H1
addi x13, x0, 1024          # x13 = 1024 (per-half bytes)

# --- Secondary VMEM offsets (second half of each pair) ---
addi x11, x1, 1024          # x11 = 0x2400 VMEM_X_H1
addi x12, x2, 1024          # x12 = 0x3400 VMEM_INV_DIM_H1
addi x14, x3, 1024          # x14 = 0x4400 VMEM_EPS_H1
addi x15, x4, 1024          # x15 = 0x5400 VMEM_OUT_H1

# --- DMA: DRAM -> VMEM ---
dma.config x0
dma.load.ch0 x1, x5, x13   # X_H0
dma.load.ch1 x11, x6, x13  # X_H1
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x2, x7, x13   # INV_DIM h0
dma.load.ch1 x12, x7, x13  # INV_DIM h1 (same DRAM block, broadcast)
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x8, x13   # EPS h0
dma.load.ch1 x14, x8, x13  # EPS h1 (same DRAM block, broadcast)
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)             # X h0
delay 34
vload m1, 32(x1)            # X h1
delay 34
vload m6, 0(x2)             # inv_dim h0
delay 34
vload m7, 32(x2)            # inv_dim h1
delay 34
vload m8, 0(x3)             # eps h0
delay 34
vload m9, 32(x3)            # eps h1
delay 34

# --- Compute: y = x * rsqrt(mean(x^2) + eps) ---
vsquare.bf16 m2, m0         # (m2, m3) = X^2
delay 66
vredsum.row.bf16 m4, m2     # (m4, m5) = rowsum(X^2)
delay 39
vmul.bf16 m10, m4, m6       # (m10, m11) = mean(X^2) = rowsum * inv_dim
delay 66
vadd.bf16 m12, m10, m8      # (m12, m13) = mean + eps
delay 66
vsqrt.bf16 m14, m12         # (m14, m15) = sqrt(mean + eps)
delay 66
vrecip.bf16 m4, m14         # (m4, m5) = inv_rms = 1/sqrt (reuses pair 4/5)
delay 66
vmul.bf16 m6, m0, m4        # (m6, m7) = Y = X * inv_rms (reuses pair 6/7)
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m6, 0(x4)
delay 34
vstore m7, 32(x4)
delay 34
dma.store.ch0 x9, x4, x13
dma.store.ch1 x10, x15, x13
dma.wait.ch0
dma.wait.ch1

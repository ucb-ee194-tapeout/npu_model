# SmolVLA matmul K-chain kernel
# C[32,32] = A[32,0:32] @ B[0:32,32]   (reset acc)
#           + A[32,32:64] @ B[32:64,32] (accumulate)
#
# Scalar register map:
#   DRAM:  x1=A_K0=0x0000  x2=A_K1=0x0400  x3=B_K0=0x0800
#          x4=B_K1=0x0C00  x5=OUT=0x1000
#   VMEM:  x6=A_K0=0x0000  x7=A_K1=0x0400  x8=B_K0=0x0800
#          x9=B_K1=0x0C00  x10=OUT_H0=0x1000  x11=OUT_H1=0x1400
#   Sizes: x12=1024 (fp8 tile)  x13=2048 (bf16 output)

# --- Scalar setup: DRAM addresses ---
lui x1, 0x80000  # x1 = 0x0000 DRAM_A_K0
addi x2, x1, 1024           # x2 = 0x0400 DRAM_A_K1
addi x3, x2, 1024           # x3 = 0x0800 DRAM_B_K0
addi x4, x3, 1024           # x4 = 0x0C00 DRAM_B_K1
addi x5, x4, 1024           # x5 = 0x1000 DRAM_OUT

# --- Scalar setup: VMEM addresses ---
addi x6, x0, 0              # x6  = 0x0000 VMEM_A_K0
addi x7, x6, 1024           # x7  = 0x0400 VMEM_A_K1
addi x8, x7, 1024           # x8  = 0x0800 VMEM_B_K0
addi x9, x8, 1024           # x9  = 0x0C00 VMEM_B_K1
addi x10, x9, 1024          # x10 = 0x1000 VMEM_OUT_H0
addi x11, x10, 1024         # x11 = 0x1400 VMEM_OUT_H1

# --- Transfer sizes ---
addi x12, x0, 1024          # x12 = 1024 (fp8 tile bytes)
addi x13, x12, 1024         # x13 = 2048 (bf16 output bytes)

# --- DMA: DRAM -> VMEM ---
dma.config x0

dma.wait.ch0
dma.load.ch0 x6, x1, x12   # A_K0
dma.load.ch1 x8, x3, x12   # B_K0
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x7, x2, x12   # A_K1
dma.load.ch1 x9, x4, x12   # B_K1
dma.wait.ch0
dma.wait.ch1

# --- K tile 0: acc[0] = A_K0 @ B_K0 (reset) ---
vload m0, 0(x6)             # A_K0
delay 34
vload m1, 0(x8)             # B_K0
delay 34
vmatpush.weight.mxu0 w0, m1
delay 34
vmatmul.mxu0 acc0, m0, w0
delay 96

# --- K tile 1: acc[0] += A_K1 @ B_K1 (accumulate) ---
vload m2, 0(x7)             # A_K1
delay 34
vload m3, 0(x9)             # B_K1
delay 34
vmatpush.weight.mxu0 w0, m3
delay 32
vmatmul.acc.mxu0 acc0, m2, w0
delay 96

# --- Pop accumulator, store halves to VMEM, DMA out ---
vmatpop.bf16.acc.mxu0 m4, acc0     # (m4, m5) = C bf16
delay 32
vstore m4, 0(x10)           # OUT_H0
delay 34
vstore m5, 0(x11)           # OUT_H1
delay 34
dma.store.ch0 x5, x10, x13
dma.wait.ch0
ecall

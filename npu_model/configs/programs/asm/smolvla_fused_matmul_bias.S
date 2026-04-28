# SmolVLA fused_matmul_bias kernel
# (A_fp8 @ B_fp8)_bf16 + bias_bf16 on 32x32 tiles.
#
# Scalar register map:
#   x1 = VMEM_A   = 0x2000    x2 = VMEM_B    = 0x2400
#   x3 = VMEM_BIAS = 0x2800   x4 = VMEM_OUT  = 0x3000
#   x5 = DRAM_A   = 0x0000    x6 = DRAM_B    = 0x0400
#   x7 = DRAM_BIAS = 0x0800   x8 = DRAM_OUT  = 0x1000
#   x9 = 1024 (fp8 tile)      x10 = 2048 (bf16 tile)
#
# MRF register map:
#   m0        = A fp8      m2      = B fp8
#   (m4, m5)  = bias bf16  (m6, m7)  = matmul result bf16
#   (m8, m9)  = out bf16

# --- Scalar setup: VMEM addresses ---
lui x1, 0x2                 # x1 = 0x2000 VMEM_A
lui x2, 0x2                 # x2 = 0x2000
addi x2, x2, 1024           # x2 = 0x2400 VMEM_B
lui x3, 0x3                 # x3 = 0x3000
addi x3, x3, -2048          # x3 = 0x2800 VMEM_BIAS
lui x4, 0x3                 # x4 = 0x3000 VMEM_OUT

# --- Scalar setup: DRAM addresses + transfer sizes ---
addi x5, x0, 0              # x5 = 0x0000 DRAM_A
addi x6, x0, 1024           # x6 = 0x0400 DRAM_B
lui x7, 0x1                 # x7 = 0x1000
addi x7, x7, -2048          # x7 = 0x0800 DRAM_BIAS
lui x8, 0x1                 # x8 = 0x1000 DRAM_OUT
addi x9, x0, 1024           # x9 = 1024 (fp8 tile bytes)
lui x10, 0x1                # x10 = 0x1000
addi x10, x10, -2048        # x10 = 2048 (bf16 tile bytes)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x5, x9    # A fp8
dma.load.ch1 x2, x6, x9    # B fp8
dma.wait.ch0
dma.wait.ch1
dma.load.ch0 x3, x7, x10   # bias bf16
dma.wait.ch0

# --- VMEM -> MRF ---
vload m0, 0(x1)             # A fp8
delay 34
vload m2, 0(x2)             # B fp8
delay 34
vload m4, 0(x3)             # bias h0
delay 34
vload m5, 32(x3)            # bias h1
delay 34

# --- op_a: fp8 matmul via MXU ---
vmatpush.weight.mxu0 w0, m2
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m6, acc0     # (m6, m7) = C bf16
delay 32

# --- op_b: bias add ---
vadd.bf16 m8, m6, m4                # (m8, m9) = C + bias
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m8, 0(x4)
delay 34
vstore m9, 32(x4)
delay 34
dma.store.ch0 x8, x4, x10
dma.wait.ch0

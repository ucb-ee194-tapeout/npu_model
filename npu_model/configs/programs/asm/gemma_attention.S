# Gemma attention kernel (simplified, single-head)
# Scaled dot-product attention: softmax(Q @ K^T * scale) @ V
#   - vmatmul.mxu0 for Q @ K
#   - vexp, vredsum.row, vrecip, vmul to implement softmax
#
# Register map:
#   x1  = VMEM_QUERY_BASE  = 0x2000
#   x2  = VMEM_KEY_BASE    = 0x2400
#   x4  = VMEM_SCALE_BASE  = 0x2800
#   x5  = VMEM_OUTPUT_BASE = 0x3000
#   x6  = DRAM_QUERY_BASE  = 0x0000
#   x7  = DRAM_KEY_BASE    = 0x0400
#   x9  = DRAM_SCALE_BASE  = 0x0800
#   x10 = DRAM_OUTPUT_BASE = 0x1000
#   x11 = 1024  (fp8 tile bytes)
#   x12 = 2048  (bf16 tile bytes)

# --- Register Setup (VMEM addresses) ---
li x1, 0x2000           # VMEM_QUERY_BASE
li x2, 0x2400           # VMEM_KEY_BASE
li x4, 0x2800           # VMEM_SCALE_BASE
li x5, 0x3000           # VMEM_OUTPUT_BASE

# --- Register Setup (DRAM addresses) ---
addi x6, x0, 0          # DRAM_QUERY_BASE = 0x0000
addi x7, x0, 1024       # DRAM_KEY_BASE = 0x0400
li x9, 0x0800           # DRAM_SCALE_BASE
li x10, 0x1000          # DRAM_OUTPUT_BASE
# Byte lengths: fp8 tile (1024) and bf16 tile (2048)
addi x11, x0, 1024
li x12, 2048

# --- DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x6, x11   # Q tile
dma.load.ch1 x2, x7, x11   # K tile
dma.load.ch2 x4, x9, x12   # scale (bf16 tile)
dma.wait.ch0
dma.wait.ch1
dma.wait.ch2

# --- VMEM -> MRF ---
vload m0, 0(x1)     # Q (fp8 tile)
delay 34
vload m1, 0(x2)     # K (fp8 tile)
delay 34
vload m2, 0(x4)     # scale low half
delay 34
vload m3, 32(x4)    # scale high half
delay 34

# --- Push K to WB slot 0, compute scores = Q @ K, pop bf16 ---
vmatpush.weight.mxu0 w0, m1
delay 32
vmatmul.mxu0 acc0, m0, w0
delay 96
vmatpop.bf16.acc.mxu0 m4, acc0     # scores -> m4/m5
delay 32

# scores_scaled = scores * scale
vmul.bf16 m6, m4, m2
delay 66

# --- Softmax (unnormalized variant: no max subtraction) ---
# exp_scores = exp(scores_scaled)
vexp.bf16 m8, m6
delay 66
# row_sum = sum(exp_scores) broadcast across columns
vredsum.row.bf16 m10, m8
delay 39
# inv_row_sum = 1 / row_sum
vrecip.bf16 m12, m10
delay 66
# softmax_scores = exp_scores * inv_row_sum
vmul.bf16 m14, m8, m12
delay 66

# --- Store softmax scores (bf16 tile) ---
vstore m14, 0(x5)
delay 34
vstore m15, 32(x5)
delay 34
dma.store.ch0 x10, x5, x12
dma.wait.ch0

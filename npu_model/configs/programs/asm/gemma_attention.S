# Gemma attention kernel (simplified, single-head)
# Scaled dot-product attention: softmax(Q @ K^T * scale) @ V
#   - vmatmul.mxu0 for Q @ K
#   - vexp, vredsum.row, vrecip, vmul to implement softmax
#
# Register map:
#   x1 = VMEM_QUERY_BASE = 0x800 (word address of byte 0x2000)
#   x2 = VMEM_KEY_BASE = 0x900 (word address of byte 0x2400)
#   x4 = VMEM_SCALE_BASE = 0xa00 (word address of byte 0x2800)
#   x5 = VMEM_OUTPUT_BASE = 0xc00 (word address of byte 0x3000)
#   x6  = DRAM_QUERY_BASE  = 0x0000
#   x7  = DRAM_KEY_BASE    = 0x0400
#   x9  = DRAM_SCALE_BASE  = 0x0800
#   x10 = DRAM_OUTPUT_BASE = 0x1000
#   x11 = 1024  (fp8 tile bytes)
#   x12 = 2048  (bf16 tile bytes)

# --- Register Setup (VMEM addresses) ---
li x1, 0x800  # VMEM_QUERY_BASE
li x2, 0x900  # VMEM_KEY_BASE
li x4, 0xa00  # VMEM_SCALE_BASE
li x5, 0xc00  # VMEM_OUTPUT_BASE

# --- Register Setup (DRAM addresses) ---
addi x6, x0, 0          # DRAM_QUERY_BASE = 0x0000
addi x7, x0, 1024       # DRAM_KEY_BASE = 0x0400
li x9, 0x0800           # DRAM_SCALE_BASE
li x10, 0x1000          # DRAM_OUTPUT_BASE
# Byte lengths: fp8 tile (1024) and bf16 tile (2048)
addi x11, x0, 1024
li x12, 2048

# --- DRAM -> VMEM ---
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x6, x11, 0   # Q tile
dma.load x2, x7, x11, 1   # K tile
dma.load x4, x9, x12, 2   # scale (bf16 tile)
dma.wait 0
dma.wait 1
dma.wait 2

# --- VMEM -> MRF ---
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0     # Q (fp8 tile)
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 1, x31, 0     # K (fp8 tile)
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vload 2, x31, 0     # scale low half
delay 34
addi x31, x4, 0  # DMA and VLS bases both count words
vload 3, x31, 8    # scale high half
delay 34

# --- Push K to WB slot 0, compute scores = Q @ K, pop bf16 ---
vtrpose.xlu 1, 1   # MXU weights are streamed as columns
delay 66
vmatpush.w.mxu0 0, 1
delay 32
vmatmul.mxu0 0, 0, 0
delay 96
vmatpop.bf16.mxu0 4, 0     # scores -> m4/m5
delay 32

# scores_scaled = scores * scale
vmul.bf16 6, 4, 2
delay 66

# --- Softmax (unnormalized variant: no max subtraction) ---
# exp_scores = exp(scores_scaled)
vexp 8, 6
delay 66
# row_sum = sum(exp_scores) broadcast across columns
vredsum.row.bf16 10, 8
delay 39
# inv_row_sum = 1 / row_sum
vrecip.bf16 12, 10
delay 66
# softmax_scores = exp_scores * inv_row_sum
vmul.bf16 14, 8, 12
delay 66

# --- Store softmax scores (bf16 tile) ---
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 14, x31, 0
delay 34
addi x31, x5, 0  # DMA and VLS bases both count words
vstore 15, x31, 8
delay 34
dma.store x10, x5, x12, 0
dma.wait 0

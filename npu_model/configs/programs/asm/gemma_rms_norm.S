# Gemma RMS norm kernel
# RMS norm: x * rsqrt(mean(x^2) + eps)
#
# Register map:
#   x1 = VMEM_INPUT_BASE = 0x800 (word address of byte 0x2000)
#   x2 = VMEM_EPS_BASE = 0xa00 (word address of byte 0x2800)
#   x3 = VMEM_OUTPUT_BASE = 0xc00 (word address of byte 0x3000)
#   x4 = DRAM_INPUT_BASE  = 0x0000
#   x5 = DRAM_EPS_BASE    = 0x0800
#   x6 = DRAM_OUTPUT_BASE = 0x1000
#   x7 = 2048  (bf16 tile bytes)

# --- Register Setup ---
li x1, 0x800  # VMEM_INPUT_BASE
li x2, 0xa00  # VMEM_EPS_BASE
li x3, 0xc00  # VMEM_OUTPUT_BASE
addi x4, x0, 0     # DRAM_INPUT_BASE = 0x0000
li x5, 0x800        # DRAM_EPS_BASE
li x6, 0x1000       # DRAM_OUTPUT_BASE
li x7, 2048         # bf16 tile bytes

# --- DRAM -> VMEM ---
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config x30, 0
dma.wait 0
dma.load x1, x4, x7, 0    # input
dma.load x2, x5, x7, 1    # eps
dma.wait 0
dma.wait 1

# --- VMEM -> MRF ---
addi x31, x1, 0  # DMA and VLS bases both count words
vload 0, x31, 0     # x low half
delay 34
addi x31, x1, 0  # DMA and VLS bases both count words
vload 1, x31, 8    # x high half
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 2, x31, 0     # eps low
delay 34
addi x31, x2, 0  # DMA and VLS bases both count words
vload 3, x31, 8    # eps high
delay 34

# --- RMS Norm Compute ---
# x_sq = x * x
vsquare.bf16 4, 0
delay 66
# sum_sq over columns, broadcast back across each row
vredsum.row.bf16 6, 4
delay 39
# mean_sq = sum_sq * (1/ROW_SIZE)
vli.all 8, 0x4200
delay 65
vrecip.bf16 10, 8
delay 66
vmul.bf16 12, 6, 10
delay 66
# var_eps = mean_sq + eps
vadd.bf16 14, 12, 2
delay 66
# rsqrt = 1 / sqrt(var_eps)
vsqrt 16, 14
delay 66
vrecip.bf16 18, 16
delay 66
# output = x * rsqrt
vmul.bf16 20, 0, 18
delay 66

# --- MRF -> VMEM -> DRAM ---
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 20, x31, 0
delay 34
addi x31, x3, 0  # DMA and VLS bases both count words
vstore 21, x31, 8
delay 34
dma.store x6, x3, x7, 0
dma.wait 0

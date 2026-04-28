# Gemma RMS norm kernel
# RMS norm: x * rsqrt(mean(x^2) + eps)
#
# Register map:
#   x1 = VMEM_INPUT_BASE  = 0x2000
#   x2 = VMEM_EPS_BASE    = 0x2800
#   x3 = VMEM_OUTPUT_BASE = 0x3000
#   x4 = DRAM_INPUT_BASE  = 0x0000
#   x5 = DRAM_EPS_BASE    = 0x0800
#   x6 = DRAM_OUTPUT_BASE = 0x1000
#   x7 = 2048  (bf16 tile bytes)

# --- Register Setup ---
li x1, 0x2000       # VMEM_INPUT_BASE
li x2, 0x2800       # VMEM_EPS_BASE
li x3, 0x3000       # VMEM_OUTPUT_BASE
addi x4, x0, 0     # DRAM_INPUT_BASE = 0x0000
li x5, 0x800        # DRAM_EPS_BASE
li x6, 0x1000       # DRAM_OUTPUT_BASE
li x7, 2048         # bf16 tile bytes

# --- DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x4, x7    # input
dma.load.ch1 x2, x5, x7    # eps
dma.wait.ch0
dma.wait.ch1

# --- VMEM -> MRF ---
vload m0, 0(x1)     # x low half
delay 34
vload m1, 32(x1)    # x high half
delay 34
vload m2, 0(x2)     # eps low
delay 34
vload m3, 32(x2)    # eps high
delay 34

# --- RMS Norm Compute ---
# x_sq = x * x
vsquare.bf16 m4, m0
delay 66
# sum_sq over columns, broadcast back across each row
vredsum.row.bf16 m6, m4
delay 39
# mean_sq = sum_sq * (1/ROW_SIZE)
vli.all m8, 0x4200
delay 65
vli.all m9, 0x4200
delay 65
vrecip.bf16 m10, m8
delay 66
vmul.bf16 m12, m6, m10
delay 66
# var_eps = mean_sq + eps
vadd.bf16 m14, m12, m2
delay 66
# rsqrt = 1 / sqrt(var_eps)
vsqrt.bf16 m16, m14
delay 66
vrecip.bf16 m18, m16
delay 66
# output = x * rsqrt
vmul.bf16 m20, m0, m18
delay 66

# --- MRF -> VMEM -> DRAM ---
vstore m20, 0(x3)
delay 34
vstore m21, 32(x3)
delay 34
dma.store.ch0 x6, x3, x7
dma.wait.ch0

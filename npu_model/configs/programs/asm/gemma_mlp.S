# Gemma MLP kernel (simplified)
# Gate and up projections, then elementwise gate*up (simplified GeGLU)
#
# Register map:
#   x1  = VMEM_GATE_WEIGHT_BASE  = 0x2000
#   x2  = VMEM_UP_WEIGHT_BASE    = 0x2400
#   x3  = VMEM_ACTIVATION_BASE   = 0x2800
#   x4  = VMEM_OUTPUT_BASE       = 0x2C00
#   x5  = DRAM_GATE_WEIGHT_BASE  = 0x0000
#   x6  = DRAM_UP_WEIGHT_BASE    = 0x0400
#   x7  = DRAM_ACTIVATION_BASE   = 0x0800
#   x8  = DRAM_OUTPUT_BASE       = 0x0C00
#   x9  = 1024  (fp8 tile bytes)
#   x10 = 2048  (bf16 tile bytes)

# --- Register Setup (VMEM addresses) ---
lui x1, 0x20002         # VMEM_GATE_WEIGHT_BASE = 0x20002000
addi x2, x1, 1024       # VMEM_UP_WEIGHT_BASE = 0x20002400
addi x3, x2, 1024       # VMEM_ACTIVATION_BASE = 0x20002800
addi x4, x3, 1024       # VMEM_OUTPUT_BASE = 0x20002C00

# --- Register Setup (DRAM addresses) ---
lui x5, 0x80000         # DRAM_GATE_WEIGHT_BASE = 0x80000000
addi x6, x5, 1024       # DRAM_UP_WEIGHT_BASE = 0x80000400
addi x7, x6, 1024       # DRAM_ACTIVATION_BASE = 0x80000800
addi x8, x7, 1024       # DRAM_OUTPUT_BASE = 0x80000C00
# Byte lengths
addi x9, x0, 1024       # fp8 tile bytes  (32*32*1)
li x10, 2048            # bf16 tile bytes (32*32*2)

# --- DRAM -> VMEM ---
dma.config x0
dma.load.ch0 x1, x5, x9    # gate weight tile
dma.load.ch1 x2, x6, x9    # up weight tile
dma.load.ch2 x3, x7, x9    # activation tile
dma.wait.ch0
dma.wait.ch1
dma.wait.ch2

# --- VMEM -> MRF (weights + activation) ---
vload m0, 0(x1)     # gate W (fp8)
delay 34
vload m1, 0(x2)     # up W (fp8)
delay 34
vload m2, 0(x3)     # act (fp8)
delay 34

# --- Push weights to MXU0 WB slots 0 and 1 ---
vmatpush.weight.mxu0 w0, m0
delay 32
vmatpush.weight.mxu0 w1, m1
delay 32

# --- Matrix Multiplications ---
# Gate projection: activation @ gate_weight -> acc0
vmatmul.mxu0 acc0, m2, w0
delay 96
vmatpop.bf16.acc.mxu0 m4, acc0     # gate -> m4/m5
delay 32
# Up projection: activation @ up_weight -> acc0
vmatmul.mxu0 acc0, m2, w1
delay 96
vmatpop.bf16.acc.mxu0 m6, acc0     # up -> m6/m7
delay 32

# --- Element-wise Multiplication (GeGLU simplified: gate * up) ---
vmul.bf16 m8, m4, m6
delay 66

# --- Store Results ---
vstore m8, 0(x4)
delay 34
vstore m9, 32(x4)
delay 34
dma.store.ch0 x8, x4, x10
dma.wait.ch0

# --- 1. Setup Scalar Registers ---
lui x1, 0x80000     # x1 = 0x80000000 DRAM_A
addi x2, x1, 1024   # x2 = 0x80000400 DRAM_B
lui x3, 0x20002     # x3 = 0x20002000 VMEM_A
addi x4, x3, 1024   # x4 = 0x20002400 VMEM_B
addi x5, x0, 1024   # x5 = 1024 (size)

# --- 2. Configure DMA Channels ---
dma.config x0

# --- 3. Load thing ---
dma.load.ch0 x3, x1, x5    # VMEM[x3] <- DRAM[x1], size=x5
dma.load.ch1 x4, x2, x5    # VMEM[x4] <- DRAM[x2], size=x5
# Wait to get these things in VMEM
dma.wait.ch0
dma.wait.ch1
# Move VMEM data to actual computational registers
vload m1, 0(x3)
delay 34
vload m0, 0(x4)
delay 34
# Push Temporary MRF 1 -> MXU0 Weight Buffer 1
vmatpush.weight.mxu0 w0, m0
delay 32

# --- 4. Do unnecessary loads (Overlapped with Matmul) ---
# We do not need to reconfigure the DMA channels if the base addresses are unchanged
# Issue DMA loads again (DRAM -> VMEM)
dma.load.ch0 x3, x1, x5
dma.load.ch1 x4, x2, x5
delay 16

# --- 5. Do matmul ---
vmatmul.mxu0 acc0, m1, w0
delay 96 # TODO - verify delays
# FIXME: I'm like, not at all sure how erf is supposed to work with exponent register
# the temp fix is to just assume that we need to set it to 1 manually.
seli e0, 1 
vmatpop.fp8.acc.mxu0 m0, acc0, e0
# delay 32
# Wait to finish unnecessary loads
dma.wait.ch0
dma.wait.ch1
# End delay
delay 0

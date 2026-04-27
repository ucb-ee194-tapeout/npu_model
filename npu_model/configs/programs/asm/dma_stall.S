# --- 1. Setup Scalar Registers ---
# Set x2 = 1024 (Base Address & Size 1024)
addi x1, x0, 0
addi x2, x0, 1024
addi x3, x0, 0
addi x4, x0, 1024

# --- 2. Configure DMA Channels ---
# Configure Base Address x1 (0)
dma.config.ch0 x0
dma.wait.ch0

# --- 3. Load thing ---
# Load 1024 bytes (x2) from DRAM to VMEM(x1) on Channel 0
# Note: rs1 specifies VMEM offset, rs2 specifies length. Base address comes from dma.config
dma.load.ch0 x0, x1, x2
# Load 1024 bytes (x2) from DRAM to VMEM(x2) on Channel 1
dma.load.ch1 x2, x2, x2
# Wait to get these things in VMEM
dma.wait.ch0
dma.wait.ch1
# Move VMEM data to actual computational registers
# vload VMEM(x1=0) -> MRF 2
vload m1, 0(x0)
delay 34
# vload VMEM(x2=1024) -> Temporary MRF 1
vload m0, 0(x2)
delay 34
# Push Temporary MRF 1 -> MXU0 Weight Buffer 1
vmatpush.weight.mxu0 w0, m0
delay 32

# --- 4. Do unnecessary loads (Overlapped with Matmul) ---
# We do not need to reconfigure the DMA channels if the base addresses are unchanged
# Issue DMA loads again (DRAM -> VMEM)
dma.load.ch0 x3, x0, x1
dma.load.ch1 x4, x1, x1
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

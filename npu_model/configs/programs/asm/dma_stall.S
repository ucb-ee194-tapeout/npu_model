# --- 1. Setup Scalar Registers ---
# x2 = 1024 (DRAM offset and size, bytes); DMA VMEM operands count words:
# x4 = x5 = word 256 = byte 1024.
addi x1, x0, 0
addi x2, x0, 1024
addi x3, x0, 0
addi x4, x0, 256
addi x5, x0, 256

# --- 2. Configure DMA Channels ---
# Configure Base Address x1 (0)
addi x30, x0, 1  # dma.base 1: DRAM operands address 4 GiB + offset (Program.dram_base)
dma.config.ch0 x30
dma.wait.ch0

# --- 3. Load thing ---
# Load 1024 bytes (x2) from DRAM offset x1 to VMEM word 0 on Channel 0
# Note: rd is the VMEM word address, rs1 the DRAM offset, rs2 the length. Base address comes from dma.config
dma.load.ch0 x0, x1, x2
# Load 1024 bytes (x2) from DRAM offset x2 to VMEM word x5 (byte 1024) on Channel 1
dma.load.ch1 x5, x2, x2
# Wait to get these things in VMEM
dma.wait.ch0
dma.wait.ch1
# Move VMEM data to actual computational registers
# vload VMEM(x1=0) -> MRF 2
srli x31, x0, 2   # VLS bases count words; DMA bases count bytes
vload m1, 0(x31)
delay 34
# vload VMEM(x2=1024) -> Temporary MRF 1
srli x31, x2, 2   # VLS bases count words; DMA bases count bytes
vload m0, 0(x31)
delay 34
# Push Temporary MRF 1 -> MXU0 Weight Buffer 1
vtrpose.xlu m0, m0   # MXU weights are streamed as columns
delay 66
vmatpush.weight.mxu0 w0, m0
delay 32

# --- 4. Do unnecessary loads (Overlapped with Matmul) ---
# We do not need to reconfigure the DMA channels if the base addresses are unchanged
# Issue DMA loads again (DRAM -> VMEM). Sizes must be non-zero multiples of
# 32 bytes: DmaEngine asserts on a zero-beat command.
dma.load.ch0 x3, x0, x2
dma.load.ch1 x4, x1, x2
delay 16

# --- 5. Do matmul ---
vmatmul.mxu0 acc0, m1, w0
delay 96 # TODO - verify delays
# FIXME: I'm like, not at all sure how erf is supposed to work with exponent register
# the temp fix is to just assume that we need to set it to 1 manually.
seli e0, 127
vmatpop.fp8.acc.mxu0 m0, acc0, e0
# delay 32
# Wait to finish unnecessary loads
dma.wait.ch0
dma.wait.ch1
# End delay
delay 0
